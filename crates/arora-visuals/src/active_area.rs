//! The active area: the part of a view the user can see, when panels cover
//! the rest, so the view frames its subject there rather than in the middle
//! of the whole canvas.
//!
//! A host puts an [`ActiveArea`] on a camera: a rectangle within the camera's
//! viewport, in logical pixels from its top-left corner. The camera then
//! renders as a window onto a view the size of that rectangle, laid so the
//! rectangle's centre is the view's — Studio's `activeBounds`, which it
//! applies with three's `setViewOffset`, and here Bevy's [`SubCameraView`],
//! which takes the same three values.
//!
//! **The rectangle eases rather than jumps**, settling in about a tenth of a
//! second as Studio's spring does, so a panel sliding in carries the subject
//! along with it. [`ActiveAreaShown`] is where the eased rectangle stands this
//! frame, for anything placed relative to it — the view gizmo's corner, say.
//!
//! **It frames; it does not decide gestures.** What happens to a press over a
//! covered part of the canvas is the host's, as it is in Studio, where the
//! panels are DOM elements that take their own presses.
//!
//! Removing [`ActiveArea`] renders the whole view again.

use bevy::camera::{CameraUpdateSystems, SubCameraView};
use bevy::prelude::*;

/// The part of a camera's viewport the user can see ([module docs](self)).
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct ActiveArea(pub Rect);

/// Where a camera's active area stands this frame, as it eases toward its
/// [`ActiveArea`].
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct ActiveAreaShown(pub Rect);

/// How quickly the shown area closes on its target, as a time constant in
/// seconds: about a tenth of a second to settle, as Studio's spring does.
const EASE: f32 = 0.03;

/// Closer than this, in logical pixels, the shown area is the target.
const SETTLED: f32 = 0.01;

/// Where a camera's framing is set — its active area eased, a host's view
/// offset applied — in `PostUpdate`, before Bevy computes the camera's
/// projection from it, so a view is framed where this frame says. Anything
/// that follows a camera's framing, such as an outline's mask camera, orders
/// after it.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct FramingSystems;

pub struct ActiveAreaPlugin;

impl Plugin for ActiveAreaPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            (ease_areas, release_areas)
                .in_set(FramingSystems)
                .before(CameraUpdateSystems),
        );
    }
}

/// The sub-view that frames `area` in a viewport of `viewport`: the whole
/// view is the area's size, and the viewport looks onto it from the area's
/// origin turned back, so the area's centre lands on the view's.
pub fn framing(area: Rect, viewport: Vec2) -> SubCameraView {
    SubCameraView {
        full_size: area.size().round().max(Vec2::ONE).as_uvec2(),
        offset: -area.min,
        size: viewport.round().max(Vec2::ONE).as_uvec2(),
    }
}

/// Eases each camera's shown area toward its target and frames the camera on
/// it, writing only a change.
fn ease_areas(
    time: Res<Time>,
    mut commands: Commands,
    mut cameras: Query<(
        Entity,
        &mut Camera,
        &ActiveArea,
        Option<&mut ActiveAreaShown>,
    )>,
) {
    let step = 1.0 - (-time.delta_secs() / EASE).exp();
    for (entity, mut camera, area, shown) in &mut cameras {
        let Some(viewport) = camera.logical_viewport_size() else {
            continue;
        };
        let target = area.0;
        let now = match shown {
            Some(mut shown) => {
                let eased = Rect {
                    min: shown.0.min.lerp(target.min, step),
                    max: shown.0.max.lerp(target.max, step),
                };
                let settled = eased.min.distance(target.min) < SETTLED
                    && eased.max.distance(target.max) < SETTLED;
                let next = if settled { target } else { eased };
                if shown.0 != next {
                    shown.0 = next;
                }
                next
            }
            // A camera's first area is taken at once: there is nothing on
            // screen yet to carry along.
            None => {
                commands.entity(entity).insert(ActiveAreaShown(target));
                target
            }
        };
        let wanted = Some(framing(now, viewport));
        if camera.sub_camera_view != wanted {
            camera.sub_camera_view = wanted;
        }
    }
}

/// Renders the whole view again on a camera whose active area was removed.
fn release_areas(
    mut commands: Commands,
    mut removed: RemovedComponents<ActiveArea>,
    mut cameras: Query<&mut Camera>,
) {
    for entity in removed.read() {
        if let Ok(mut camera) = cameras.get_mut(entity) {
            camera.sub_camera_view = None;
        }
        if let Ok(mut entity) = commands.get_entity(entity) {
            entity.remove::<ActiveAreaShown>();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An area covering the whole viewport frames it unchanged; one beside a
    /// panel is the whole view, looked onto from its origin turned back — the
    /// values Studio hands `setViewOffset`.
    #[test]
    fn framing_is_studios_view_offset() {
        let viewport = Vec2::new(1280.0, 800.0);
        let whole = framing(Rect::new(0.0, 0.0, 1280.0, 800.0), viewport);
        assert_eq!(whole.full_size, UVec2::new(1280, 800));
        assert_eq!(whole.offset, Vec2::ZERO);
        assert_eq!(whole.size, UVec2::new(1280, 800));

        let beside = framing(Rect::new(0.0, 0.0, 900.0, 800.0), viewport);
        assert_eq!(beside.full_size, UVec2::new(900, 800));
        assert_eq!(beside.offset, Vec2::ZERO);
        assert_eq!(beside.size, UVec2::new(1280, 800));

        let right_of = framing(Rect::new(380.0, 0.0, 1280.0, 800.0), viewport);
        assert_eq!(right_of.full_size, UVec2::new(900, 800));
        assert_eq!(right_of.offset, Vec2::new(-380.0, 0.0));
    }
}
