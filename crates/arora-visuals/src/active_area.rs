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
//! along with it. Each ease is timed from the frame its target is first seen,
//! not from the frame before: a host that draws only on change hands the first
//! frame after an idle stretch the whole stretch as its delta, and counting it
//! would land a change set after idling at once. [`ActiveAreaShown`] is where
//! the eased rectangle stands this frame, for anything placed relative to it —
//! the view gizmo's corner, say.
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

/// An ease under way on a camera: the rectangle it set out from, the target
/// it heads for, and when it set out. Kept apart from [`ActiveAreaShown`] so
/// each frame is placed from the ease's start rather than stepped from the
/// last frame by that frame's delta, which after an idle stretch is the whole
/// stretch.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
struct AreaEase {
    from: Rect,
    to: Rect,
    /// In seconds since startup, as `f64` so an ease keeps its timing in a
    /// session left open for days.
    since: f64,
}

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
    /// Both consumers sharing an app may add it; it installs once.
    fn is_unique(&self) -> bool {
        false
    }

    fn build(&self, app: &mut App) {
        if !crate::installed::first_install::<Self>(app) {
            return;
        }
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

/// A camera with an active area, where its area stands, and any ease under
/// way.
type FramedCamera<'a> = (
    Entity,
    &'a mut Camera,
    &'a ActiveArea,
    Option<&'a mut ActiveAreaShown>,
    Option<&'a mut AreaEase>,
);

/// Eases each camera's shown area toward its target and frames the camera on
/// it, writing only a change.
fn ease_areas(time: Res<Time>, mut commands: Commands, mut cameras: Query<FramedCamera>) {
    let elapsed = time.elapsed_secs_f64();
    for (entity, mut camera, area, shown, ease) in &mut cameras {
        let Some(viewport) = camera.logical_viewport_size() else {
            continue;
        };
        let target = area.0;
        let now = match shown {
            Some(mut shown) => {
                let ease = match ease {
                    Some(ease) if ease.to == target => Some(*ease),
                    // A new target sets out from where the area stands, timed
                    // from this frame, so the first frame of the ease does not
                    // move the area however long the scene sat idle before it.
                    _ if shown.0 != target => {
                        let fresh = AreaEase {
                            from: shown.0,
                            to: target,
                            since: elapsed,
                        };
                        match ease {
                            Some(mut ease) => *ease = fresh,
                            None => {
                                commands.entity(entity).insert(fresh);
                            }
                        }
                        Some(fresh)
                    }
                    // Back at its target before a stale ease settled.
                    Some(_) => {
                        commands.entity(entity).remove::<AreaEase>();
                        None
                    }
                    None => None,
                };
                let next = match ease {
                    Some(ease) => {
                        let progress = 1.0 - (-((elapsed - ease.since) as f32) / EASE).exp();
                        let eased = Rect {
                            min: ease.from.min.lerp(target.min, progress),
                            max: ease.from.max.lerp(target.max, progress),
                        };
                        let settled = eased.min.distance(target.min) < SETTLED
                            && eased.max.distance(target.max) < SETTLED;
                        if settled {
                            commands.entity(entity).remove::<AreaEase>();
                            target
                        } else {
                            eased
                        }
                    }
                    None => target,
                };
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
            entity.remove::<(ActiveAreaShown, AreaEase)>();
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

    /// A camera's first area is taken at once; a new one set after the scene
    /// sat idle — its first frame carrying the whole idle stretch as its
    /// delta, as a host drawing only on change hands it — still sets out from
    /// where the area stood and eases over the frames that follow.
    #[test]
    fn an_area_set_after_idling_eases() {
        use bevy::camera::{ComputedCameraValues, RenderTargetInfo};
        use bevy::ecs::system::RunSystemOnce;
        use std::time::Duration;

        let mut world = World::new();
        world.init_resource::<Time>();
        let camera = world
            .spawn((
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: UVec2::new(1280, 800),
                            scale_factor: 1.0,
                        }),
                        ..default()
                    },
                    ..default()
                },
                ActiveArea(Rect::new(0.0, 0.0, 1280.0, 800.0)),
            ))
            .id();
        let frame = |world: &mut World, delta: Duration| {
            world.resource_mut::<Time>().advance_by(delta);
            world.run_system_once(ease_areas).unwrap();
            world.get::<ActiveAreaShown>(camera).unwrap().0
        };
        let nominal = Duration::from_secs_f32(1.0 / 60.0);

        let first = frame(&mut world, nominal);
        assert_eq!(first, Rect::new(0.0, 0.0, 1280.0, 800.0), "taken at once");

        let beside = Rect::new(0.0, 0.0, 900.0, 800.0);
        world.get_mut::<ActiveArea>(camera).unwrap().0 = beside;
        let mut widths = vec![frame(&mut world, Duration::from_millis(250)).max.x];
        for _ in 0..3 {
            widths.push(frame(&mut world, nominal).max.x);
        }
        // The long frame leaves most of the way still to go, rather than
        // landing within a fraction of a pixel of the target.
        assert!(
            widths[0] > 1090.0,
            "{widths:?}: the frame after idling moves less than halfway"
        );
        assert!(
            widths.windows(2).all(|w| w[1] < w[0]) && widths[3] > 901.0,
            "{widths:?}: the area closes on its target over several frames"
        );
        for _ in 0..30 {
            frame(&mut world, nominal);
        }
        assert_eq!(frame(&mut world, nominal), beside, "and settles on it");
    }
}
