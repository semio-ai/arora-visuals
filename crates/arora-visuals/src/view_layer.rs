//! Render layers that belong to one view.
//!
//! Two consumers can share a Bevy app: a robot scene and the face on its
//! screen, each drawn by cameras of its own. Their content is kept apart by
//! render layers the consumers choose. What this crate draws *for* a view —
//! a gizmo's handles, the outline's quad, the selection mask — has to be kept
//! to that view as well, or a gizmo used through an editing camera would also
//! be drawn by the camera filling a screen's texture.
//!
//! So a view that needs it is given two layers of its own, [`ViewLayers`]:
//!
//! - **`aids`**, which the view joins, and which what is drawn for it goes on.
//!   No other camera draws it.
//! - **`mask`**, which the view does not join. The outline's mask camera draws
//!   it, and a selected mesh joins it for each view that outlines it, so one
//!   view's mask never holds what another view selected, even where the two
//!   views' content stands in the same place.
//!
//! A module asks for a camera's layers by putting [`WantsViewLayers`] on it,
//! and reads them back from [`ViewLayers`] once given. A view keeps its layers
//! while it lives; they are freed when it is despawned.

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;

use crate::axes::FrameAxes;
use crate::axis_line::AxisLine;
use crate::joint_gizmo::JointGizmo;
use crate::transform_gizmo::TransformGizmo;

/// The first render layer handed to views, two per view upward from here.
///
/// **A host must not draw its own entities on these.** A view joins its
/// `aids` layer, so anything else on it would be drawn into that view, and a
/// mesh on another view's `mask` layer would be outlined there.
pub const VIEW_LAYERS_FROM: usize = 31;

/// On a camera: the render layers that are its alone ([module docs](self)).
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewLayers {
    /// What is drawn for this view goes on this layer, which the view joins.
    pub aids: usize,
    /// The layer this view's selection mask is drawn from. The view does not
    /// join it.
    pub mask: usize,
}

impl ViewLayers {
    /// Whether `layer` is one of these.
    pub fn holds(&self, layer: usize) -> bool {
        layer == self.aids || layer == self.mask
    }
}

/// On a camera: give it its [`ViewLayers`].
#[derive(Component, Default)]
pub struct WantsViewLayers;

/// Gives views their layers, in `PostUpdate`. Anything that reads
/// [`ViewLayers`] to place what it draws runs after it.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ViewLayerSystems;

/// Which views hold which pair of layers: slot `i` is layers
/// `VIEW_LAYERS_FROM + 2i` and `VIEW_LAYERS_FROM + 2i + 1`.
#[derive(Resource, Default)]
struct Held(Vec<Option<Entity>>);

impl Held {
    /// A free pair for `view`, reclaiming any slot whose view is gone.
    fn take(&mut self, view: Entity, alive: impl Fn(Entity) -> bool) -> ViewLayers {
        let slot = match self
            .0
            .iter()
            .position(|held| held.is_none_or(|holder| !alive(holder)))
        {
            Some(slot) => slot,
            None => {
                self.0.push(None);
                self.0.len() - 1
            }
        };
        self.0[slot] = Some(view);
        ViewLayers {
            aids: VIEW_LAYERS_FROM + 2 * slot,
            mask: VIEW_LAYERS_FROM + 2 * slot + 1,
        }
    }
}

pub struct ViewLayerPlugin;

/// Marks the plugin installed, so the modules that need it can each add it.
#[derive(Resource, Default)]
struct Installed;

impl Plugin for ViewLayerPlugin {
    fn is_unique(&self) -> bool {
        false
    }

    fn build(&self, app: &mut App) {
        if app.world().contains_resource::<Installed>() {
            return;
        }
        app.init_resource::<Installed>()
            .init_resource::<Held>()
            .add_systems(
                PostUpdate,
                (give_layers, keep_joined).chain().in_set(ViewLayerSystems),
            )
            // Last, so every part an aid built this frame, wherever in the
            // frame it was built, is on its layer before the frame is drawn:
            // a camera first seen this frame is asked for and given its
            // layers here too, so nothing new draws a frame in every view.
            .add_systems(Last, (ask_for_aid_layers, give_layers, layer_aids).chain());
    }
}

/// Every gizmo, frame and axis line, with the camera it is seen through.
/// One without a camera is drawn in every view, as before there were views
/// to keep it to.
#[derive(bevy::ecs::system::SystemParam)]
struct Aids<'w, 's> {
    gizmos: Query<'w, 's, (Entity, &'static TransformGizmo)>,
    joints: Query<'w, 's, (Entity, &'static JointGizmo)>,
    frames: Query<'w, 's, (Entity, &'static FrameAxes)>,
    lines: Query<'w, 's, (Entity, &'static AxisLine)>,
}

impl Aids<'_, '_> {
    fn seen(&self) -> impl Iterator<Item = (Entity, Entity)> + '_ {
        let gizmos = self
            .gizmos
            .iter()
            .map(|(aid, gizmo)| (aid, Some(gizmo.camera)));
        let joints = self
            .joints
            .iter()
            .map(|(aid, joint)| (aid, Some(joint.camera)));
        let frames = self.frames.iter().map(|(aid, frame)| (aid, frame.camera));
        let lines = self.lines.iter().map(|(aid, line)| (aid, line.camera));
        gizmos
            .chain(joints)
            .chain(frames)
            .chain(lines)
            .filter_map(|(aid, camera)| camera.map(|camera| (aid, camera)))
    }
}

/// Asks for the layers of every camera an aid is seen through.
#[allow(clippy::type_complexity)]
fn ask_for_aid_layers(
    mut commands: Commands,
    aids: Aids,
    asked: Query<(), Or<(With<ViewLayers>, With<WantsViewLayers>)>>,
) {
    for (_, camera) in aids.seen() {
        if !asked.contains(camera) {
            if let Ok(mut camera) = commands.get_entity(camera) {
                camera.insert(WantsViewLayers);
            }
        }
    }
}

/// Puts each aid on its camera's `aids` layer.
fn layer_aids(
    mut commands: Commands,
    aids: Aids,
    views: Query<&ViewLayers>,
    children: Query<&Children>,
    mut layers: Query<Option<&mut RenderLayers>, With<Mesh3d>>,
) {
    for (aid, camera) in aids.seen() {
        layer_aid(aid, camera, &views, &children, &mut layers, &mut commands);
    }
}

/// Gives each camera asking for them its layers, and joins it to its `aids`.
#[allow(clippy::type_complexity)]
fn give_layers(
    mut commands: Commands,
    wanting: Query<(Entity, Option<&RenderLayers>), (With<WantsViewLayers>, Without<ViewLayers>)>,
    holders: Query<(), Or<(With<ViewLayers>, With<WantsViewLayers>)>>,
    mut held: ResMut<Held>,
) {
    for (view, layers) in &wanting {
        // A view given its layers earlier in this same pass has none yet, but
        // still asks, so it keeps the pair it was given.
        let given = held.take(view, |holder| holders.contains(holder));
        commands
            .entity(view)
            .insert((given, layers.cloned().unwrap_or_default().with(given.aids)));
    }
}

/// Keeps each view on its `aids` layer and off its `mask` layer if the host
/// has since replaced the view's layers.
fn keep_joined(
    mut views: Query<(Entity, &ViewLayers, Option<&mut RenderLayers>)>,
    mut commands: Commands,
) {
    for (view, given, layers) in &mut views {
        match layers {
            Some(mut layers) => {
                if !layers.intersects(&RenderLayers::layer(given.aids))
                    || layers.intersects(&RenderLayers::layer(given.mask))
                {
                    let wanted = layers.clone().with(given.aids).without(given.mask);
                    *layers = wanted;
                }
            }
            None => {
                commands
                    .entity(view)
                    .insert(RenderLayers::default().with(given.aids));
            }
        }
    }
}

/// The layers a view draws content on: all of its own, less the ones this
/// module gave it.
pub fn content_layers(layers: Option<&RenderLayers>, given: Option<&ViewLayers>) -> RenderLayers {
    let all = layers.cloned().unwrap_or_default();
    match given {
        Some(given) => all.without(given.aids).without(given.mask),
        None => all,
    }
}

/// Puts what is drawn for `camera` on its `aids` layer: `entity` and every
/// mesh beneath it. Nothing changes until the camera has its layers.
fn layer_aid(
    entity: Entity,
    camera: Entity,
    views: &Query<&ViewLayers>,
    children: &Query<&Children>,
    layers: &mut Query<Option<&mut RenderLayers>, With<Mesh3d>>,
    commands: &mut Commands,
) {
    let Ok(given) = views.get(camera) else {
        return;
    };
    let wanted = RenderLayers::layer(given.aids);
    let mut place = |entity: Entity| {
        if let Ok(held) = layers.get_mut(entity) {
            match held {
                Some(mut held) => {
                    if *held != wanted {
                        *held = wanted.clone();
                    }
                }
                None => {
                    commands.entity(entity).insert(wanted.clone());
                }
            }
        }
    };
    place(entity);
    for below in children.iter_descendants(entity) {
        place(below);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins(ViewLayerPlugin);
        app
    }

    #[test]
    fn each_view_gets_layers_of_its_own() {
        let mut app = app();
        let one = app.world_mut().spawn(WantsViewLayers).id();
        let two = app
            .world_mut()
            .spawn((WantsViewLayers, RenderLayers::layer(3)))
            .id();
        app.update();
        let layers = |app: &App, view| *app.world().get::<ViewLayers>(view).unwrap();
        let (a, b) = (layers(&app, one), layers(&app, two));
        let all = [a.aids, a.mask, b.aids, b.mask];
        for (i, x) in all.iter().enumerate() {
            assert!(*x >= VIEW_LAYERS_FROM);
            assert!(!all[i + 1..].contains(x), "layer {x} handed out twice");
        }
        assert_eq!(
            app.world().get::<RenderLayers>(two),
            Some(&RenderLayers::layer(3).with(b.aids)),
            "a view keeps its own layers and joins its aids, not its mask",
        );
    }

    #[test]
    fn a_despawned_views_layers_go_to_the_next() {
        let mut app = app();
        let first = app.world_mut().spawn(WantsViewLayers).id();
        app.update();
        let had = *app.world().get::<ViewLayers>(first).unwrap();
        app.world_mut().despawn(first);
        let next = app.world_mut().spawn(WantsViewLayers).id();
        app.update();
        assert_eq!(app.world().get::<ViewLayers>(next), Some(&had));
    }

    #[test]
    fn a_view_whose_layers_are_replaced_rejoins_its_aids() {
        let mut app = app();
        let view = app.world_mut().spawn(WantsViewLayers).id();
        app.update();
        let given = *app.world().get::<ViewLayers>(view).unwrap();
        app.world_mut()
            .entity_mut(view)
            .insert(RenderLayers::from_layers(&[0, given.mask]));
        app.update();
        assert_eq!(
            app.world().get::<RenderLayers>(view),
            Some(&RenderLayers::layer(0).with(given.aids)),
        );
    }

    /// A gizmo's meshes go on its camera's `aids` layer in the frame they
    /// appear, so a gizmo seen through one camera never draws in another.
    #[test]
    fn a_gizmos_meshes_are_on_its_cameras_aids_layer_from_the_first_frame() {
        let mut app = app();
        let camera = app.world_mut().spawn_empty().id();
        let other = app.world_mut().spawn(WantsViewLayers).id();
        let target = app.world_mut().spawn_empty().id();
        let gizmo = app
            .world_mut()
            .spawn(TransformGizmo::new(target, camera))
            .id();
        let part = app
            .world_mut()
            .spawn((Mesh3d(Handle::default()), ChildOf(gizmo)))
            .id();
        let unseen = app
            .world_mut()
            .spawn(FrameAxes::default())
            .with_child(Mesh3d(Handle::default()))
            .id();
        app.update();
        let given = *app.world().get::<ViewLayers>(camera).expect("asked for");
        assert_eq!(
            app.world().get::<RenderLayers>(part),
            Some(&RenderLayers::layer(given.aids)),
        );
        let theirs = *app.world().get::<ViewLayers>(other).unwrap();
        assert_ne!(theirs.aids, given.aids);
        let unseen_mesh = app.world().get::<Children>(unseen).unwrap()[0];
        assert!(
            app.world().get::<RenderLayers>(unseen_mesh).is_none(),
            "a frame seen through no camera is left in every view",
        );
    }

    #[test]
    fn adding_the_plugin_twice_is_harmless() {
        let mut app = app();
        app.add_plugins(ViewLayerPlugin);
        let view = app.world_mut().spawn(WantsViewLayers).id();
        app.update();
        assert!(app.world().get::<ViewLayers>(view).is_some());
    }
}
