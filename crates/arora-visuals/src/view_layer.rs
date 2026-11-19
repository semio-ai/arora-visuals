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

use bevy::camera::visibility::VisibilitySystems;
use bevy::camera::CameraUpdateSystems;

use crate::axes::{AxesOf, FrameAxes, FrameAxesSystems};
use crate::axis_line::{AxisLine, AxisLineSystems, Half};
use crate::joint_gizmo::{JointGizmo, JointPart};
use crate::transform_gizmo::{GizmoPart, Part, TransformGizmo, TransformGizmoSystems};

/// A set of one layer. Bevy's `RenderLayers::layer` holds only layers below
/// 64, and views are given layers from 31 up without a ceiling.
pub fn only(layer: usize) -> RenderLayers {
    RenderLayers::none().with(layer)
}

/// The first render layer handed to views, two per view upward from here.
///
/// **A host must not draw its own entities on these.** A view joins its
/// `aids` layer, so anything else on it would be drawn into that view, and a
/// mesh on another view's `mask` layer would be outlined there.
pub const VIEW_LAYERS_FROM: usize = 31;

/// Whether `layer` is some view's `mask` layer, given or not.
pub fn is_mask(layer: usize) -> bool {
    layer >= VIEW_LAYERS_FROM && (layer - VIEW_LAYERS_FROM) % 2 == 1
}

/// `layers` less every view's `mask` layer: what a mesh is drawn on, apart
/// from the masks the outline has joined it to.
pub fn without_masks(layers: &RenderLayers) -> RenderLayers {
    layers.iter().filter(|layer| !is_mask(*layer)).collect()
}

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
pub(crate) struct Held(Vec<Option<Entity>>);

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

impl Plugin for ViewLayerPlugin {
    fn is_unique(&self) -> bool {
        false
    }

    fn build(&self, app: &mut App) {
        if !crate::installed::first_install::<Self>(app) {
            return;
        }
        // After every aid has placed its parts this frame, and after Bevy
        // has computed this frame's projections, which the aids size
        // themselves by; before Bevy decides what each camera sees, so a
        // gizmo moved to another camera leaves the first one's picture in the
        // same frame. A camera first seen this frame is asked for and given
        // its layers here too.
        app.init_resource::<Held>().add_systems(
            PostUpdate,
            (ask_for_aid_layers, give_layers, keep_joined, layer_aids)
                .chain()
                .in_set(ViewLayerSystems)
                .after(TransformGizmoSystems)
                .after(AxisLineSystems)
                .after(FrameAxesSystems)
                .after(CameraUpdateSystems)
                .before(VisibilitySystems::CheckVisibility),
        );
    }
}

/// Every gizmo, frame and axis line, with the camera it is seen through.
#[derive(bevy::ecs::system::SystemParam)]
struct Aids<'w, 's> {
    gizmos: Query<'w, 's, (Entity, &'static TransformGizmo)>,
    joints: Query<'w, 's, (Entity, &'static JointGizmo)>,
    frames: Query<'w, 's, (Entity, &'static FrameAxes)>,
    lines: Query<'w, 's, (Entity, &'static AxisLine)>,
}

impl Aids<'_, '_> {
    /// Each aid and the camera it is seen through, if it names one.
    fn all(&self) -> impl Iterator<Item = (Entity, Option<Entity>)> + '_ {
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
        gizmos.chain(joints).chain(frames).chain(lines)
    }
}

/// Asks for the layers of every camera an aid is seen through.
#[allow(clippy::type_complexity)]
fn ask_for_aid_layers(
    mut commands: Commands,
    aids: Aids,
    asked: Query<(), Or<(With<ViewLayers>, With<WantsViewLayers>)>>,
) {
    for camera in aids.all().filter_map(|(_, camera)| camera) {
        if !asked.contains(camera) {
            if let Ok(mut camera) = commands.get_entity(camera) {
                camera.insert(WantsViewLayers);
            }
        }
    }
}

/// Puts each aid's parts where they belong. A part belongs to the aid that
/// built it, told by the mark each kind of aid puts on its parts and found
/// above the part — never to the geometry an aid stands on, which a frame's
/// axes sit among, and never to another aid sharing an entity with it, such
/// as a joint's gizmo and its frame. It goes:
///
/// - for an aid seen through a camera, on that camera's `aids` layer;
/// - for one seen through a camera that is gone, on no layer at all. The
///   camera's layer is handed to the next view, which would otherwise draw
///   the orphan;
/// - for one seen through no camera, on the default layer, drawn in every
///   view that draws it, as before there were views to keep aids to.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn layer_aids(
    mut commands: Commands,
    aids: Aids,
    kinds: PartKinds,
    views: Query<&ViewLayers>,
    alive: Query<()>,
    parents: Query<&ChildOf>,
    mut parts: Query<(Entity, Option<&mut RenderLayers>), (With<Mesh3d>, With<GizmoPart>)>,
) {
    for (part, held) in &mut parts {
        let Some(camera) = kinds.camera_of(part, &aids, &parents) else {
            continue;
        };
        let wanted = match camera {
            None => RenderLayers::default(),
            Some(camera) => match views.get(camera) {
                Ok(given) => only(given.aids),
                Err(_) if !alive.contains(camera) => RenderLayers::none(),
                // Every live camera an aid names was asked for and given its
                // layers earlier in this pass, so this is not reached; a
                // part is left as it is rather than guessed at.
                Err(_) => continue,
            },
        };
        match held {
            Some(mut held) => {
                if *held != wanted {
                    *held = wanted;
                }
            }
            None => {
                if wanted != RenderLayers::default() {
                    commands.entity(part).insert(wanted);
                }
            }
        }
    }
}

/// The marks each kind of aid puts on the parts it builds.
#[derive(bevy::ecs::system::SystemParam)]
struct PartKinds<'w, 's> {
    handles: Query<'w, 's, (), With<Part>>,
    joint_pieces: Query<'w, 's, (), With<JointPart>>,
    halves: Query<'w, 's, (), With<Half>>,
    arrows_of: Query<'w, 's, &'static AxesOf>,
}

impl PartKinds<'_, '_> {
    /// The camera the aid that built `part` is seen through — `Some(None)`
    /// for an aid that names none — or `None` for a part no aid above it
    /// built.
    fn camera_of(
        &self,
        part: Entity,
        aids: &Aids,
        parents: &Query<&ChildOf>,
    ) -> Option<Option<Entity>> {
        let mut at = part;
        loop {
            at = parents.get(at).ok()?.parent();
            if self.handles.contains(part) {
                if let Ok((_, gizmo)) = aids.gizmos.get(at) {
                    return Some(Some(gizmo.camera));
                }
            } else if self.joint_pieces.contains(part) {
                if let Ok((_, joint)) = aids.joints.get(at) {
                    return Some(Some(joint.camera));
                }
            } else if self.halves.contains(part) {
                if let Ok((_, line)) = aids.lines.get(at) {
                    return Some(line.camera);
                }
            } else if let Ok(AxesOf(frame)) = self.arrows_of.get(at) {
                return aids.frames.get(*frame).ok().map(|(_, frame)| frame.camera);
            }
        }
    }
}

/// Gives each camera asking for them its layers. Joining them is
/// [`keep_joined`]'s, which runs next.
#[allow(clippy::type_complexity)]
pub(crate) fn give_layers(
    mut commands: Commands,
    wanting: Query<Entity, (With<WantsViewLayers>, Without<ViewLayers>)>,
    holders: Query<(), Or<(With<ViewLayers>, With<WantsViewLayers>)>>,
    mut held: ResMut<Held>,
) {
    for view in &wanting {
        // A view given its layers earlier in this same pass has none yet, but
        // still asks, so it keeps the pair it was given.
        let given = held.take(view, |holder| holders.contains(holder));
        commands.entity(view).insert(given);
    }
}

/// Keeps each view on its `aids` layer and off its `mask` layer, whatever the
/// host has since set the view's layers to, or taken them away.
pub(crate) fn keep_joined(
    mut views: Query<(Entity, &ViewLayers, Option<&mut RenderLayers>)>,
    mut commands: Commands,
) {
    for (view, given, layers) in &mut views {
        match layers {
            Some(mut layers) => {
                if !layers.intersects(&only(given.aids)) || layers.intersects(&only(given.mask)) {
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

    /// Whatever a host sets a view's layers to, the view stays on its `aids`
    /// and off its `mask`: each is checked on its own, and a view whose
    /// layers are taken away gets the default ones and its `aids` back.
    #[test]
    fn a_view_whose_layers_are_replaced_rejoins_its_aids() {
        let mut app = app();
        let view = app.world_mut().spawn(WantsViewLayers).id();
        app.update();
        let given = *app.world().get::<ViewLayers>(view).unwrap();
        let after = |app: &mut App, set: Option<RenderLayers>| {
            match set {
                Some(layers) => app.world_mut().entity_mut(view).insert(layers),
                None => app.world_mut().entity_mut(view).remove::<RenderLayers>(),
            };
            app.update();
            app.world().get::<RenderLayers>(view).cloned()
        };
        let joined = Some(RenderLayers::layer(0).with(given.aids));
        assert_eq!(
            after(&mut app, Some(RenderLayers::layer(0))),
            joined,
            "aids dropped"
        );
        assert_eq!(
            after(
                &mut app,
                Some(RenderLayers::layer(0).with(given.aids).with(given.mask))
            ),
            joined,
            "mask joined",
        );
        assert_eq!(after(&mut app, None), joined, "layers taken away");
    }

    /// The kinds of part an aid builds, each with the mark its aid gives it.
    #[derive(Clone, Copy)]
    enum Kind {
        Handle,
        JointPiece,
        Half,
        Arrow,
    }

    /// A part of `kind` built for `aid`, placed as its aid places it: under
    /// the aid, or for a frame's arrows under the holder beneath it.
    fn part(app: &mut App, aid: Entity, kind: Kind) -> Entity {
        let world = app.world_mut();
        let mesh = (Mesh3d(Handle::default()), GizmoPart);
        match kind {
            Kind::Handle => world.spawn((mesh, Part::stub(), ChildOf(aid))).id(),
            Kind::JointPiece => world.spawn((mesh, JointPart::stub(), ChildOf(aid))).id(),
            Kind::Half => world.spawn((mesh, Half::stub(), ChildOf(aid))).id(),
            Kind::Arrow => {
                let holder = world.spawn((AxesOf(aid), ChildOf(aid))).id();
                world.spawn((mesh, ChildOf(holder))).id()
            }
        }
    }

    fn joint(target: Entity, camera: Entity) -> JointGizmo {
        JointGizmo::new(
            target,
            camera,
            crate::joint_gizmo::JointMotion::Turn,
            Dir3::Z,
            -1.0,
            1.0,
            0.0,
            Color::WHITE,
        )
    }

    fn frame(camera: Option<Entity>) -> FrameAxes {
        FrameAxes {
            camera,
            ..default()
        }
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
        let handle = part(&mut app, gizmo, Kind::Handle);
        let unseen = app.world_mut().spawn(frame(None)).id();
        let unseen_arrow = part(&mut app, unseen, Kind::Arrow);
        app.update();
        let given = *app.world().get::<ViewLayers>(camera).expect("asked for");
        assert_eq!(
            app.world().get::<RenderLayers>(handle),
            Some(&only(given.aids))
        );
        let theirs = *app.world().get::<ViewLayers>(other).unwrap();
        assert_ne!(theirs.aids, given.aids);
        assert!(
            app.world().get::<RenderLayers>(unseen_arrow).is_none(),
            "a frame seen through no camera is left in every view",
        );
    }

    /// A frame's axes stand on the body they show, among its own meshes.
    /// Only the arrows go on the camera's layer: the body keeps its layers,
    /// and a frame further down, seen through another camera, keeps its own.
    #[test]
    fn a_frame_layers_its_arrows_and_not_the_body_it_stands_on() {
        let mut app = app();
        let [near, far] = [(); 2].map(|_| app.world_mut().spawn_empty().id());
        let body = app
            .world_mut()
            .spawn((Mesh3d(Handle::default()), frame(Some(near))))
            .id();
        let arrow = part(&mut app, body, Kind::Arrow);
        let limb = app
            .world_mut()
            .spawn((
                Mesh3d(Handle::default()),
                RenderLayers::layer(4),
                ChildOf(body),
                frame(Some(far)),
            ))
            .id();
        let limb_arrow = part(&mut app, limb, Kind::Arrow);
        app.update();
        let aids = |view| app.world().get::<ViewLayers>(view).unwrap().aids;
        let layers = |entity| app.world().get::<RenderLayers>(entity).cloned();
        assert_eq!(layers(arrow), Some(only(aids(near))));
        assert_eq!(layers(limb_arrow), Some(only(aids(far))));
        assert_eq!(layers(body), None, "the body is left on the default layer");
        assert_eq!(layers(limb), Some(RenderLayers::layer(4)));
    }

    /// Views are given layers past 63, which Bevy's `RenderLayers::layer`
    /// refuses; every view still gets its pair, and its aids reach them.
    #[test]
    fn views_past_the_sixty_fourth_layer_still_get_theirs() {
        let mut app = app();
        let target = app.world_mut().spawn_empty().id();
        let views: Vec<Entity> = (0..20)
            .map(|_| app.world_mut().spawn(WantsViewLayers).id())
            .collect();
        let last = *views.last().unwrap();
        let gizmo = app
            .world_mut()
            .spawn(TransformGizmo::new(target, last))
            .id();
        let handle = part(&mut app, gizmo, Kind::Handle);
        app.update();
        let given = *app.world().get::<ViewLayers>(last).unwrap();
        assert!(given.mask > 64, "{given:?}");
        assert_eq!(
            app.world().get::<RenderLayers>(handle),
            Some(&only(given.aids))
        );
        assert!(app
            .world()
            .get::<RenderLayers>(last)
            .unwrap()
            .intersects(&only(given.aids)));
    }

    /// Joint gizmos and axis lines are kept to their camera as transform
    /// gizmos and frames are.
    #[test]
    fn joint_gizmos_and_axis_lines_are_kept_to_their_camera() {
        use crate::transform_gizmo::GizmoSize;

        let mut app = app();
        let camera = app.world_mut().spawn_empty().id();
        let target = app.world_mut().spawn_empty().id();
        let joint_gizmo = app.world_mut().spawn(joint(target, camera)).id();
        let piece = part(&mut app, joint_gizmo, Kind::JointPiece);
        let line = app
            .world_mut()
            .spawn(AxisLine {
                direction: Dir3::X,
                reach: 1.0,
                both_ways: false,
                size: GizmoSize::World(1.0),
                camera: Some(camera),
                colour: Color::WHITE,
            })
            .id();
        let half = part(&mut app, line, Kind::Half);
        app.update();
        let wanted = Some(only(app.world().get::<ViewLayers>(camera).unwrap().aids));
        for part in [piece, half] {
            assert_eq!(app.world().get::<RenderLayers>(part).cloned(), wanted);
        }
    }

    /// A gizmo whose camera is gone draws nowhere: the camera's layer goes
    /// to the next view, which must not draw the gizmo left behind. One that
    /// stops naming a camera is drawn in every view again.
    #[test]
    fn a_gizmo_whose_camera_goes_draws_nowhere_and_one_with_none_everywhere() {
        let mut app = app();
        let camera = app.world_mut().spawn_empty().id();
        let target = app.world_mut().spawn_empty().id();
        let gizmo = app
            .world_mut()
            .spawn(TransformGizmo::new(target, camera))
            .id();
        let handle = part(&mut app, gizmo, Kind::Handle);
        let shown = app.world_mut().spawn(frame(Some(camera))).id();
        let arrow = part(&mut app, shown, Kind::Arrow);
        app.update();
        let had = *app.world().get::<ViewLayers>(camera).unwrap();
        app.world_mut().despawn(camera);
        let next = app.world_mut().spawn(WantsViewLayers).id();
        app.world_mut().get_mut::<FrameAxes>(shown).unwrap().camera = None;
        app.update();
        assert_eq!(
            app.world().get::<ViewLayers>(next),
            Some(&had),
            "the layers moved on"
        );
        assert_eq!(
            app.world().get::<RenderLayers>(handle),
            Some(&RenderLayers::none()),
        );
        assert_eq!(
            app.world().get::<RenderLayers>(arrow),
            Some(&RenderLayers::default()),
        );
    }

    /// Aids standing on one entity keep their parts to their own cameras:
    /// a joint's gizmo seen through one, its frame through another, and a
    /// transform gizmo and a frame naming none beside them.
    #[test]
    fn aids_sharing_an_entity_keep_to_their_own_cameras() {
        let mut app = app();
        let [one, two] = [(); 2].map(|_| app.world_mut().spawn_empty().id());
        let target = app.world_mut().spawn_empty().id();
        let both = app
            .world_mut()
            .spawn((joint(target, one), frame(Some(two))))
            .id();
        let piece = part(&mut app, both, Kind::JointPiece);
        let arrow = part(&mut app, both, Kind::Arrow);
        let pair = app
            .world_mut()
            .spawn((TransformGizmo::new(target, one), frame(None)))
            .id();
        let handle = part(&mut app, pair, Kind::Handle);
        let unseen_arrow = part(&mut app, pair, Kind::Arrow);
        app.update();
        let aids = |view| app.world().get::<ViewLayers>(view).unwrap().aids;
        let layers = |entity| app.world().get::<RenderLayers>(entity).cloned();
        assert_eq!(layers(piece), Some(only(aids(one))));
        assert_eq!(layers(arrow), Some(only(aids(two))));
        assert_eq!(layers(handle), Some(only(aids(one))), "not in every view");
        assert_eq!(layers(unseen_arrow), None);
    }

    /// Added twice, the plugin still gives each view one pair of layers:
    /// built twice, it would give each view two, one pair wasted.
    #[test]
    fn adding_the_plugin_twice_is_harmless() {
        let mut app = app();
        app.add_plugins(ViewLayerPlugin);
        let [one, two] = [(); 2].map(|_| app.world_mut().spawn(WantsViewLayers).id());
        app.update();
        let mut given: Vec<usize> = [one, two]
            .map(|view| app.world().get::<ViewLayers>(view).unwrap().aids)
            .to_vec();
        given.sort();
        assert_eq!(given, vec![VIEW_LAYERS_FROM, VIEW_LAYERS_FROM + 2]);
    }
}
