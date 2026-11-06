//! The transform gizmo: handles for moving, turning and scaling an entity by
//! hand.
//!
//! **It is a thing of its own, not a property of what it moves.** A host
//! spawns a [`TransformGizmo`], points it at a target and a camera, and styles
//! it — colours, size, an offset from the target — so the same gizmo serves an
//! instance, a light, a camera, or anything else with a transform.
//!
//! **Any of the three modes, at once or one at a time.** Every enabled mode's
//! handles show together, laid out as Studio's pivot controls are: an arrow
//! along each axis and a square in each plane to move, a ring about each axis
//! to turn, and a ball beyond each arrow to scale.
//!
//! **The moving and scaling handles stay on the side facing the viewer.** Each
//! arrow points along its axis or against it, whichever way is toward the
//! camera, and the squares and balls go with the arrows. Otherwise, as the
//! view comes round, an arrow ends up behind the rings and the centre, and
//! taking it means reaching through them. An arrow seen nearly end-on keeps
//! the side it had until the view is clearly past it, so it does not flicker
//! between the two, and nothing changes side while a drag is under way. A host that wants one mode
//! at a time sets [`TransformGizmo::only`] — from whatever key it binds — and
//! the others hide, so the choice of keys is the host's.
//!
//! **It looks like the joint controls.** Each handle is in its axis's colour,
//! slim and translucent at rest and thicker and brighter in hand. Rest and
//! in-hand are two meshes and two materials per handle, swapped rather than
//! animated.
//!
//! **It is drawn over everything, as Studio's is.** A gizmo usually stands at
//! its target's origin, which is often inside the target — a robot's is at its
//! feet, within its base — so a handle tested against depth would be hidden
//! by the very thing it moves. A depth bias does not answer that: on a
//! floating-point depth buffer it scales with depth, so it nudges rather than
//! lifts. The handles' material skips the depth test instead.
//!
//! **What is under the pointer is decided on screen, by measurement.** Each
//! handle has a shape on screen — a segment, a ring, a square, a disc — and
//! the pointer's distance from it in pixels decides which one is hot, so a
//! handle is as easy to grab whether it is near or far, and no mesh picking is
//! needed. The drag then works in the world: the pointer's ray is met with the
//! plane that handle moves in, so the handle stays under the pointer whatever
//! the camera does.
//!
//! **A scale ball stands on a stem when the arrows are not showing.** With
//! moving offered and shown, a ball sits beyond its arrow's head; without
//! it, a ball would float apart from the centre it scales about, so a plain
//! line runs out to it. The stem is part of the ball — it thickens when the
//! ball is in hand — but only the ball can be taken.
//!
//! **Any combination of the three modes can be offered**, and changing the
//! combination on a gizmo rebuilds its handles.
//!
//! **A scale ball stretches the target along its own axis nearest the
//! ball's.** The balls sit with the arrows, in the gizmo's axes; in local
//! space those are the target's own and the stretch is exact, but a turned
//! object cannot be stretched along a world axis, so in world space the ball
//! takes the target's axis that lies closest to it.
//!
//! A drag reports itself as [`GizmoDrag`] — start, each move, end — carrying
//! the target's new local transform; with [`TransformGizmo::apply`] the gizmo
//! also writes it. [`GizmoPointer`] says whether the pointer is on a handle or
//! holding one, which a host reads to keep its camera still, and which
//! picking reads so that grabbing a handle does not also select what is
//! behind it.

use bevy::asset::{embedded_asset, embedded_path, AssetPath};
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::picking::events::PointerPress;
use bevy::picking::pointer::PointerButton;
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, CompareFunction, RenderPipelineDescriptor, ShaderType,
    SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::axis_line::{AxisLine, AxisLinePlugin, AxisLineSystems};
use crate::joint_gizmo::{
    distance as joint_distance, dragged, reach as joint_reach, take, JointDrag, JointGizmo,
    JointHeld, JointReach, Rest,
};
use crate::pick::on_the_scene;

/// What a gizmo's handles can do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransformMode {
    Translate,
    Rotate,
    Scale,
}

/// Which modes a gizmo offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransformModes {
    pub translate: bool,
    pub rotate: bool,
    pub scale: bool,
}

impl TransformModes {
    pub const ALL: Self = Self {
        translate: true,
        rotate: true,
        scale: true,
    };

    pub fn has(&self, mode: TransformMode) -> bool {
        match mode {
            TransformMode::Translate => self.translate,
            TransformMode::Rotate => self.rotate,
            TransformMode::Scale => self.scale,
        }
    }
}

/// Whose axes the moving and turning handles follow.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GizmoSpace {
    #[default]
    World,
    /// The target's own, as it is turned.
    Local,
}

/// How big a gizmo is: the reach of its arrows, or a joint gizmo's radius
/// or length. Every gizmo takes one, and the two kinds are a gizmo's whole
/// sizing choice: this is drei `PivotControls`' `fixed` and `scale` as one
/// value, so a size can never be read in the other kind's unit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GizmoSize {
    /// The same on screen however near or far, in logical pixels.
    Pixels(f32),
    /// Fixed in the world, in metres, so it shrinks with distance as what it
    /// moves does.
    World(f32),
}

/// A transform gizmo ([module docs](self)).
#[derive(Component, Clone, Debug)]
#[require(Transform, Visibility)]
pub struct TransformGizmo {
    /// What the gizmo moves.
    pub target: Entity,
    /// The camera it is used through: its view decides what is under the
    /// pointer and how big a pixel is.
    pub camera: Entity,
    pub modes: TransformModes,
    /// Show only this mode's handles, if it is one the gizmo offers. A host
    /// sets it from a key held or a tool chosen.
    pub only: Option<TransformMode>,
    pub space: GizmoSpace,
    pub size: GizmoSize,
    /// Where the gizmo stands and how it is turned, relative to the target:
    /// its position is the offset's translation in the target's own frame,
    /// and its axes are turned by the offset's rotation — after the target's
    /// own rotation in local space, by itself in world space. Turning a target
    /// about an offset gizmo swings it round the gizmo.
    pub offset: Transform,
    /// The directions the gizmo calls X, Y and Z, in the space it works in:
    /// Bevy's own by default, or [`TransformGizmo::Z_UP`] for a Z-up world.
    /// They must be a right-handed set.
    pub axes: [Dir3; 3],
    /// The X, Y and Z handles' colours.
    pub colours: [Color; 3],
    /// Write each drag onto the target. Without it the gizmo only reports.
    pub apply: bool,
    /// While a handle is held, show the dashed line along each axis it
    /// constrains, through the gizmo's origin and both ways: a move, turn
    /// or stretch along an axis shows that axis; a plane shows its two.
    pub guides: bool,
    /// What the host calls this gizmo: the name it goes by in the
    /// [overlay](crate::overlay), where an entity id would mean nothing. Any
    /// string the host chooses — the blackboard key the gizmo writes through,
    /// say — since this crate reads nothing into it.
    pub key: String,
}

impl TransformGizmo {
    /// A gizmo on `target`, seen through `camera`, with every mode, in world
    /// space, reaching a quarter of a metre, in Studio's axis colours.
    ///
    /// Sized in the world, as Studio's `PivotControls` are by default and the
    /// joint gizmo is, so every gizmo in a scene answers a zoom the same way.
    pub fn new(target: Entity, camera: Entity) -> Self {
        Self {
            target,
            camera,
            modes: TransformModes::ALL,
            only: None,
            space: GizmoSpace::World,
            size: GIZMO_SIZE,
            offset: Transform::IDENTITY,
            axes: [Dir3::X, Dir3::Y, Dir3::Z],
            colours: AXIS_COLOURS,
            apply: true,
            guides: true,
            key: String::new(),
        }
    }

    /// The axes of a Z-up world as Bevy's Y-up one holds it: Z is Bevy's up,
    /// and Y is Bevy's −Z, so the three still form a right-handed set.
    pub const Z_UP: [Dir3; 3] = [Dir3::X, Dir3::NEG_Z, Dir3::Y];

    /// The turn that carries Bevy's X, Y and Z onto the directions the gizmo
    /// calls by those names. Folding it into the gizmo's frame is what lets
    /// every handle be laid out along Bevy's axes and still point the way its
    /// label says.
    fn labels(&self) -> Quat {
        let [x, y, z] = self.axes;
        debug_assert!(
            x.cross(*y).abs_diff_eq(*z, 1e-4),
            "a gizmo's axes must be a right-handed set"
        );
        Quat::from_mat3(&Mat3::from_cols(*x, *y, *z))
    }

    fn shows(&self, mode: TransformMode) -> bool {
        self.modes.has(mode) && self.only.is_none_or(|only| only == mode)
    }
}

/// One handle of a gizmo: a transform gizmo's, by what it does and about
/// which axis — a plane's axis being the one it is square to — or a joint
/// gizmo's ring or bar, which is its only handle. `Joint` is here so that
/// [`GizmoPointer`] can name either kind; the transform gizmo's own code never
/// meets it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum GizmoHandle {
    Arrow(usize),
    Plane(usize),
    Ring(usize),
    Ball(usize),
    Joint,
}

impl GizmoHandle {
    /// The transform mode a handle belongs to; a joint's belongs to none.
    fn mode(self) -> Option<TransformMode> {
        match self {
            GizmoHandle::Arrow(_) | GizmoHandle::Plane(_) => Some(TransformMode::Translate),
            GizmoHandle::Ring(_) => Some(TransformMode::Rotate),
            GizmoHandle::Ball(_) => Some(TransformMode::Scale),
            GizmoHandle::Joint => None,
        }
    }

    fn axis(self) -> usize {
        match self {
            GizmoHandle::Arrow(axis)
            | GizmoHandle::Plane(axis)
            | GizmoHandle::Ring(axis)
            | GizmoHandle::Ball(axis) => axis,
            GizmoHandle::Joint => 0,
        }
    }

    const ALL: [GizmoHandle; 12] = [
        GizmoHandle::Arrow(0),
        GizmoHandle::Arrow(1),
        GizmoHandle::Arrow(2),
        GizmoHandle::Plane(0),
        GizmoHandle::Plane(1),
        GizmoHandle::Plane(2),
        GizmoHandle::Ring(0),
        GizmoHandle::Ring(1),
        GizmoHandle::Ring(2),
        GizmoHandle::Ball(0),
        GizmoHandle::Ball(1),
        GizmoHandle::Ball(2),
    ];
}

/// Where a drag is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DragPhase {
    Start,
    Move,
    End,
}

/// A drag on a gizmo's handle, triggered on the gizmo.
#[derive(EntityEvent, Clone, Copy, Debug)]
pub struct GizmoDrag {
    /// The gizmo.
    pub entity: Entity,
    pub target: Entity,
    pub handle: GizmoHandle,
    pub phase: DragPhase,
    /// The target's local transform as the drag has it now.
    pub transform: Transform,
}

/// What the pointer is doing with the gizmos.
#[derive(Resource, Default, Debug)]
pub struct GizmoPointer {
    /// The handle under the pointer, and its gizmo.
    pub hovered: Option<(Entity, GizmoHandle)>,
    /// The handle being dragged, and its gizmo.
    pub dragging: Option<(Entity, GizmoHandle)>,
}

impl GizmoPointer {
    /// Whether the pointer belongs to a gizmo: a press now would grab a
    /// handle, or one is held.
    pub fn engaged(&self) -> bool {
        self.hovered.is_some() || self.dragging.is_some()
    }
}

/// Marks every mesh a gizmo draws, so a host can leave them out of anything
/// that measures the scene.
#[derive(Component, Clone, Copy, Debug)]
pub struct GizmoPart;

// Proportions, in units of the gizmo's size, from Studio's pivot controls.
pub(crate) const SHAFT_LENGTH: f32 = 0.8;
pub(crate) const CONE_LENGTH: f32 = 0.2;
pub(crate) const CONE_RADIUS: f32 = 0.05;
/// A ring's radius, in units of the gizmo's size; public so a host can place
/// things — a label, say — clear of a ring.
pub const RING_RADIUS: f32 = 0.65;
const PLANE_OFFSET: f32 = 0.243;
const PLANE_SIDE: f32 = 0.225;
const BALL_REACH: f32 = 1.2;
const BALL_RADIUS: f32 = 0.044;
/// An arrow is grabbed along this stretch of it, short of the centre, where
/// the planes and the other arrows crowd.
const ARROW_FROM: f32 = 0.25;
/// How far from a handle on screen the pointer may be and still take it.
pub(crate) const REACH_PX: f32 = 8.0;
/// Below this share of facing the camera a plane is too nearly edge-on to
/// drag in.
const EDGE_ON: f32 = 0.15;
/// How far past side-on an axis has to turn, as a share of facing the camera,
/// before its handles change side.
const FLIP: f32 = 0.1;

/// The radius of an arrow's shaft and of a ring's tube, at rest and in hand:
/// the two are one line drawn two ways, so they are one thickness, and the
/// swap from thin to thick is the joint controls' hover.
pub(crate) const TUBE: [f32; 2] = [0.012, 0.04];
/// How far every handle's surface moves out in hand: the tubes' thickening,
/// so a ball and an arrowhead grow with the lines they sit on rather than by
/// a share of their own size.
const GROW: f32 = TUBE[1] - TUBE[0];
/// How much the plane squares swell in hand. A square has no edge thickness
/// to match the tubes by, so it grows by a share of its side.
const SWELL: [f32; 2] = [1.0, 1.3];
/// Rest and in-hand opacities, for lines and solids and for the planes.
const OPACITY: [f32; 2] = [0.75, 1.0];
const RING_OPACITY: [f32; 2] = [0.5, 1.0];
const PLANE_OPACITY: [f32; 2] = [0.3, 0.6];
/// How far toward white a handle in hand is lightened.
const LIGHTEN: f32 = 0.2;

#[derive(ShaderType, Clone, Debug, Default)]
pub struct HandleUniform {
    pub colour: Vec4,
}

/// A handle's material: one flat colour, never hidden; see `handle.wesl`.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug, Default)]
pub struct HandleMaterial {
    #[uniform(0)]
    pub handle: HandleUniform,
}

impl Material for HandleMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(embedded_path!("handle.wesl")).with_source("embedded"),
        )
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Seen from inside as well as out: a ring or a cone can be.
        descriptor.primitive.cull_mode = None;
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_compare = Some(CompareFunction::Always);
            depth.depth_write_enabled = Some(false);
        }
        Ok(())
    }
}

/// A drawn piece of a handle, with its rest and in-hand looks.
#[derive(Component)]
struct Part {
    handle: GizmoHandle,
    /// Where the piece sits, in gizmo units, laid out about +X before its
    /// handle's axis and facing are applied. Every piece's mesh is built
    /// along its own Y and is laid along that axis.
    offset: Vec3,
    /// A scale ball's stem, shown only while the arrows are not.
    stem: bool,
    meshes: [Handle<Mesh>; 2],
    materials: [Handle<HandleMaterial>; 2],
    /// Which look it shows now.
    hot: bool,
}

/// Which way along each of the gizmo's axes its moving and scaling handles
/// point: `1` along it, `-1` against it, whichever is toward the viewer.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub(crate) struct Facing(Vec3);

/// The settings a gizmo's parts were built from; a change to any of them
/// rebuilds the parts, while `only` just shows and hides them.
#[derive(Component, PartialEq)]
struct Built {
    modes: TransformModes,
    colours: [Color; 3],
}

/// A drag in progress.
#[derive(Clone, Copy)]
struct Drag {
    gizmo: Entity,
    handle: GizmoHandle,
    /// The gizmo's origin and axes when the drag began.
    origin: Vec3,
    axes: Quat,
    /// The target's own scale component a ball stretches.
    stretch_along: Vec3,
    /// Where the pointer's ray met the drag's plane when it began.
    grabbed: Vec3,
    /// The plane the drag works in.
    plane: Vec3,
    start_global: GlobalTransform,
    start_local: Transform,
    /// The transform the drag last reported. A drag that ends where the
    /// pointer cannot be read ends here: with `apply` off the target has not
    /// moved, so its own transform would report the drag undone.
    last: Transform,
}

/// The gizmos' pointer handling in `Update`, and their placement in
/// `PostUpdate`. A host orders anything that shares the pointer — a camera
/// controller above all — after it in `Update`, so that [`GizmoPointer`] is
/// this frame's answer when it reads it; anything reporting where the gizmos
/// stand, the overlay among them, orders after it in `PostUpdate`.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct TransformGizmoSystems;

/// A transform gizmo's size unless a host sets one, and a frame's arrows':
/// a quarter of a metre in the world, so an arrow along a frame and a handle
/// along the same axis reach the same length.
pub const GIZMO_SIZE: GizmoSize = GizmoSize::World(0.25);

/// Studio's axis colours, X, Y and Z, as its gizmos draw them.
pub const AXIS_COLOURS: [Color; 3] = [
    Color::srgb_u8(0xef, 0x44, 0x44),
    Color::srgb_u8(0x22, 0xc5, 0x5e),
    Color::srgb_u8(0x3b, 0x82, 0xf6),
];

/// Registers the handle material once, for whichever of the plugins drawing
/// with it is added first.
pub(crate) fn add_handle_material(app: &mut App) {
    if app.is_plugin_added::<MaterialPlugin<HandleMaterial>>() {
        return;
    }
    embedded_asset!(app, "handle.wesl");
    app.add_plugins(MaterialPlugin::<HandleMaterial>::default());
}

pub struct TransformGizmoPlugin;

impl Plugin for TransformGizmoPlugin {
    /// The joint gizmo installs this one, and a host may too, in any order;
    /// installing it again does nothing.
    fn is_unique(&self) -> bool {
        false
    }

    fn build(&self, app: &mut App) {
        if app.world().contains_resource::<GizmoPointer>() {
            return;
        }
        add_handle_material(app);
        app.add_plugins(AxisLinePlugin);
        app.init_resource::<GizmoPointer>()
            .init_resource::<PressedScene>()
            .add_observer(note_scene_press)
            .add_systems(Update, drive.in_set(TransformGizmoSystems))
            .add_systems(
                PostUpdate,
                (build_parts, place_parts)
                    .chain()
                    .in_set(TransformGizmoSystems)
                    .after(TransformSystems::Propagate)
                    .before(AxisLineSystems),
            );
    }
}

/// Whether this frame's left press landed on the scene, rather than on a
/// panel, a toggle or anything else drawn over it: only such a press takes a
/// handle, so a click on a control over a handle stays the control's.
#[derive(Resource, Default)]
struct PressedScene(bool);

fn note_scene_press(
    press: On<PointerPress>,
    windows: Query<(), With<Window>>,
    mut pressed: ResMut<PressedScene>,
) {
    if press.button == PointerButton::Primary
        && on_the_scene(press.entity, press.original_event_target(), &windows)
    {
        pressed.0 = true;
    }
}

/// A part to spawn: its handle, where it sits, its rest and in-hand meshes
/// and materials, and whether it is a scale ball's stem.
type Piece = (
    GizmoHandle,
    Vec3,
    [Handle<Mesh>; 2],
    [Handle<HandleMaterial>; 2],
    bool,
);

/// Builds the parts of each gizmo that is new or whose look has changed.
fn build_parts(
    mut commands: Commands,
    gizmos: Query<(Entity, &TransformGizmo, Option<&Built>, Option<&Children>)>,
    parts: Query<(), With<Part>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<HandleMaterial>>,
) {
    for (entity, gizmo, built, children) in &gizmos {
        let wanted = Built {
            modes: gizmo.modes,
            colours: gizmo.colours,
        };
        if built == Some(&wanted) {
            continue;
        }
        for child in children.into_iter().flatten() {
            if parts.contains(*child) {
                commands.entity(*child).despawn();
            }
        }
        commands.entity(entity).insert((wanted, Facing(Vec3::ONE)));

        let mut material = |colour: Color, opacity: [f32; 2]| {
            [0, 1].map(|hot| {
                let base = if hot == 1 {
                    colour.mix(&Color::WHITE, LIGHTEN)
                } else {
                    colour
                };
                materials.add(HandleMaterial {
                    handle: HandleUniform {
                        colour: LinearRgba::from(base.with_alpha(opacity[hot])).to_vec4(),
                    },
                })
            })
        };
        let mut pieces: Vec<Piece> = Vec::new();
        for handle in GizmoHandle::ALL {
            if !handle.mode().is_some_and(|mode| gizmo.modes.has(mode)) {
                continue;
            }
            let colour = gizmo.colours[handle.axis()];
            match handle {
                GizmoHandle::Arrow(_) => {
                    let look = material(colour, OPACITY);
                    pieces.push((
                        handle,
                        Vec3::X * SHAFT_LENGTH / 2.0,
                        TUBE.map(|radius| meshes.add(Cylinder::new(radius, SHAFT_LENGTH))),
                        look.clone(),
                        false,
                    ));
                    pieces.push((
                        handle,
                        Vec3::X * SHAFT_LENGTH,
                        // The head keeps its shape in hand, scaled so its rim
                        // moves out as far as the shaft's surface does, and
                        // stands on the shaft's end: its mesh is moved so its
                        // base, not its middle, is at the piece's origin.
                        [0.0, GROW].map(|grow| {
                            let scale = (CONE_RADIUS + grow) / CONE_RADIUS;
                            let height = CONE_LENGTH * scale;
                            meshes.add(
                                Mesh::from(Cone {
                                    radius: CONE_RADIUS * scale,
                                    height,
                                })
                                .translated_by(Vec3::Y * height / 2.0),
                            )
                        }),
                        look,
                        false,
                    ));
                }
                GizmoHandle::Plane(_) => {
                    let side = PLANE_SIDE / 2.0;
                    // A square in the plane square to X, out along the other
                    // two axes from the centre.
                    pieces.push((
                        handle,
                        Vec3::new(0.0, PLANE_OFFSET, PLANE_OFFSET),
                        SWELL.map(|swell| {
                            meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(side * swell)))
                        }),
                        material(colour, PLANE_OPACITY),
                        false,
                    ));
                }
                GizmoHandle::Ring(_) => {
                    pieces.push((
                        handle,
                        Vec3::ZERO,
                        TUBE.map(|tube| {
                            meshes.add(Torus {
                                minor_radius: tube,
                                major_radius: RING_RADIUS,
                            })
                        }),
                        material(colour, RING_OPACITY),
                        false,
                    ));
                }
                // Not one of a transform gizmo's; never among `ALL`.
                GizmoHandle::Joint => {}
                GizmoHandle::Ball(_) => {
                    let look = material(colour, OPACITY);
                    pieces.push((
                        handle,
                        Vec3::X * BALL_REACH,
                        [0.0, GROW].map(|grow| meshes.add(Sphere::new(BALL_RADIUS + grow))),
                        look.clone(),
                        false,
                    ));
                    pieces.push((
                        handle,
                        Vec3::X * BALL_REACH / 2.0,
                        TUBE.map(|radius| meshes.add(Cylinder::new(radius, BALL_REACH))),
                        look,
                        true,
                    ));
                }
            }
        }
        commands.entity(entity).with_children(|gizmo| {
            for (handle, offset, meshes, materials, stem) in pieces {
                gizmo.spawn((
                    Part {
                        handle,
                        offset,
                        stem,
                        meshes: meshes.clone(),
                        materials: materials.clone(),
                        hot: false,
                    },
                    GizmoPart,
                    Mesh3d(meshes[0].clone()),
                    MeshMaterial3d(materials[0].clone()),
                    Transform::default(),
                    NotShadowCaster,
                    NotShadowReceiver,
                    // The parts move every frame with the target and the view;
                    // their bounds are not worth keeping current for culling.
                    NoFrustumCulling,
                ));
            }
        });
    }
}

/// The turn that carries a part laid out about X onto `axis`, keeping the
/// three axes a right-handed set: X to Y takes Y to Z, and X to Z takes Z to
/// X. A plane laid out across Y and Z thereby lands across the other two.
fn onto_axis(axis: usize) -> Quat {
    match axis {
        0 => Quat::IDENTITY,
        1 => Quat::from_mat3(&Mat3::from_cols(Vec3::Y, Vec3::Z, Vec3::X)),
        _ => Quat::from_mat3(&Mat3::from_cols(Vec3::Z, Vec3::X, Vec3::Y)),
    }
}

/// Where a gizmo stands, how its axes are turned, which side of each its
/// handles are on, and how many metres one gizmo unit is.
#[derive(Clone, Copy)]
struct Placement {
    origin: Vec3,
    axes: Quat,
    facing: Vec3,
    unit: f32,
}

impl Placement {
    /// The direction a handle reaches along: its axis, turned toward the
    /// viewer for the handles that face.
    fn reach(&self, handle: GizmoHandle) -> Vec3 {
        let index = handle.axis();
        let axis = self.axes * axis_vector(index);
        match handle {
            GizmoHandle::Arrow(_) | GizmoHandle::Ball(_) => axis * self.facing[index],
            GizmoHandle::Plane(_) | GizmoHandle::Ring(_) | GizmoHandle::Joint => axis,
        }
    }
}

fn placement(
    gizmo: &TransformGizmo,
    facing: Vec3,
    target: &GlobalTransform,
    camera: (&Camera, &GlobalTransform),
) -> Option<Placement> {
    let (_, turned, position) = target.to_scale_rotation_translation();
    let labels = gizmo.labels();
    let axes = match gizmo.space {
        GizmoSpace::World => gizmo.offset.rotation * labels,
        GizmoSpace::Local => turned * gizmo.offset.rotation * labels,
    };
    // In the target's own frame, but not its scale: an offset is a distance
    // in the world, not a share of the target's size.
    let origin = position + turned * gizmo.offset.translation;
    let unit = unit_size(gizmo.size, camera, origin)?;
    Some(Placement {
        origin,
        axes,
        facing,
        unit,
    })
}

/// The points marking how far a gizmo's shown handles reach in the world:
/// its centre, the tip of each axis's handles on the side they face, and
/// each ring all the way round. What the overlay boxes a gizmo by.
pub(crate) fn reach_points(
    gizmo: &TransformGizmo,
    facing: &Facing,
    target: &GlobalTransform,
    camera: (&Camera, &GlobalTransform),
) -> Option<Vec<Vec3>> {
    let place = placement(gizmo, facing.0, target, camera)?;
    let ring = if gizmo.shows(TransformMode::Rotate) {
        RING_RADIUS
    } else {
        0.0
    };
    let tip = if gizmo.shows(TransformMode::Scale) {
        BALL_REACH + BALL_RADIUS
    } else if gizmo.shows(TransformMode::Translate) {
        SHAFT_LENGTH + CONE_LENGTH
    } else {
        0.0
    }
    .max(ring);
    let mut points = vec![place.origin];
    for axis in 0..3 {
        let toward = place.reach(GizmoHandle::Arrow(axis));
        points.push(place.origin + toward * tip * place.unit);
    }
    // A ring seen at an angle bulges past the points where it crosses the
    // axes, so each is sampled around its circle, out to its tube's edge.
    if gizmo.shows(TransformMode::Rotate) {
        let radius = (RING_RADIUS + TUBE[1]) * place.unit;
        for axis in 0..3 {
            let u = place.axes * axis_vector((axis + 1) % 3);
            let v = place.axes * axis_vector((axis + 2) % 3);
            for step in 0..RING_SAMPLES {
                let angle = step as f32 * std::f32::consts::TAU / RING_SAMPLES as f32;
                points.push(place.origin + (u * angle.cos() + v * angle.sin()) * radius);
            }
        }
    }
    Some(points)
}

/// Points around each ring, for a gizmo's box.
const RING_SAMPLES: usize = 24;

/// How many metres one gizmo unit is, for a gizmo of `size` standing at
/// `point`.
pub(crate) fn unit_size(
    size: GizmoSize,
    camera: (&Camera, &GlobalTransform),
    point: Vec3,
) -> Option<f32> {
    match size {
        GizmoSize::World(metres) => Some(metres),
        GizmoSize::Pixels(pixels) => Some(pixels * metres_per_pixel(camera, point)?),
    }
}

/// Which side of each axis faces the viewer, keeping `previous` for an axis
/// too near side-on to say.
fn facing_toward(previous: Vec3, axes: Quat, origin: Vec3, eye: Vec3) -> Vec3 {
    let toward = (eye - origin).normalize_or_zero();
    Vec3::from_array(std::array::from_fn(|index| {
        let along = (axes * axis_vector(index)).dot(toward);
        if along > FLIP {
            1.0
        } else if along < -FLIP {
            -1.0
        } else {
            previous[index]
        }
    }))
}

/// The target's own axis, as a component of its scale, that lies closest to
/// `axis` in the world, for a target turned by `turned`.
fn nearest_component(turned: Quat, axis: Vec3) -> Vec3 {
    let index = (0..3)
        .max_by(|&a, &b| {
            let lean = |i: usize| (turned * axis_vector(i)).dot(axis).abs();
            lean(a).total_cmp(&lean(b))
        })
        .unwrap_or(0);
    axis_vector(index)
}

/// How many metres one logical pixel spans at `point`, from two rays a pixel
/// apart met with the plane through `point` facing the camera. Rays rather
/// than a field of view, so orthographic cameras are measured the same way.
pub(crate) fn metres_per_pixel(
    (camera, at): (&Camera, &GlobalTransform),
    point: Vec3,
) -> Option<f32> {
    let centre = camera.world_to_viewport(at, point).ok()?;
    let facing = at.forward().as_vec3();
    let meet = |pixel: Vec2| {
        let ray = camera.viewport_to_world(at, pixel).ok()?;
        meet_plane(ray, facing, point)
    };
    Some(meet(centre)?.distance(meet(centre + Vec2::X)?))
}

/// Where a ray meets the plane through `origin` square to `normal`.
pub(crate) fn meet_plane(ray: Ray3d, normal: Vec3, origin: Vec3) -> Option<Vec3> {
    let along = ray.direction.dot(normal);
    if along.abs() < f32::EPSILON {
        return None;
    }
    let distance = (origin - ray.origin).dot(normal) / along;
    Some(ray.origin + *ray.direction * distance)
}

/// A plane containing `axis` and turned as far toward the viewer as it can
/// be: dragging along an axis is meeting the pointer's ray with this.
pub(crate) fn plane_along(ray: Ray3d, axis: Vec3) -> Vec3 {
    let across = ray.direction.cross(axis);
    if across.length_squared() < f32::EPSILON {
        return axis.any_orthonormal_vector();
    }
    axis.cross(across.normalize()).normalize()
}

pub(crate) fn segment_distance(point: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let t = ((point - a).dot(ab) / ab.length_squared().max(f32::EPSILON)).clamp(0.0, 1.0);
    point.distance(a + ab * t)
}

/// How far the pointer is from a handle on screen, in pixels; zero inside a
/// plane's square or a ball's disc.
fn distance_to(
    handle: GizmoHandle,
    place: &Placement,
    pointer: Vec2,
    ray: Ray3d,
    (camera, at): (&Camera, &GlobalTransform),
) -> Option<f32> {
    let axis = place.reach(handle);
    let to_screen = |world: Vec3| camera.world_to_viewport(at, world).ok();
    match handle {
        GizmoHandle::Arrow(_) => {
            let from = to_screen(place.origin + axis * ARROW_FROM * place.unit)?;
            let to = to_screen(place.origin + axis * (SHAFT_LENGTH + CONE_LENGTH) * place.unit)?;
            Some(segment_distance(pointer, from, to))
        }
        GizmoHandle::Ball(_) => {
            let centre = place.origin + axis * BALL_REACH * place.unit;
            let on_screen = to_screen(centre)?;
            let edge = to_screen(centre + at.right() * BALL_RADIUS * place.unit)?;
            Some((pointer.distance(on_screen) - on_screen.distance(edge)).max(0.0))
        }
        GizmoHandle::Ring(_) => {
            const SAMPLES: usize = 64;
            let turn = Quat::from_rotation_arc(Vec3::Z, axis);
            let mut nearest = f32::MAX;
            let mut previous = None;
            for i in 0..=SAMPLES {
                let angle = (i % SAMPLES) as f32 * std::f32::consts::TAU / SAMPLES as f32;
                let point = place.origin
                    + turn * Vec3::new(angle.cos(), angle.sin(), 0.0) * RING_RADIUS * place.unit;
                let screen = to_screen(point);
                if let (Some(a), Some(b)) = (previous, screen) {
                    nearest = nearest.min(segment_distance(pointer, a, b));
                }
                previous = screen;
            }
            (nearest < f32::MAX).then_some(nearest)
        }
        // A joint gizmo's handle is measured by its own module.
        GizmoHandle::Joint => None,
        GizmoHandle::Plane(index) => {
            if ray.direction.dot(axis).abs() < EDGE_ON {
                return None;
            }
            let hit = meet_plane(ray, axis, place.origin)?;
            let u = place.reach(GizmoHandle::Arrow((index + 1) % 3));
            let v = place.reach(GizmoHandle::Arrow((index + 2) % 3));
            let offset = (hit - place.origin) / place.unit;
            let half = PLANE_SIDE / 2.0;
            let inside = |along: f32| (along - PLANE_OFFSET).abs() <= half;
            (inside(offset.dot(u)) && inside(offset.dot(v))).then_some(0.0)
        }
    }
}

fn axis_vector(axis: usize) -> Vec3 {
    [Vec3::X, Vec3::Y, Vec3::Z][axis]
}

/// A drag in hand, of either kind of gizmo.
#[derive(Clone, Copy)]
enum Held {
    Transform(Drag),
    Joint(JointHeld),
}

/// A handle under the pointer, of either kind of gizmo, with what taking it
/// needs.
enum Candidate {
    Transform(Entity, GizmoHandle, Placement, Ray3d),
    Joint(Entity, JointReach, Ray3d),
}

/// Finds the handle under the pointer, and takes, drags and lets go of
/// handles with the left button — for transform and joint gizmos alike, so
/// whichever handle of either is nearest the pointer is the one taken.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn drive(
    mut commands: Commands,
    gizmos: Query<(
        Entity,
        &TransformGizmo,
        &InheritedVisibility,
        Option<&Facing>,
    )>,
    mut joints: Query<(Entity, &mut JointGizmo, Option<&Rest>, &InheritedVisibility)>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    globals: Query<&GlobalTransform>,
    mut locals: Query<&mut Transform>,
    parents: Query<&ChildOf>,
    windows: Query<&Window>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut pointer: ResMut<GizmoPointer>,
    mut pressed_scene: ResMut<PressedScene>,
    mut held: Local<Option<Held>>,
) {
    // Taken every frame, so a press noted and not acted on cannot linger.
    let on_scene = std::mem::take(&mut pressed_scene.0);
    let cursor = windows.iter().find_map(Window::cursor_position);
    let ray_through = |camera: Entity| {
        cursor
            .zip(cameras.get(camera).ok())
            .and_then(|(cursor, (camera, at))| camera.viewport_to_world(at, cursor).ok())
    };
    let parent_of = |target: Entity| {
        parents
            .get(target)
            .ok()
            .and_then(|child_of| globals.get(child_of.parent()).ok())
            .copied()
            .unwrap_or(GlobalTransform::IDENTITY)
    };
    let released = !buttons.pressed(MouseButton::Left);

    match *held {
        Some(Held::Transform(drag)) => {
            let Ok((_, gizmo, ..)) = gizmos.get(drag.gizmo) else {
                *held = None;
                pointer.dragging = None;
                return;
            };
            let moved = ray_through(gizmo.camera).and_then(|ray| dragged_to(&drag, ray));
            if let Some(moved) = moved {
                let local = match moved {
                    Moved::World(global) => {
                        // Moving and turning leave the scale as it was; taking
                        // it back out of the world would only add rounding.
                        Transform {
                            scale: drag.start_local.scale,
                            ..global.reparented_to(&parent_of(gizmo.target))
                        }
                    }
                    Moved::Stretch(stretch) => Transform {
                        scale: drag.start_local.scale * stretch,
                        ..drag.start_local
                    },
                };
                if gizmo.apply {
                    if let Ok(mut transform) = locals.get_mut(gizmo.target) {
                        *transform = local;
                    }
                }
                commands.trigger(GizmoDrag {
                    entity: drag.gizmo,
                    target: gizmo.target,
                    handle: drag.handle,
                    phase: if released {
                        DragPhase::End
                    } else {
                        DragPhase::Move
                    },
                    transform: local,
                });
                *held = Some(Held::Transform(Drag {
                    last: local,
                    ..drag
                }));
            } else if released {
                commands.trigger(GizmoDrag {
                    entity: drag.gizmo,
                    target: gizmo.target,
                    handle: drag.handle,
                    phase: DragPhase::End,
                    transform: drag.last,
                });
            }
            if released {
                *held = None;
                pointer.dragging = None;
            }
            return;
        }
        Some(Held::Joint(grip)) => {
            let Ok((_, mut gizmo, ..)) = joints.get_mut(grip.gizmo) else {
                *held = None;
                pointer.dragging = None;
                return;
            };
            let value = ray_through(gizmo.camera).and_then(|ray| dragged(&gizmo, &grip, ray));
            if let Some(value) = value {
                if gizmo.value != value {
                    gizmo.value = value;
                }
            }
            commands.trigger(JointDrag {
                entity: grip.gizmo,
                target: gizmo.target,
                phase: if released {
                    DragPhase::End
                } else {
                    DragPhase::Move
                },
                value: gizmo.value,
            });
            if released {
                *held = None;
                pointer.dragging = None;
            }
            return;
        }
        None => {}
    }

    // Nothing held: find what is under the pointer — unless a button is
    // already down for something else, such as a turn of the view, which a
    // handle passing under the pointer must not interrupt.
    pointer.hovered = None;
    if buttons.get_pressed().next().is_some() && !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let Some(cursor) = cursor else {
        return;
    };
    let mut nearest: Option<(f32, Candidate)> = None;
    let mut offer = |distance: f32, candidate: Candidate| {
        if distance <= REACH_PX && nearest.as_ref().is_none_or(|(best, _)| distance < *best) {
            nearest = Some((distance, candidate));
        }
    };
    for (entity, gizmo, visible, facing) in &gizmos {
        if !visible.get() {
            continue;
        }
        // Facing comes with the parts; until they are built, nothing shows.
        let Some(facing) = facing else {
            continue;
        };
        let (Ok(camera), Ok(target)) = (cameras.get(gizmo.camera), globals.get(gizmo.target))
        else {
            continue;
        };
        let Ok(ray) = camera.0.viewport_to_world(camera.1, cursor) else {
            continue;
        };
        let Some(place) = placement(gizmo, facing.0, target, camera) else {
            continue;
        };
        for handle in GizmoHandle::ALL {
            if !handle.mode().is_some_and(|mode| gizmo.shows(mode)) {
                continue;
            }
            if let Some(distance) = distance_to(handle, &place, cursor, ray, camera) {
                offer(distance, Candidate::Transform(entity, handle, place, ray));
            }
        }
    }
    for (entity, gizmo, rest, visible) in &joints {
        if !visible.get() {
            continue;
        }
        let Ok(camera) = cameras.get(gizmo.camera) else {
            continue;
        };
        let Ok(ray) = camera.0.viewport_to_world(camera.1, cursor) else {
            continue;
        };
        let Some(reach) = joint_reach(gizmo, rest, &parent_of(gizmo.target), camera) else {
            continue;
        };
        if let Some(distance) = joint_distance(gizmo, &reach, cursor, camera) {
            offer(distance, Candidate::Joint(entity, reach, ray));
        }
    }
    let Some((_, candidate)) = nearest else {
        return;
    };
    let hovered = match &candidate {
        Candidate::Transform(entity, handle, ..) => (*entity, *handle),
        Candidate::Joint(entity, ..) => (*entity, GizmoHandle::Joint),
    };
    pointer.hovered = Some(hovered);

    if !buttons.just_pressed(MouseButton::Left) || !on_scene {
        return;
    }
    match candidate {
        Candidate::Transform(entity, handle, place, ray) => {
            let Ok((_, gizmo, ..)) = gizmos.get(entity) else {
                return;
            };
            let (Ok(start_global), Ok(start_local)) =
                (globals.get(gizmo.target), locals.get(gizmo.target))
            else {
                return;
            };
            let axis = place.reach(handle);
            let plane = match handle {
                GizmoHandle::Arrow(_) | GizmoHandle::Ball(_) => plane_along(ray, axis),
                GizmoHandle::Plane(_) | GizmoHandle::Ring(_) | GizmoHandle::Joint => axis,
            };
            let Some(grabbed) = meet_plane(ray, plane, place.origin) else {
                return;
            };
            *held = Some(Held::Transform(Drag {
                gizmo: entity,
                handle,
                origin: place.origin,
                axes: place.axes,
                stretch_along: nearest_component(start_global.rotation(), axis),
                grabbed,
                plane,
                start_global: *start_global,
                start_local: *start_local,
                last: *start_local,
            }));
            commands.trigger(GizmoDrag {
                entity,
                target: gizmo.target,
                handle,
                phase: DragPhase::Start,
                transform: *start_local,
            });
        }
        Candidate::Joint(entity, reach, ray) => {
            let Ok((_, gizmo, ..)) = joints.get(entity) else {
                return;
            };
            let Some(grip) = take(entity, gizmo, &reach, ray) else {
                return;
            };
            *held = Some(Held::Joint(grip));
            commands.trigger(JointDrag {
                entity,
                target: gizmo.target,
                phase: DragPhase::Start,
                value: gizmo.value,
            });
        }
    }
    pointer.dragging = Some(hovered);
}

/// What a drag does to its target.
#[derive(Debug)]
enum Moved {
    /// Moves or turns it to here, in the world.
    World(GlobalTransform),
    /// Stretches it along its own axes by this much. Scale is the target's
    /// own, so it is changed there rather than taken through the world, where
    /// a parent's scale would be folded into it.
    Stretch(Vec3),
}

/// Where a drag has taken the target, for the pointer's ray now.
fn dragged_to(drag: &Drag, ray: Ray3d) -> Option<Moved> {
    let point = meet_plane(ray, drag.plane, drag.origin)?;
    let (scale, rotation, translation) = drag.start_global.to_scale_rotation_translation();
    let axis = |axes: Quat, index: usize| axes * axis_vector(index);
    let moved = match drag.handle {
        GizmoHandle::Arrow(index) => {
            let axis = axis(drag.axes, index);
            Transform {
                translation: translation + axis * (point - drag.grabbed).dot(axis),
                rotation,
                scale,
            }
        }
        GizmoHandle::Plane(_) => Transform {
            translation: translation + (point - drag.grabbed),
            rotation,
            scale,
        },
        GizmoHandle::Ring(index) => {
            let axis = axis(drag.axes, index);
            let from = drag.grabbed - drag.origin;
            let to = point - drag.origin;
            let angle = from.cross(to).dot(axis).atan2(from.dot(to));
            let turn = Quat::from_axis_angle(axis, angle);
            Transform {
                translation: drag.origin + turn * (translation - drag.origin),
                rotation: turn * rotation,
                scale,
            }
        }
        // A joint drag is carried by its own module, never as a `Drag`.
        GizmoHandle::Joint => return None,
        GizmoHandle::Ball(index) => {
            let axis = axis(drag.axes, index);
            let from = (drag.grabbed - drag.origin).dot(axis);
            let to = (point - drag.origin).dot(axis);
            if from.abs() < f32::EPSILON {
                return None;
            }
            let stretch = Vec3::ONE + drag.stretch_along * ((to / from).max(0.01) - 1.0);
            return Some(Moved::Stretch(stretch));
        }
    };
    Some(Moved::World(GlobalTransform::from(moved)))
}

/// A guide a gizmo shows along one of its axes while a handle is held.
#[derive(Component)]
struct Guide(usize);

/// How far a guide reaches each way, in gizmo units: well past the handles,
/// so the line reads as the axis rather than as another handle.
const GUIDE_REACH: f32 = 8.0;

/// The axes a held handle constrains: its own, or a plane's two.
fn guided(handle: GizmoHandle) -> Vec<usize> {
    match handle {
        GizmoHandle::Arrow(axis) | GizmoHandle::Ring(axis) | GizmoHandle::Ball(axis) => {
            vec![axis]
        }
        GizmoHandle::Plane(axis) => (0..3).filter(|other| *other != axis).collect(),
        GizmoHandle::Joint => Vec::new(),
    }
}

/// The guides under gizmos, placed by hand with them.
type Guides<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static Guide,
        &'static ChildOf,
        &'static mut AxisLine,
        &'static mut GlobalTransform,
    ),
    (Without<Part>, Without<TransformGizmo>),
>;

/// Stands each gizmo at its target, sizes it, turns each part onto its axis,
/// shows only the handles it should, and gives the hot handle its in-hand
/// look.
///
/// It runs after transforms propagate, since where the gizmo stands depends
/// on where its target and camera ended up this frame; so it writes the world
/// transforms of the gizmo and its parts itself, or they would be drawn where
/// they stood a frame ago and trail the view as it turns. That is also why a
/// gizmo has to stand at the top of the hierarchy.
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn place_parts(
    mut commands: Commands,
    mut guides: Guides,
    pointer: Res<GizmoPointer>,
    cameras: Query<
        (&Camera, &GlobalTransform),
        (Without<Part>, Without<TransformGizmo>, Without<Guide>),
    >,
    targets: Query<&GlobalTransform, (Without<Part>, Without<TransformGizmo>, Without<Guide>)>,
    mut gizmos: Query<
        (
            Entity,
            &TransformGizmo,
            &mut Transform,
            &mut GlobalTransform,
            &Children,
            Option<&mut Facing>,
        ),
        Without<Part>,
    >,
    mut parts: Query<
        (
            &mut Part,
            &mut Transform,
            &mut GlobalTransform,
            &mut Visibility,
            &mut Mesh3d,
            &mut MeshMaterial3d<HandleMaterial>,
        ),
        Without<TransformGizmo>,
    >,
) {
    let hot = pointer.dragging.or(pointer.hovered);
    for (entity, gizmo, mut transform, mut global, children, facing) in &mut gizmos {
        let Some(mut facing) = facing else {
            continue;
        };
        let (Ok(camera), Ok(target)) = (cameras.get(gizmo.camera), targets.get(gizmo.target))
        else {
            continue;
        };
        let Some(mut place) = placement(gizmo, facing.0, target, camera) else {
            continue;
        };
        // Sides change only between drags, so a handle in hand stays where
        // the pointer took it.
        let held = pointer.dragging.is_some_and(|(gizmo, _)| gizmo == entity);
        if !held {
            let toward = facing_toward(facing.0, place.axes, place.origin, camera.1.translation());
            if toward != facing.0 {
                facing.0 = toward;
            }
            place.facing = toward;
        }
        *transform = Transform::from_translation(place.origin).with_scale(Vec3::splat(place.unit));
        *global = GlobalTransform::from(*transform);
        for child in children.iter() {
            let Ok((mut part, mut at, mut part_global, mut visibility, mut mesh, mut material)) =
                parts.get_mut(child)
            else {
                continue;
            };
            let index = part.handle.axis();
            let turn = place.axes * onto_axis(index);
            // The layout's X is the handle's own axis and its Y and Z the
            // other two, so each flips with the side its axis faces. A ring
            // has no side.
            let flip = match part.handle {
                GizmoHandle::Ring(_) => Vec3::ONE,
                _ => Vec3::new(
                    place.facing[index],
                    place.facing[(index + 1) % 3],
                    place.facing[(index + 2) % 3],
                ),
            };
            *at = Transform {
                translation: turn * (part.offset * flip),
                rotation: turn * Quat::from_rotation_arc(Vec3::Y, Vec3::X * flip.x),
                scale: Vec3::ONE,
            };
            *part_global = global.mul_transform(*at);
            let shown = part.handle.mode().is_some_and(|mode| gizmo.shows(mode))
                && !(part.stem && gizmo.shows(TransformMode::Translate));
            *visibility = if shown {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
            let is_hot = hot == Some((entity, part.handle));
            if part.hot != is_hot {
                part.hot = is_hot;
                let look = usize::from(is_hot);
                mesh.0 = part.meshes[look].clone();
                material.0 = part.materials[look].clone();
            }
        }

        let wanted = match pointer.dragging {
            Some((held, handle)) if held == entity && gizmo.guides => guided(handle),
            _ => Vec::new(),
        };
        let line = |axis: usize| AxisLine {
            direction: Dir3::new(place.axes * axis_vector(axis)).unwrap_or(Dir3::X),
            reach: GUIDE_REACH,
            both_ways: true,
            size: gizmo.size,
            camera: Some(gizmo.camera),
            colour: gizmo.colours[axis],
        };
        let mut present = Vec::new();
        for (guide, axis, parent, mut shown, mut guide_global) in &mut guides {
            if parent.parent() != entity {
                continue;
            }
            if !wanted.contains(&axis.0) {
                commands.entity(guide).despawn();
                continue;
            }
            present.push(axis.0);
            shown.set_if_neq(line(axis.0));
            // A guide stands at the gizmo's origin, which was placed by hand
            // after propagation; so is the guide.
            *guide_global = *global;
        }
        for axis in wanted.into_iter().filter(|axis| !present.contains(axis)) {
            commands.spawn((
                Guide(axis),
                line(axis),
                Transform::default(),
                *global,
                Visibility::default(),
                ChildOf(entity),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A move, turn or stretch shows its own axis; a plane shows its two.
    #[test]
    fn a_held_handle_guides_the_axes_it_constrains() {
        assert_eq!(guided(GizmoHandle::Arrow(1)), vec![1]);
        assert_eq!(guided(GizmoHandle::Ring(2)), vec![2]);
        assert_eq!(guided(GizmoHandle::Ball(0)), vec![0]);
        assert_eq!(guided(GizmoHandle::Plane(1)), vec![0, 2]);
        assert!(guided(GizmoHandle::Joint).is_empty());
    }

    /// Placing parts and guides by hand writes transforms its other queries
    /// read; Bevy refuses overlapping access when the system first runs, not
    /// when it compiles, so this runs it.
    #[test]
    fn placing_parts_runs_without_conflicting_access() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<GizmoPointer>()
            .add_systems(PostUpdate, place_parts);
        app.update();
    }

    /// Carrying X onto each axis carries the plane laid out across Y and Z
    /// onto the plane across the other two, in the same handedness.
    #[test]
    fn onto_axis_keeps_the_axes_right_handed() {
        for axis in 0..3 {
            let turn = onto_axis(axis);
            let x = turn * Vec3::X;
            let y = turn * Vec3::Y;
            let z = turn * Vec3::Z;
            assert!(x.abs_diff_eq(axis_vector(axis), 1e-6));
            assert!(y.abs_diff_eq(axis_vector((axis + 1) % 3), 1e-6));
            assert!(z.abs_diff_eq(axis_vector((axis + 2) % 3), 1e-6));
        }
    }

    fn drag(handle: GizmoHandle, grabbed: Vec3, plane: Vec3) -> Drag {
        Drag {
            gizmo: Entity::PLACEHOLDER,
            handle,
            origin: Vec3::ZERO,
            axes: Quat::IDENTITY,
            stretch_along: axis_vector(handle.axis()),
            grabbed,
            plane,
            start_global: GlobalTransform::IDENTITY,
            start_local: Transform::IDENTITY,
            last: Transform::IDENTITY,
        }
    }

    fn world(moved: Option<Moved>) -> GlobalTransform {
        match moved {
            Some(Moved::World(global)) => global,
            other => panic!("expected a move in the world, got {other:?}"),
        }
    }

    /// Under Z-up labels the Z arrow points along Bevy's up and the Y arrow
    /// along Bevy's −Z.
    #[test]
    fn z_up_labels_point_the_handles_their_way() {
        let gizmo = TransformGizmo {
            axes: TransformGizmo::Z_UP,
            ..TransformGizmo::new(Entity::PLACEHOLDER, Entity::PLACEHOLDER)
        };
        let labels = gizmo.labels();
        assert!((labels * axis_vector(2)).abs_diff_eq(Vec3::Y, 1e-6));
        assert!((labels * axis_vector(1)).abs_diff_eq(Vec3::NEG_Z, 1e-6));
    }

    /// Each axis faces the side the eye is on, and one nearly side-on keeps
    /// the side it had rather than flickering.
    #[test]
    fn handles_face_the_viewer_and_hold_when_side_on() {
        let eye = Vec3::new(-5.0, 5.0, 0.2);
        let toward = facing_toward(Vec3::ONE, Quat::IDENTITY, Vec3::ZERO, eye);
        // X is behind, Y in front, Z all but side-on so it keeps its side.
        assert_eq!(toward, Vec3::new(-1.0, 1.0, 1.0));
        let kept = facing_toward(Vec3::NEG_ONE, Quat::IDENTITY, Vec3::ZERO, eye);
        assert_eq!(kept, Vec3::new(-1.0, 1.0, -1.0));
    }

    /// A facing handle reaches the way it faces; a ring does not face.
    #[test]
    fn a_facing_handle_reaches_toward_its_side() {
        let place = Placement {
            origin: Vec3::ZERO,
            axes: Quat::IDENTITY,
            facing: Vec3::new(-1.0, 1.0, 1.0),
            unit: 1.0,
        };
        assert_eq!(place.reach(GizmoHandle::Arrow(0)), Vec3::NEG_X);
        assert_eq!(place.reach(GizmoHandle::Ball(0)), Vec3::NEG_X);
        assert_eq!(place.reach(GizmoHandle::Ring(0)), Vec3::X);
    }

    /// In world space a ball on a turned target stretches the target's own
    /// axis that lies along the ball's: a quarter turn about Y lays the
    /// target's Z along the world's X.
    #[test]
    fn a_ball_stretches_the_nearest_own_axis() {
        let turned = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        assert_eq!(nearest_component(turned, Vec3::X), Vec3::Z);
        assert_eq!(nearest_component(turned, Vec3::NEG_X), Vec3::Z);
        assert_eq!(nearest_component(Quat::IDENTITY, Vec3::Y), Vec3::Y);
    }

    fn ray_down_at(x: f32, z: f32) -> Ray3d {
        Ray3d::new(Vec3::new(x, 10.0, z), Dir3::NEG_Y)
    }

    /// An arrow moves only along its axis, however far off it the pointer
    /// strays.
    #[test]
    fn an_arrow_moves_only_along_its_axis() {
        let d = drag(GizmoHandle::Arrow(0), Vec3::new(0.5, 0.0, 0.0), Vec3::Y);
        let moved = world(dragged_to(&d, ray_down_at(2.0, 3.0)));
        assert!(moved
            .translation()
            .abs_diff_eq(Vec3::new(1.5, 0.0, 0.0), 1e-5));
    }

    /// A plane moves freely within itself.
    #[test]
    fn a_plane_moves_within_itself() {
        let d = drag(GizmoHandle::Plane(1), Vec3::new(0.3, 0.0, 0.3), Vec3::Y);
        let moved = world(dragged_to(&d, ray_down_at(1.3, -0.7)));
        assert!(moved
            .translation()
            .abs_diff_eq(Vec3::new(1.0, 0.0, -1.0), 1e-5));
    }

    /// A ring turns by the angle the pointer sweeps about the gizmo's centre,
    /// in the sense of the axis it is about.
    #[test]
    fn a_ring_turns_by_the_angle_swept() {
        let d = drag(GizmoHandle::Ring(1), Vec3::new(1.0, 0.0, 0.0), Vec3::Y);
        // A quarter turn about +Y carries +X to -Z.
        let moved = world(dragged_to(&d, ray_down_at(0.0, -1.0)));
        assert!((moved.rotation() * Vec3::X).abs_diff_eq(Vec3::NEG_Z, 1e-5));
    }

    /// A ball scales its own axis by how much farther out the pointer is than
    /// where it took hold, and leaves the others.
    #[test]
    fn a_ball_scales_only_its_axis() {
        let d = drag(GizmoHandle::Ball(2), Vec3::new(0.0, 0.0, 1.0), Vec3::Y);
        let Some(Moved::Stretch(stretch)) = dragged_to(&d, ray_down_at(0.4, 2.0)) else {
            panic!("a ball stretches");
        };
        assert!(stretch.abs_diff_eq(Vec3::new(1.0, 1.0, 2.0), 1e-5));
    }
}
