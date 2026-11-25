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
//! **Each arrow points the positive way along its axis, unless the gizmo is
//! told to face the viewer.** An arrow that always points to +X reads as the
//! axis's direction, which is what a driver writing poses back is reasoning
//! in. With [`TransformGizmo::face_viewer`] set, each arrow instead points
//! along its axis or against it, whichever way is toward the camera, and the
//! squares and balls go with the arrows, so that as the view comes round no
//! arrow ends up behind the rings and the centre, where taking it means
//! reaching through them. An arrow seen nearly end-on keeps the side it had
//! until the view is clearly past it, so it does not flicker between the two,
//! and nothing changes side while a drag is under way. A host that wants one mode
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
    distance as joint_distance, dragged, reach as joint_reach, take, value_of, JointDrag,
    JointGizmo, JointHeld, JointReach, Rest,
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
///
/// A driver writes one as `{ "pixels": 80 }` or `{ "metres": 0.25 }`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GizmoSize {
    /// The same on screen however near or far, in logical pixels.
    Pixels(f32),
    /// Fixed in the world, in metres, so it shrinks with distance as what it
    /// moves does.
    #[serde(rename = "metres")]
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
    /// Handles drawn as the one under the pointer is, whatever the pointer
    /// does: a host pointing at this gizmo from elsewhere, such as the row
    /// for it in a list beside the view. It takes nothing from the pointer,
    /// so hovering and dragging work as ever.
    pub highlight: Vec<GizmoHandle>,
    /// How opaque the whole gizmo draws, from 0 to 1, over each handle's own
    /// look and its colour's alpha, its drag guides included. A host fades a
    /// control with it. It is how the gizmo looks only: a faded gizmo, even
    /// an invisible one, still takes the pointer until it is not
    /// [`interactive`](Self::interactive).
    pub opacity: f32,
    /// Whether the pointer can hover, take or press its handles. Without it
    /// the pointer finds nothing here and reaches what lies beyond, the view
    /// and the scene; a drag in hand when it goes off ends. A host's
    /// [`highlight`](Self::highlight) still shows.
    pub interactive: bool,
    /// Turn each axis's arrow, square and ball to whichever side of the
    /// origin faces the camera, rather than keeping them on the positive
    /// side ([module docs](self)).
    pub face_viewer: bool,
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
            highlight: Vec::new(),
            opacity: 1.0,
            interactive: true,
            face_viewer: false,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

    /// Every handle a transform gizmo has, which a host highlights to
    /// highlight the whole gizmo.
    pub const ALL: [GizmoHandle; 12] = [
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

/// Where a drag is. Every drag has one `Start`, a `Move` for each frame its
/// value changes, and then exactly one of `End` or `Cancel`, and nothing after
/// it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DragPhase {
    Start,
    Move,
    /// The pointer let go: the drag is finished, at its last value.
    End,
    /// The drag was abandoned rather than finished: a host cancelled it
    /// ([`GizmoDrags::cancel_drag`]), or its gizmo went inert or was hidden
    /// — by its control, or by anything above it, such as an ancestor's
    /// `Visibility` — with the button still down. A gizmo that moves its
    /// target itself puts the target back where the drag found it.
    Cancel,
}

/// A drag on a gizmo's handle, triggered on the gizmo.
#[derive(EntityEvent, Clone, Copy, Debug)]
pub struct GizmoDrag {
    /// The gizmo.
    pub entity: Entity,
    pub target: Entity,
    pub handle: GizmoHandle,
    pub phase: DragPhase,
    /// The drag this belongs to, the same on all of its events
    /// ([`GizmoDrags`] numbers them).
    pub gesture: u32,
    /// The target's local transform as the drag has it now; on a `Cancel`,
    /// as the drag found it.
    pub transform: Transform,
}

/// A press with a button other than the primary on a gizmo's handle, triggered
/// on the gizmo. The primary button takes a handle; any other is the host's,
/// to open a menu for the control, say, and starts no turn or pan of the
/// view.
#[derive(EntityEvent, Clone, Copy, Debug)]
pub struct GizmoPress {
    /// The gizmo.
    pub entity: Entity,
    pub handle: GizmoHandle,
    pub button: PointerButton,
    /// Where the press was, in the window's logical pixels.
    pub position: Vec2,
}

/// What the pointer is doing with the gizmos.
#[derive(Resource, Default, Debug)]
pub struct GizmoPointer {
    /// The handle under the pointer, and its gizmo.
    pub hovered: Option<(Entity, GizmoHandle)>,
    /// The handle being dragged, and its gizmo.
    pub dragging: Option<(Entity, GizmoHandle)>,
    /// A drag ended by its gizmo going inert or being taken away, while the
    /// button that took it is still down. Nothing is held, but the press is
    /// still the gizmo's: its release is no click, and the view does not take
    /// it up. See [`GizmoPointer::owns_button`].
    pub ended: Option<(Entity, GizmoHandle)>,
    /// How far the transform gizmo's drag in hand has gone, in its handle's
    /// own measure ([`Reading`]); empty when no transform handle is held.
    pub readings: Vec<Reading>,
}

/// The numbering of drags, and a host's way to abandon the one in hand.
#[derive(Resource, Default, Debug)]
pub struct GizmoDrags {
    /// The number the next drag takes: each drag, of either kind of gizmo,
    /// takes the next, so a host can tell one drag's events from another's.
    next_gesture: u32,
    /// Whether a host has asked for the drag in hand to be cancelled.
    cancel: bool,
}

/// One amount a held handle has moved its target by, for a host to show
/// beside the gizmo: along one axis, or about it.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Reading {
    /// The axis it is along or about — 0, 1 or 2 for X, Y or Z, as the gizmo
    /// names them in [`TransformGizmo::axes`]. `None` for a joint, whose one
    /// axis is its own.
    pub axis: Option<usize>,
    pub amount: f32,
    pub unit: ReadingUnit,
}

/// What a [`Reading`] counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ReadingUnit {
    /// A distance moved: along an arrow, across a plane, or a sliding
    /// joint's travel.
    Metres,
    /// An angle turned: about a ring, or a turning joint's.
    Degrees,
    /// A stretch: the target's scale along the axis, over its scale when the
    /// drag began.
    Factor,
}

/// How far a drag has carried its target from where it began, in the
/// held handle's own measure: the move along an arrow, a plane's move along
/// each of its two axes, the turn about a ring, or a ball's stretch. Axes
/// are the gizmo's as the drag began, so a reading does not drift as a
/// gizmo in local space turns with what it turns.
fn readings(drag: &Drag, moved: &Moved) -> Vec<Reading> {
    let axis = |index: usize| drag.axes * axis_vector(index);
    let along = |index: usize, offset: Vec3| Reading {
        axis: Some(index),
        amount: offset.dot(axis(index)),
        unit: ReadingUnit::Metres,
    };
    match (drag.handle, moved) {
        (GizmoHandle::Arrow(index), Moved::World(global)) => {
            vec![along(
                index,
                global.translation() - drag.start_global.translation(),
            )]
        }
        (GizmoHandle::Plane(index), Moved::World(global)) => {
            let offset = global.translation() - drag.start_global.translation();
            // In axis order, so a host stacking them shows X above Z.
            (0..3)
                .filter(|other| *other != index)
                .map(|other| along(other, offset))
                .collect()
        }
        (GizmoHandle::Ring(index), Moved::World(global)) => {
            // The twist about the ring's axis: the part of the turn a ring
            // can make, measured either way round, up to half a turn.
            let turn = global.rotation() * drag.start_global.rotation().inverse();
            let twist = 2.0 * turn.xyz().dot(axis(index)).atan2(turn.w);
            let twist = (twist + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
                - std::f32::consts::PI;
            vec![Reading {
                axis: Some(index),
                amount: twist.to_degrees(),
                unit: ReadingUnit::Degrees,
            }]
        }
        (GizmoHandle::Ball(index), Moved::Stretch(stretch)) => vec![Reading {
            axis: Some(index),
            amount: stretch.dot(drag.stretch_along),
            unit: ReadingUnit::Factor,
        }],
        _ => Vec::new(),
    }
}

impl GizmoPointer {
    /// Whether the pointer belongs to a gizmo: a press now would grab a
    /// handle, or the button down is a gizmo's ([`Self::owns_button`]).
    pub fn engaged(&self) -> bool {
        self.hovered.is_some() || self.owns_button()
    }

    /// Whether the left button now down is a gizmo's: it took a handle, and
    /// the drag is in hand or was ended by its gizmo going inert, being
    /// hidden or being taken away. Until it comes up, its release is not a
    /// click and no gesture of the view's starts from it.
    pub fn owns_button(&self) -> bool {
        self.dragging.is_some() || self.ended.is_some()
    }
}

impl GizmoDrags {
    /// Abandons the drag in hand, if there is one, at the start of the next
    /// frame's drag handling: it closes with [`DragPhase::Cancel`], and
    /// nothing more is sent for it — the button later coming up included,
    /// which is not taken for a click either. A host calls this for what the
    /// scene cannot see for itself: Escape, the window losing focus, the
    /// pointer being cancelled, the page being hidden. Asked between frames,
    /// it takes effect before anything else happens to the drag.
    pub fn cancel_drag(&mut self) {
        self.cancel = true;
    }

    fn take_gesture(&mut self) -> u32 {
        let gesture = self.next_gesture;
        self.next_gesture = self.next_gesture.wrapping_add(1);
        gesture
    }
}

/// Marks every mesh this crate draws as an aid to the view rather than as
/// part of what it shows — a gizmo's handles, a frame's axes, a guide line —
/// so a host can leave them out of anything that measures the scene.
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

#[cfg(test)]
impl Part {
    /// A part with nothing to draw, for a test that only needs its mark.
    pub(crate) fn stub() -> Self {
        Self {
            handle: GizmoHandle::Arrow(0),
            offset: Vec3::ZERO,
            stem: false,
            meshes: Default::default(),
            materials: Default::default(),
            alphas: [1.0; 2],
            faded: 1.0,
            hot: false,
        }
    }
}

/// A drawn piece of a handle, with its rest and in-hand looks.
#[derive(Component)]
pub(crate) struct Part {
    handle: GizmoHandle,
    /// Where the piece sits, in gizmo units, laid out about +X before its
    /// handle's axis and facing are applied. Every piece's mesh is built
    /// along its own Y and is laid along that axis.
    offset: Vec3,
    /// A scale ball's stem, shown only while the arrows are not.
    stem: bool,
    meshes: [Handle<Mesh>; 2],
    materials: [Handle<HandleMaterial>; 2],
    /// Each look's alpha before the gizmo's opacity: its handle's own, times
    /// its colour's.
    alphas: [f32; 2],
    /// The gizmo opacity its materials were last given.
    faded: f32,
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
    gesture: u32,
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
    /// The target's position and turn in the world when the drag last
    /// reported: the pose a drag reports, which its parent moving changes
    /// though the drag's local transform does not.
    reported: (Vec3, Quat),
}

/// The global transform of `target`'s parent as it stands now, not as last
/// propagated; the world's frame for a target with no parent.
fn parent_now(
    target: Entity,
    parents: &Query<&ChildOf>,
    locals: &Query<&mut Transform>,
    globals: &Query<&GlobalTransform>,
) -> GlobalTransform {
    parents
        .get(target)
        .ok()
        .and_then(|child_of| {
            current_global(
                child_of.parent(),
                |entity| locals.get(entity).ok().copied(),
                parents,
                globals,
            )
        })
        .map_or(GlobalTransform::IDENTITY, GlobalTransform::from)
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
        if !crate::installed::first_install::<Self>(app) {
            return;
        }
        add_handle_material(app);
        app.add_plugins(AxisLinePlugin);
        app.init_resource::<GizmoPointer>()
            .init_resource::<GizmoDrags>()
            .init_resource::<PressedScene>()
            .add_observer(note_scene_press)
            .add_observer(report_other_press)
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

/// The handle a press with `button` is the host's on, if it is: any button
/// but the primary, on the handle `under` the press, while the left button is
/// not a gizmo's ([`GizmoPointer::owns_button`]).
///
/// `under` is asked at the press's own position rather than read from
/// [`GizmoPointer::hovered`], which the last frame left: a pointer that
/// moved onto or off a handle and pressed between two frames would be taken
/// for where it was. During a drag the held handle stays hovered wherever
/// the pointer goes, so a press then is on nothing in particular.
pub(crate) fn hosts_press(
    button: PointerButton,
    pointer: &GizmoPointer,
    under: impl FnOnce() -> Option<(Entity, GizmoHandle)>,
) -> Option<(Entity, GizmoHandle)> {
    if button == PointerButton::Primary || pointer.owns_button() {
        return None;
    }
    under()
}

/// Whether a press is the gizmos' rather than the view's: the button down
/// is a gizmo's, or the press is a host's on a handle ([`hosts_press`]).
pub(crate) fn keeps_press(
    button: PointerButton,
    pointer: &GizmoPointer,
    under: impl FnOnce() -> Option<(Entity, GizmoHandle)>,
) -> bool {
    pointer.owns_button() || hosts_press(button, pointer, under).is_some()
}

/// Whether a part looks as the one in hand does: its handle is the one the
/// pointer has, or one the host highlights.
fn looks_hot(
    hot: Option<(Entity, GizmoHandle)>,
    entity: Entity,
    gizmo: &TransformGizmo,
    handle: GizmoHandle,
) -> bool {
    hot == Some((entity, handle)) || gizmo.highlight.contains(&handle)
}

/// Reports a press that is the host's ([`hosts_press`]) as a [`GizmoPress`].
fn report_other_press(
    press: On<PointerPress>,
    windows: Query<(), With<Window>>,
    pointer: Res<GizmoPointer>,
    finder: HandleFinder,
    mut commands: Commands,
) {
    if !on_the_scene(press.entity, press.original_event_target(), &windows) {
        return;
    }
    let pointer_at = &press.pointer;
    if let Some((entity, handle)) = hosts_press(press.button, &pointer, || {
        finder.at(pointer_at.position, &pointer_at.target)
    }) {
        commands.trigger(GizmoPress {
            entity,
            handle,
            button: press.button,
            position: press.pointer.position,
        });
    }
}

/// A part to spawn: its handle, where it sits, its rest and in-hand meshes
/// and materials, and whether it is a scale ball's stem.
type Piece = (GizmoHandle, Vec3, [Handle<Mesh>; 2], Look, bool);

/// A piece's two looks: its materials, and the alpha each has before the
/// gizmo's opacity.
#[derive(Clone)]
struct Look {
    materials: [Handle<HandleMaterial>; 2],
    alphas: [f32; 2],
}

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
            let alphas = opacity.map(|alpha| alpha * colour.alpha());
            Look {
                materials: [0, 1].map(|hot| {
                    let base = if hot == 1 {
                        colour.mix(&Color::WHITE, LIGHTEN)
                    } else {
                        colour
                    };
                    materials.add(HandleMaterial {
                        handle: HandleUniform {
                            colour: LinearRgba::from(base.with_alpha(alphas[hot] * gizmo.opacity))
                                .to_vec4(),
                        },
                    })
                }),
                alphas,
            }
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
        let gizmo_opacity = gizmo.opacity;
        commands.entity(entity).with_children(|gizmo| {
            for (handle, offset, meshes, look, stem) in pieces {
                let materials = look.materials;
                gizmo.spawn((
                    Part {
                        handle,
                        offset,
                        stem,
                        meshes: meshes.clone(),
                        materials: materials.clone(),
                        alphas: look.alphas,
                        faded: gizmo_opacity,
                        hot: false,
                    },
                    GizmoPart,
                    // A handle is taken by this crate's own pointer handling;
                    // a mesh picking backend must not take the press first.
                    Pickable::IGNORE,
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

/// Points around each ring, for a gizmo's box: the transform gizmo's rings
/// and a joint's alike.
pub(crate) const RING_SAMPLES: usize = 24;

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

/// The side of each axis a gizmo's handles go on: the positive side, or with
/// `face_viewer` the side toward the eye.
fn sides(face_viewer: bool, previous: Vec3, axes: Quat, origin: Vec3, eye: Vec3) -> Vec3 {
    if face_viewer {
        facing_toward(previous, axes, origin, eye)
    } else {
        Vec3::ONE
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

/// A drag in hand, of either kind of gizmo. Not boxed though its variants
/// differ in size: there is one, in a `Local`, and copying it out each frame
/// is what lets the drag be read while the pointer state is changed.
#[derive(Clone, Copy)]
#[allow(clippy::large_enum_variant)]
enum Held {
    Transform(Drag),
    Joint {
        grip: JointHeld,
        gesture: u32,
        /// The joint's value when the drag began.
        start: f32,
    },
}

/// A handle under the pointer, of either kind of gizmo, with what taking it
/// needs.
enum Candidate {
    Transform(Entity, GizmoHandle, Placement, Ray3d),
    Joint(Entity, JointReach, Ray3d),
}

impl Candidate {
    /// The gizmo and the handle this is.
    fn grip(&self) -> (Entity, GizmoHandle) {
        match self {
            Candidate::Transform(entity, handle, ..) => (*entity, *handle),
            Candidate::Joint(entity, ..) => (*entity, GizmoHandle::Joint),
        }
    }
}

/// The handle nearest `cursor` within reach, of any visible gizmo that takes
/// the pointer, through
/// each gizmo's own camera: what the pointer is over. Hover, a take and a
/// press with another button all ask this one search, so they cannot
/// disagree about what a pointer is on.
fn nearest_handle<'a>(
    cursor: Vec2,
    gizmos: impl Iterator<
        Item = (
            Entity,
            &'a TransformGizmo,
            &'a InheritedVisibility,
            Option<&'a Facing>,
        ),
    >,
    joints: impl Iterator<
        Item = (
            Entity,
            &'a JointGizmo,
            Option<&'a Rest>,
            &'a InheritedVisibility,
        ),
    >,
    cameras: &Query<(&Camera, &GlobalTransform)>,
    globals: &Query<&GlobalTransform>,
    parent_of: impl Fn(Entity) -> GlobalTransform,
) -> Option<Candidate> {
    let mut nearest: Option<(f32, Candidate)> = None;
    let mut offer = |distance: f32, candidate: Candidate| {
        if distance <= REACH_PX && nearest.as_ref().is_none_or(|(best, _)| distance < *best) {
            nearest = Some((distance, candidate));
        }
    };
    for (entity, gizmo, visible, facing) in gizmos {
        if !visible.get() || !gizmo.interactive {
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
    for (entity, gizmo, rest, visible) in joints {
        if !visible.get() || !gizmo.interactive {
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
    nearest.map(|(_, candidate)| candidate)
}

/// What [`nearest_handle`] reads, for an observer that asks it of a press.
#[derive(bevy::ecs::system::SystemParam)]
pub(crate) struct HandleFinder<'w, 's> {
    gizmos: Query<
        'w,
        's,
        (
            Entity,
            &'static TransformGizmo,
            &'static InheritedVisibility,
            Option<&'static Facing>,
        ),
    >,
    joints: Query<
        'w,
        's,
        (
            Entity,
            &'static JointGizmo,
            Option<&'static Rest>,
            &'static InheritedVisibility,
        ),
    >,
    cameras: Query<'w, 's, (&'static Camera, &'static GlobalTransform)>,
    globals: Query<'w, 's, &'static GlobalTransform>,
    parents: Query<'w, 's, &'static ChildOf>,
    targets: Query<'w, 's, &'static bevy::camera::RenderTarget>,
    primary: Query<'w, 's, Entity, With<bevy::window::PrimaryWindow>>,
}

/// Whether a camera drawing to `drawn` draws to `wanted`, where "the primary
/// window" means `primary`. A pointer's position means something only in the
/// window it is in, so only the gizmos seen there can be under it.
pub(crate) fn draws_to(
    drawn: Option<&bevy::camera::RenderTarget>,
    primary: Option<Entity>,
    wanted: &bevy::camera::NormalizedRenderTarget,
) -> bool {
    drawn.and_then(|drawn| drawn.normalize(primary)).as_ref() == Some(wanted)
}

impl HandleFinder<'_, '_> {
    /// The gizmo and handle under `position`, in the logical pixels of the
    /// window `target` names, among the gizmos seen in that window.
    pub(crate) fn at(
        &self,
        position: Vec2,
        target: &bevy::camera::NormalizedRenderTarget,
    ) -> Option<(Entity, GizmoHandle)> {
        let primary = self.primary.iter().next();
        let seen = |camera: Entity| draws_to(self.targets.get(camera).ok(), primary, target);
        let parent_of = |target: Entity| {
            self.parents
                .get(target)
                .ok()
                .and_then(|child_of| self.globals.get(child_of.parent()).ok())
                .copied()
                .unwrap_or(GlobalTransform::IDENTITY)
        };
        nearest_handle(
            position,
            self.gizmos
                .iter()
                .filter(|(_, gizmo, ..)| seen(gizmo.camera)),
            self.joints
                .iter()
                .filter(|(_, gizmo, ..)| seen(gizmo.camera)),
            &self.cameras,
            &self.globals,
            parent_of,
        )
        .map(|candidate| candidate.grip())
    }
}

/// Where `entity` stands in the world as the transforms say now. Its
/// `GlobalTransform` lags a transform written earlier in the frame — a parent
/// a host moved this frame — until propagation in `PostUpdate`, so the chain
/// is composed here, falling back to the propagated transform of an ancestor
/// with none of its own.
///
/// A drag's local transform is taken against the parent as this says, and a
/// control reports the drag in the world against the same parent, so what is
/// applied, what is reported and what a driver writes back all agree.
pub(crate) fn current_global(
    mut entity: Entity,
    transform_of: impl Fn(Entity) -> Option<Transform>,
    parents: &Query<&ChildOf>,
    globals: &Query<&GlobalTransform>,
) -> Option<bevy::math::Affine3A> {
    let mut affine = bevy::math::Affine3A::IDENTITY;
    loop {
        let Some(transform) = transform_of(entity) else {
            return globals.get(entity).ok().map(|g| g.affine() * affine);
        };
        affine = transform.compute_affine() * affine;
        match parents.get(entity) {
            Ok(parent) => entity = parent.parent(),
            Err(_) => return Some(affine),
        }
    }
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
    windows: Query<(Entity, &Window)>,
    targets: Query<&bevy::camera::RenderTarget>,
    primary: Query<Entity, With<bevy::window::PrimaryWindow>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut pointer: ResMut<GizmoPointer>,
    mut drags: ResMut<GizmoDrags>,
    mut pressed_scene: ResMut<PressedScene>,
    mut held: Local<Option<Held>>,
) {
    // Taken every frame, so a press noted and not acted on cannot linger.
    let on_scene = std::mem::take(&mut pressed_scene.0);
    let (window, cursor) = windows
        .iter()
        .find_map(|(entity, window)| window.cursor_position().map(|cursor| (entity, cursor)))
        .unzip();
    // Only a gizmo seen in the window the cursor is in can be under it: one
    // seen through a camera drawing into an image — a screen's texture — is
    // nowhere on that window, whatever its handles project to.
    let primary = primary.iter().next();
    let in_window = window
        .and_then(|window| bevy::window::WindowRef::Entity(window).normalize(primary))
        .map(bevy::camera::NormalizedRenderTarget::Window);
    let seen = |camera: Entity| {
        in_window
            .as_ref()
            .is_some_and(|in_window| draws_to(targets.get(camera).ok(), primary, in_window))
    };
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
    let button_up = !buttons.pressed(MouseButton::Left);
    if button_up {
        pointer.ended = None;
    }

    // A host's cancel, asked between frames, is seen before the button is:
    // a drag cancelled sends nothing more, its release included.
    let cancel = std::mem::take(&mut drags.cancel);
    match *held {
        Some(Held::Transform(drag)) => {
            // A gizmo taken away mid-drag lets go as one gone inert does, so
            // its release is no click. There is no gizmo left to trigger a
            // `Cancel` on; the controls close such a drag for their own.
            let Ok((_, gizmo, visible, _)) = gizmos.get(drag.gizmo) else {
                *held = None;
                pointer.readings.clear();
                let_go(&mut pointer, button_up, false);
                return;
            };
            let live = takes_pointer(gizmo.interactive, visible);
            // Going inert or hidden with the button down abandons the drag;
            // only the button coming up ends what the pointer is doing (see
            // `button_up`), so a release in the frame the gizmo stops taking
            // the pointer still finishes the drag with `End`.
            if cancel || (!live && !button_up) {
                if gizmo.apply {
                    if let Ok(mut transform) = locals.get_mut(gizmo.target) {
                        if *transform != drag.start_local {
                            *transform = drag.start_local;
                        }
                    }
                }
                commands.trigger(GizmoDrag {
                    entity: drag.gizmo,
                    target: gizmo.target,
                    handle: drag.handle,
                    phase: DragPhase::Cancel,
                    gesture: drag.gesture,
                    transform: drag.start_local,
                });
                *held = None;
                pointer.readings.clear();
                let_go(&mut pointer, button_up, live);
                return;
            }
            let released = button_up;
            let moved = ray_through(gizmo.camera).and_then(|ray| dragged_to(&drag, ray));
            // Against the parent as it stands now, not as last propagated: a
            // parent moved this frame would put the target somewhere other
            // than where it was dragged.
            let parent = parent_now(gizmo.target, &parents, &locals, &globals);
            let mut last = drag.last;
            if let Some(moved) = moved {
                pointer.readings = readings(&drag, &moved);
                last = match moved {
                    Moved::World(global) => {
                        // Moving and turning leave the scale as it was; taking
                        // it back out of the world would only add rounding.
                        Transform {
                            scale: drag.start_local.scale,
                            ..global.reparented_to(&parent)
                        }
                    }
                    Moved::Stretch(stretch) => Transform {
                        scale: drag.start_local.scale * stretch,
                        ..drag.start_local
                    },
                };
                if gizmo.apply {
                    if let Ok(mut transform) = locals.get_mut(gizmo.target) {
                        *transform = last;
                    }
                }
            }
            // A move is sent only when what it reports would change — the
            // target's position and turn in the world, or its own scale —
            // since each asks a host to show the drag afresh. A parent moving
            // under a still pointer changes the pose; a target carried along
            // by its parent without moving in the world, or a parent rescaled
            // about a target at its origin, does not.
            let (_, turn, at) = parent.mul_transform(last).to_scale_rotation_translation();
            let reported = (at, turn);
            let phase = if released {
                Some(DragPhase::End)
            } else {
                (reported != drag.reported || last.scale != drag.last.scale)
                    .then_some(DragPhase::Move)
            };
            if let Some(phase) = phase {
                commands.trigger(GizmoDrag {
                    entity: drag.gizmo,
                    target: gizmo.target,
                    handle: drag.handle,
                    phase,
                    gesture: drag.gesture,
                    transform: last,
                });
            }
            if released {
                *held = None;
                pointer.readings.clear();
                let_go(&mut pointer, button_up, live);
            } else {
                *held = Some(Held::Transform(Drag {
                    last,
                    reported,
                    ..drag
                }));
            }
            return;
        }
        Some(Held::Joint {
            grip,
            gesture,
            start,
        }) => {
            // Taken away mid-drag: as for a transform gizmo, above.
            let Ok((_, mut gizmo, rest, visible)) = joints.get_mut(grip.gizmo) else {
                *held = None;
                let_go(&mut pointer, button_up, false);
                return;
            };
            // Inert or hidden: as for a transform gizmo, above.
            let live = takes_pointer(gizmo.interactive, visible);
            if cancel || (!live && !button_up) {
                // Back to the value the joint stands at without the drag. A
                // gizmo that poses its body itself left it where the drag
                // began; otherwise the body stands wherever its host last
                // put it, which a host may have moved on since the drag
                // began, so it is read from the body rather than taken from
                // the start.
                let back = if gizmo.apply {
                    start
                } else {
                    rest.filter(|rest| rest.1 == gizmo.target)
                        .zip(locals.get(gizmo.target).ok())
                        .map_or(start, |(rest, body)| value_of(&gizmo, &rest.0, body, start))
                };
                if gizmo.value != back {
                    gizmo.value = back;
                }
                commands.trigger(JointDrag {
                    entity: grip.gizmo,
                    target: gizmo.target,
                    phase: DragPhase::Cancel,
                    gesture,
                    value: back,
                });
                *held = None;
                let_go(&mut pointer, button_up, live);
                return;
            }
            let released = button_up;
            let value = ray_through(gizmo.camera).and_then(|ray| dragged(&gizmo, &grip, ray));
            let changed = value.is_some_and(|value| value != gizmo.value);
            if let Some(value) = value.filter(|_| changed) {
                gizmo.value = value;
            }
            let phase = if released {
                Some(DragPhase::End)
            } else {
                changed.then_some(DragPhase::Move)
            };
            if let Some(phase) = phase {
                commands.trigger(JointDrag {
                    entity: grip.gizmo,
                    target: gizmo.target,
                    phase,
                    gesture,
                    value: gizmo.value,
                });
            }
            if released {
                *held = None;
                let_go(&mut pointer, button_up, live);
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
    let Some(candidate) = nearest_handle(
        cursor,
        gizmos.iter().filter(|(_, gizmo, ..)| seen(gizmo.camera)),
        joints.iter().filter(|(_, joint, ..)| seen(joint.camera)),
        &cameras,
        &globals,
        parent_of,
    ) else {
        return;
    };
    let hovered = candidate.grip();
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
            let gesture = drags.take_gesture();
            *held = Some(Held::Transform(Drag {
                gizmo: entity,
                handle,
                gesture,
                origin: place.origin,
                axes: place.axes,
                stretch_along: nearest_component(start_global.rotation(), axis),
                grabbed,
                plane,
                start_global: *start_global,
                start_local: *start_local,
                last: *start_local,
                reported: {
                    let (_, turn, at) = parent_now(gizmo.target, &parents, &locals, &globals)
                        .mul_transform(*start_local)
                        .to_scale_rotation_translation();
                    (at, turn)
                },
            }));
            commands.trigger(GizmoDrag {
                entity,
                target: gizmo.target,
                handle,
                phase: DragPhase::Start,
                gesture,
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
            let gesture = drags.take_gesture();
            *held = Some(Held::Joint {
                grip,
                gesture,
                start: gizmo.value,
            });
            commands.trigger(JointDrag {
                entity,
                target: gizmo.target,
                phase: DragPhase::Start,
                gesture,
                value: gizmo.value,
            });
        }
    }
    pointer.dragging = Some(hovered);
}

/// Whether a gizmo holding a drag still takes the pointer: it must be
/// `interactive` and visible, as it must be to be hovered or taken. A gizmo
/// hidden by anything — its control, or an ancestor's `Visibility` set by a
/// host — cannot be seen to be dragged, so a drag in hand on it is
/// abandoned. Visibility is propagated in `PostUpdate`, so a gizmo hidden
/// in one frame lets go of its drag in the next; a drag carries on for that
/// one frame, which no one can see.
fn takes_pointer(interactive: bool, visible: &InheritedVisibility) -> bool {
    interactive && visible.get()
}

/// Ends the drag in hand. One its gizmo ended by going inert or hidden, with
/// the button still down, stays the pointer's as `ended` until the button
/// comes up, so its release is not taken for a click on what lies beneath.
///
/// A gizmo gone inert or hidden is under the pointer no more either: the
/// handle held stays `hovered` through a drag wherever the pointer goes, and
/// left there it would read as hovered, inert or unseen, and perhaps far
/// from the pointer, until the next frame looks again.
fn let_go(pointer: &mut GizmoPointer, button_up: bool, interactive: bool) {
    let dragging = pointer.dragging.take();
    if !button_up {
        pointer.ended = dragging;
    }
    if !interactive {
        pointer.hovered = None;
    }
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
///
/// Every write is skipped when it would not change the value: a still view
/// then leaves the gizmo's transforms and visibilities unchanged, so nothing
/// downstream re-extracts them, and a host that draws only on change sees no
/// change.
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
    mut handle_materials: ResMut<Assets<HandleMaterial>>,
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
            let toward = sides(
                gizmo.face_viewer,
                facing.0,
                place.axes,
                place.origin,
                camera.1.translation(),
            );
            if toward != facing.0 {
                facing.0 = toward;
            }
            place.facing = toward;
        }
        transform.set_if_neq(
            Transform::from_translation(place.origin).with_scale(Vec3::splat(place.unit)),
        );
        global.set_if_neq(GlobalTransform::from(*transform));
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
            at.set_if_neq(Transform {
                translation: turn * (part.offset * flip),
                rotation: turn * Quat::from_rotation_arc(Vec3::Y, Vec3::X * flip.x),
                scale: Vec3::ONE,
            });
            part_global.set_if_neq(global.mul_transform(*at));
            let shown = part.handle.mode().is_some_and(|mode| gizmo.shows(mode))
                && !(part.stem && gizmo.shows(TransformMode::Translate));
            visibility.set_if_neq(if shown {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            });
            if part.faded != gizmo.opacity {
                part.faded = gizmo.opacity;
                for (material, alpha) in part.materials.iter().zip(part.alphas) {
                    if let Some(mut material) = handle_materials.get_mut(material) {
                        material.handle.colour.w = alpha * gizmo.opacity;
                    }
                }
            }
            let is_hot = looks_hot(hot, entity, gizmo, part.handle);
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
            colour: gizmo.colours[axis].with_alpha(gizmo.colours[axis].alpha() * gizmo.opacity),
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
            guide_global.set_if_neq(*global);
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

    /// The phases of every drag event, of either kind of gizmo, in order.
    #[derive(Resource, Default)]
    struct Phases(Vec<DragPhase>);

    /// The gesture of every drag event, in the same order.
    #[derive(Resource, Default)]
    struct Gestures(Vec<u32>);

    /// A world that runs only [`drive`]: a target at the origin seen square
    /// on by a camera drawing 400 pixels a side, a window to carry the
    /// cursor, and the gizmo `spawn` makes for the target and camera. The
    /// left button is pressed and let go by hand, with nothing clearing it
    /// between frames but the test.
    fn driving(spawn: impl FnOnce(&mut World, Entity, Entity) -> Entity) -> (App, Entity, Entity) {
        use bevy::camera::{CameraProjection, ComputedCameraValues, RenderTargetInfo};

        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<ButtonInput<MouseButton>>()
            .init_resource::<GizmoPointer>()
            .init_resource::<PressedScene>()
            .init_resource::<GizmoDrags>()
            .init_resource::<Phases>()
            .init_resource::<Gestures>()
            .add_observer(
                |drag: On<GizmoDrag>,
                 mut phases: ResMut<Phases>,
                 mut gestures: ResMut<Gestures>| {
                    phases.0.push(drag.phase);
                    gestures.0.push(drag.gesture);
                },
            )
            .add_observer(
                |drag: On<JointDrag>,
                 mut phases: ResMut<Phases>,
                 mut gestures: ResMut<Gestures>| {
                    phases.0.push(drag.phase);
                    gestures.0.push(drag.gesture);
                },
            )
            .add_systems(Update, drive);
        let world = app.world_mut();
        let window = world.spawn(Window::default()).id();
        let camera = world
            .spawn((
                Camera {
                    computed: ComputedCameraValues {
                        clip_from_view: PerspectiveProjection {
                            aspect_ratio: 1.0,
                            ..default()
                        }
                        .get_clip_from_view(),
                        target_info: Some(RenderTargetInfo {
                            physical_size: UVec2::splat(400),
                            scale_factor: 1.0,
                        }),
                        ..default()
                    },
                    ..default()
                },
                GlobalTransform::from(
                    Transform::from_xyz(0.0, -1.5, 0.4).looking_at(Vec3::ZERO, Vec3::Z),
                ),
                bevy::camera::RenderTarget::Window(bevy::window::WindowRef::Entity(window)),
            ))
            .id();
        let target = world
            .spawn((Transform::IDENTITY, GlobalTransform::IDENTITY))
            .id();
        let gizmo = spawn(world, target, camera);
        world.entity_mut(gizmo).insert(InheritedVisibility::VISIBLE);
        (app, window, gizmo)
    }

    /// A gizmo seen through a camera drawing into an image — a screen's
    /// texture — is nowhere on the window, so the cursor there finds none of
    /// its handles, wherever they project to.
    #[test]
    fn a_gizmo_drawn_into_an_image_is_not_under_the_cursor() {
        let under_cursor = |into_image: bool| {
            let (mut app, window, gizmo) = driving(|world, target, camera| {
                world
                    .spawn(TransformGizmo::new(target, camera))
                    .insert(Facing(Vec3::ONE))
                    .id()
            });
            if into_image {
                let camera = app.world().get::<TransformGizmo>(gizmo).unwrap().camera;
                app.world_mut()
                    .entity_mut(camera)
                    .insert(bevy::camera::RenderTarget::Image(
                        bevy::camera::ImageRenderTarget::from(Handle::<Image>::default()),
                    ));
            }
            (0..200).any(|x| {
                let mut held = app.world_mut().get_mut::<Window>(window).unwrap();
                held.set_cursor_position(Some(Vec2::new(200.0 + x as f32, 200.0)));
                app.update();
                app.world().resource::<GizmoPointer>().hovered.is_some()
            })
        };
        assert!(under_cursor(false), "seen in the window, a handle is found");
        assert!(!under_cursor(true));
    }

    /// How a test abandons a drag in hand.
    #[derive(Clone, Copy, PartialEq)]
    enum Abandon {
        /// The gizmo is made inert.
        Inert,
        /// The host cancels it.
        Host,
        /// The gizmo is hidden, as an ancestor's `Visibility` hides it. The
        /// harness propagates no visibility, so the gizmo's own
        /// `InheritedVisibility` is set, as propagation would set it.
        Hidden,
    }

    /// A drag abandoned part way through — its gizmo made inert or hidden,
    /// or cancelled by its host — closes at once with one `Cancel`, puts a
    /// target the gizmo moves back where the drag found it, and sends nothing
    /// after it while the button stays down, nor when it comes up. Nothing is
    /// held after it, so the overlay and the guides let go, but the button
    /// stays the gizmo's until it comes up, which is what keeps a pick on the
    /// release from taking it for a click on the scene. A hidden gizmo is not
    /// hovered until it shows again. A fresh press starts a drag with a
    /// gesture of its own. Holding still sends no move.
    ///
    /// `at_start` says whether the gizmo's target stands where the drag found
    /// it.
    fn abandoned_mid_drag(
        spawn: impl FnOnce(&mut World, Entity, Entity) -> Entity,
        set_interactive: fn(&mut World, Entity, bool),
        at_start: fn(&World, Entity) -> bool,
        how: Abandon,
    ) {
        let (mut app, window, gizmo) = driving(spawn);
        let point_at = |app: &mut App, at: Vec2| {
            let mut window = app.world_mut().get_mut::<Window>(window).unwrap();
            window.set_cursor_position(Some(at));
            app.update();
        };
        fn buttons(app: &mut App) -> Mut<'_, ButtonInput<MouseButton>> {
            app.world_mut().resource_mut::<ButtonInput<MouseButton>>()
        }
        let pointer = |app: &App| {
            let pointer = app.world().resource::<GizmoPointer>();
            (pointer.hovered, pointer.dragging, pointer.owns_button())
        };
        let phases = |app: &App| app.world().resource::<Phases>().0.clone();
        let press = |app: &mut App| {
            app.world_mut().resource_mut::<PressedScene>().0 = true;
            buttons(app).press(MouseButton::Left);
            app.update();
            // Still down, no longer just pressed.
            buttons(app).clear();
        };

        // Along the row through the target until a handle is under the
        // cursor; not a ring, which is seen edge on there and so cannot be
        // taken.
        let spot = (0..200)
            .map(|x| Vec2::new(200.0 + x as f32, 200.0))
            .find(|&at| {
                point_at(&mut app, at);
                pointer(&app).0.is_some_and(|(entity, handle)| {
                    entity == gizmo && !matches!(handle, GizmoHandle::Ring(_))
                })
            })
            .expect("a handle on the row through the target");

        press(&mut app);
        assert_eq!(phases(&app), [DragPhase::Start]);
        assert!(pointer(&app).1.is_some(), "the drag is the pointer's");
        point_at(&mut app, spot);
        point_at(&mut app, spot);
        assert_eq!(phases(&app), [DragPhase::Start], "no move while held still");
        point_at(&mut app, spot + Vec2::new(4.0, -3.0));
        assert_eq!(phases(&app).last(), Some(&DragPhase::Move));
        assert!(!at_start(app.world(), gizmo), "the drag moved its target");

        match how {
            Abandon::Inert => set_interactive(app.world_mut(), gizmo, false),
            Abandon::Host => app.world_mut().resource_mut::<GizmoDrags>().cancel_drag(),
            Abandon::Hidden => {
                app.world_mut()
                    .entity_mut(gizmo)
                    .insert(InheritedVisibility::HIDDEN);
            }
        }
        app.update();
        let (hovered, dragging, owned) = pointer(&app);
        assert_eq!(
            (dragging, owned),
            (None, true),
            "in the frame it ends: not held, the button still the gizmo's"
        );
        if how != Abandon::Host {
            assert_eq!(
                hovered, None,
                "a gizmo that stopped taking the pointer is under it no more"
            );
        }
        let ended = phases(&app);
        assert_eq!(ended.last(), Some(&DragPhase::Cancel), "{ended:?}");
        assert!(
            !ended.contains(&DragPhase::End),
            "abandoned, not finished: {ended:?}"
        );
        assert!(
            at_start(app.world(), gizmo),
            "put back where the drag found it"
        );
        let gestures = app.world().resource::<Gestures>().0.clone();
        assert!(
            gestures.iter().all(|&gesture| gesture == gestures[0]),
            "one drag, one gesture: {gestures:?}"
        );
        for at in [Vec2::new(8.0, -6.0), Vec2::ZERO, Vec2::new(-5.0, 2.0)] {
            point_at(&mut app, spot + at);
            assert_eq!(phases(&app), ended, "nothing more while the button is down");
            let (_, dragging, owned) = pointer(&app);
            assert_eq!(dragging, None, "nothing is held once the drag has ended");
            assert!(owned, "the button stays the gizmo's until it comes up");
        }

        buttons(&mut app).release(MouseButton::Left);
        app.update();
        buttons(&mut app).clear();
        assert_eq!(phases(&app), ended, "the release ends nothing more");
        assert_eq!(pointer(&app).1, None);
        assert!(!pointer(&app).2, "let go");

        match how {
            Abandon::Inert => set_interactive(app.world_mut(), gizmo, true),
            Abandon::Host => {}
            Abandon::Hidden => {
                point_at(&mut app, spot);
                assert!(pointer(&app).0.is_none(), "a hidden gizmo is not hovered");
                app.world_mut()
                    .entity_mut(gizmo)
                    .insert(InheritedVisibility::VISIBLE);
            }
        }
        point_at(&mut app, spot);
        assert!(pointer(&app).0.is_some(), "live under the cursor");
        press(&mut app);
        let again = phases(&app);
        assert_eq!(again.len(), ended.len() + 1);
        assert_eq!(again.last(), Some(&DragPhase::Start), "a fresh drag");
        let fresh = *app.world().resource::<Gestures>().0.last().unwrap();
        assert_ne!(fresh, gestures[0], "a gesture of its own");

        // An ordinary release frees the button in the frame it comes up, so
        // the very next press can be the view's.
        buttons(&mut app).release(MouseButton::Left);
        app.update();
        buttons(&mut app).clear();
        assert_eq!(phases(&app).last(), Some(&DragPhase::End));
        let (hovered, _, owned) = pointer(&app);
        assert!(!owned, "the button is free once it is up");
        assert!(
            hovered.is_some(),
            "and the handle let go of is still hovered"
        );
    }

    /// A gizmo taken away mid-drag lets go of the drag and of what it read,
    /// but the button stays its until it comes up, so the release is no
    /// click on the scene. `reads` is whether the drag shows readings, as a
    /// transform gizmo's does.
    fn taken_away_mid_drag(spawn: impl FnOnce(&mut World, Entity, Entity) -> Entity, reads: bool) {
        let (mut app, window, gizmo) = driving(spawn);
        let point_at = |app: &mut App, at: Vec2| {
            let mut cursor = app.world_mut().get_mut::<Window>(window).unwrap();
            cursor.set_cursor_position(Some(at));
            app.update();
        };
        let spot = (0..200)
            .map(|x| Vec2::new(200.0 + x as f32, 200.0))
            .find(|&at| {
                point_at(&mut app, at);
                app.world()
                    .resource::<GizmoPointer>()
                    .hovered
                    .is_some_and(|(_, handle)| !matches!(handle, GizmoHandle::Ring(_)))
            })
            .expect("a handle on the row through the target");
        app.world_mut().resource_mut::<PressedScene>().0 = true;
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Left);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .clear();
        point_at(&mut app, spot + Vec2::new(6.0, -4.0));
        let pointer = app.world().resource::<GizmoPointer>();
        assert!(pointer.dragging.is_some());
        assert_eq!(
            !pointer.readings.is_empty(),
            reads,
            "{:?}",
            pointer.readings
        );

        let sent = app.world().resource::<Phases>().0.len();
        app.world_mut().despawn(gizmo);
        app.update();
        assert_eq!(
            app.world().resource::<Phases>().0.len(),
            sent,
            "nothing is left to report on; the controls close such a drag"
        );
        let pointer = app.world().resource::<GizmoPointer>();
        assert_eq!((pointer.hovered, pointer.dragging), (None, None));
        assert!(pointer.readings.is_empty(), "nothing left to read");
        assert!(pointer.owns_button(), "the button is still the gizmo's");

        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .release(MouseButton::Left);
        app.update();
        assert!(!app.world().resource::<GizmoPointer>().owns_button());
    }

    /// A target whose parent moved this frame, not yet propagated, is moved
    /// where it was dragged: its local transform is taken against the parent
    /// as it now stands, so the parent's move is not added to the drag.
    #[test]
    fn a_drag_is_taken_against_the_parent_as_it_now_stands() {
        #[derive(Resource, Default)]
        struct Dragged(Vec<Transform>);

        let (mut app, window, gizmo) = driving(|world, target, camera| {
            // Propagated at the origin, standing ten metres along X now.
            let parent = world
                .spawn((
                    Transform::from_xyz(10.0, 0.0, 0.0),
                    GlobalTransform::IDENTITY,
                ))
                .id();
            world.entity_mut(target).insert(ChildOf(parent));
            world
                .spawn(TransformGizmo::new(target, camera))
                .insert(Facing(Vec3::ONE))
                .id()
        });
        app.init_resource::<Dragged>().add_observer(
            |drag: On<GizmoDrag>, mut dragged: ResMut<Dragged>| dragged.0.push(drag.transform),
        );
        let point_at = |app: &mut App, at: Vec2| {
            let mut cursor = app.world_mut().get_mut::<Window>(window).unwrap();
            cursor.set_cursor_position(Some(at));
            app.update();
        };
        let spot = (0..200)
            .map(|x| Vec2::new(200.0 + x as f32, 200.0))
            .find(|&at| {
                point_at(&mut app, at);
                app.world()
                    .resource::<GizmoPointer>()
                    .hovered
                    .is_some_and(|(_, handle)| !matches!(handle, GizmoHandle::Ring(_)))
            })
            .expect("a handle on the row through the target");
        app.world_mut().resource_mut::<PressedScene>().0 = true;
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Left);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .clear();
        point_at(&mut app, spot + Vec2::new(6.0, -4.0));

        let target = app.world().get::<TransformGizmo>(gizmo).unwrap().target;
        let moved = *app.world().resource::<Dragged>().0.last().unwrap();
        // Dragged a little way from the origin, where the target was drawn:
        // under a parent ten metres along X, that is about ten metres back.
        let world = Vec3::new(10.0, 0.0, 0.0) + moved.translation;
        assert!(world.length() < 1.0, "landed at {world} in the world");
        assert_eq!(
            app.world().get::<Transform>(target).unwrap().translation,
            moved.translation,
            "and applied as reported"
        );
    }

    #[test]
    fn a_transform_gizmo_taken_away_mid_drag_keeps_the_release() {
        taken_away_mid_drag(
            |world, target, camera| {
                world
                    .spawn(TransformGizmo::new(target, camera))
                    .insert(Facing(Vec3::ONE))
                    .id()
            },
            true,
        );
    }

    #[test]
    fn a_joint_gizmo_taken_away_mid_drag_keeps_the_release() {
        taken_away_mid_drag(
            |world, target, camera| {
                world
                    .spawn(JointGizmo::new(
                        target,
                        camera,
                        crate::joint_gizmo::JointMotion::Turn,
                        Dir3::Z,
                        -3.0,
                        3.0,
                        0.0,
                        Color::WHITE,
                    ))
                    .insert(crate::joint_gizmo::Rest(Transform::IDENTITY, target))
                    .id()
            },
            false,
        );
    }

    fn transform_gizmo(world: &mut World, target: Entity, camera: Entity) -> Entity {
        world
            .spawn(TransformGizmo::new(target, camera))
            .insert(Facing(Vec3::ONE))
            .id()
    }

    fn transform_at_start(world: &World, gizmo: Entity) -> bool {
        let target = world.get::<TransformGizmo>(gizmo).unwrap().target;
        *world.get::<Transform>(target).unwrap() == Transform::IDENTITY
    }

    fn joint_gizmo(world: &mut World, target: Entity, camera: Entity) -> Entity {
        world
            .spawn(JointGizmo::new(
                target,
                camera,
                crate::joint_gizmo::JointMotion::Turn,
                Dir3::Z,
                -3.0,
                3.0,
                0.0,
                Color::WHITE,
            ))
            .insert(crate::joint_gizmo::Rest(Transform::IDENTITY, target))
            .id()
    }

    fn joint_at_start(world: &World, gizmo: Entity) -> bool {
        world.get::<JointGizmo>(gizmo).unwrap().value == 0.0
    }

    #[test]
    fn a_transform_gizmo_made_inert_mid_drag_cancels_it_and_keeps_the_release() {
        abandoned_mid_drag(
            transform_gizmo,
            |world, gizmo, on| world.get_mut::<TransformGizmo>(gizmo).unwrap().interactive = on,
            transform_at_start,
            Abandon::Inert,
        );
    }

    #[test]
    fn a_joint_gizmo_made_inert_mid_drag_cancels_it_and_keeps_the_release() {
        abandoned_mid_drag(
            joint_gizmo,
            |world, gizmo, on| world.get_mut::<JointGizmo>(gizmo).unwrap().interactive = on,
            joint_at_start,
            Abandon::Inert,
        );
    }

    #[test]
    fn a_transform_drag_its_host_cancels_closes_once_and_keeps_the_release() {
        abandoned_mid_drag(
            transform_gizmo,
            |world, gizmo, on| world.get_mut::<TransformGizmo>(gizmo).unwrap().interactive = on,
            transform_at_start,
            Abandon::Host,
        );
    }

    #[test]
    fn a_joint_drag_its_host_cancels_closes_once_and_keeps_the_release() {
        abandoned_mid_drag(
            joint_gizmo,
            |world, gizmo, on| world.get_mut::<JointGizmo>(gizmo).unwrap().interactive = on,
            joint_at_start,
            Abandon::Host,
        );
    }

    #[test]
    fn a_transform_gizmo_hidden_mid_drag_cancels_it_and_keeps_the_release() {
        abandoned_mid_drag(
            transform_gizmo,
            |world, gizmo, on| world.get_mut::<TransformGizmo>(gizmo).unwrap().interactive = on,
            transform_at_start,
            Abandon::Hidden,
        );
    }

    #[test]
    fn a_joint_gizmo_hidden_mid_drag_cancels_it_and_keeps_the_release() {
        abandoned_mid_drag(
            joint_gizmo,
            |world, gizmo, on| world.get_mut::<JointGizmo>(gizmo).unwrap().interactive = on,
            joint_at_start,
            Abandon::Hidden,
        );
    }

    /// A move is sent when what a drag reports changes: a stretch held still
    /// while the target's parent moves puts the target somewhere new in the
    /// world, which is reported; a move held still while the parent moves
    /// leaves it where it was in the world, which is not.
    #[test]
    fn a_held_drag_reports_its_parent_moving_it_in_the_world() {
        for stretch in [true, false] {
            let (mut app, window, gizmo) = driving(transform_gizmo);
            let target = app.world().get::<TransformGizmo>(gizmo).unwrap().target;
            let parent = app
                .world_mut()
                .spawn((Transform::IDENTITY, GlobalTransform::IDENTITY))
                .id();
            app.world_mut().entity_mut(target).insert(ChildOf(parent));
            let point_at = |app: &mut App, at: Vec2| {
                let mut cursor = app.world_mut().get_mut::<Window>(window).unwrap();
                cursor.set_cursor_position(Some(at));
                app.update();
            };
            let wanted = |handle: GizmoHandle| match handle {
                GizmoHandle::Ball(_) => stretch,
                GizmoHandle::Arrow(_) => !stretch,
                _ => false,
            };
            let spot = (0..80)
                .flat_map(|y| (0..80).map(move |x| Vec2::new(x as f32 * 5.0, y as f32 * 5.0)))
                .find(|&at| {
                    point_at(&mut app, at);
                    app.world()
                        .resource::<GizmoPointer>()
                        .hovered
                        .is_some_and(|(_, handle)| wanted(handle))
                })
                .expect("the handle somewhere in view");
            app.world_mut().resource_mut::<PressedScene>().0 = true;
            app.world_mut()
                .resource_mut::<ButtonInput<MouseButton>>()
                .press(MouseButton::Left);
            app.update();
            app.world_mut()
                .resource_mut::<ButtonInput<MouseButton>>()
                .clear();
            point_at(&mut app, spot);
            let still = app.world().resource::<Phases>().0.len();

            *app.world_mut().get_mut::<Transform>(parent).unwrap() =
                Transform::from_xyz(0.0, 0.0, 0.3);
            point_at(&mut app, spot);
            let phases = app.world().resource::<Phases>().0.clone();
            if stretch {
                assert_eq!(phases.len(), still + 1, "stretch: {phases:?}");
                assert_eq!(phases.last(), Some(&DragPhase::Move));
            } else {
                assert_eq!(phases.len(), still, "move: {phases:?}");
            }

            // Rescaling the parent about the target, at its origin, changes
            // nothing a drag reports.
            if stretch {
                *app.world_mut().get_mut::<Transform>(parent).unwrap() = Transform::IDENTITY;
                point_at(&mut app, spot);
                let before = app.world().resource::<Phases>().0.len();
                app.world_mut().get_mut::<Transform>(parent).unwrap().scale = Vec3::splat(2.0);
                point_at(&mut app, spot);
                assert_eq!(
                    app.world().resource::<Phases>().0.len(),
                    before,
                    "a rescaled parent alone moves nothing reported"
                );
            }
        }
    }

    /// A joint gizmo that leaves its body to its host goes back, on a cancel,
    /// to where the body stands — the value its host last wrote, which may
    /// have moved on since the drag began — not to the drag's start.
    #[test]
    fn a_cancelled_joint_shows_what_its_host_last_wrote() {
        let (mut app, window, gizmo) = driving(|world, target, camera| {
            let gizmo = joint_gizmo(world, target, camera);
            world.get_mut::<JointGizmo>(gizmo).unwrap().apply = false;
            gizmo
        });
        let point_at = |app: &mut App, at: Vec2| {
            let mut cursor = app.world_mut().get_mut::<Window>(window).unwrap();
            cursor.set_cursor_position(Some(at));
            app.update();
        };
        let spot = (0..200)
            .map(|x| Vec2::new(200.0 + x as f32, 200.0))
            .find(|&at| {
                point_at(&mut app, at);
                app.world().resource::<GizmoPointer>().hovered.is_some()
            })
            .expect("the ring on the row through the target");
        app.world_mut().resource_mut::<PressedScene>().0 = true;
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Left);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .clear();
        point_at(&mut app, spot + Vec2::new(-30.0, 20.0));
        let dragged = app.world().get::<JointGizmo>(gizmo).unwrap().value;
        assert!(dragged != 0.0, "the drag turned the joint");

        // The host previews a value of its own on the body.
        let target = app.world().get::<JointGizmo>(gizmo).unwrap().target;
        *app.world_mut().get_mut::<Transform>(target).unwrap() =
            Transform::from_rotation(Quat::from_rotation_z(0.7));
        app.world_mut().resource_mut::<GizmoDrags>().cancel_drag();
        app.update();
        let value = app.world().get::<JointGizmo>(gizmo).unwrap().value;
        assert!((value - 0.7).abs() < 1e-5, "{value}");
    }

    /// A press with a button other than the primary on a handle is the
    /// host's, by the handle under the press, whatever the last frame left
    /// hovered; the primary's never is, nor a press over no handle, nor one
    /// during a drag.
    #[test]
    fn only_another_buttons_press_on_a_handle_is_the_hosts() {
        let [gizmo, stale] = [7, 8].map(|i| Entity::from_raw_u32(i).unwrap());
        let ring = (gizmo, GizmoHandle::Ring(2));
        let left_over = GizmoPointer {
            hovered: Some((stale, GizmoHandle::Arrow(0))),
            ..default()
        };
        assert_eq!(
            hosts_press(PointerButton::Secondary, &left_over, || Some(ring)),
            Some(ring),
            "the handle under the press, not the one last hovered"
        );
        assert_eq!(
            hosts_press(PointerButton::Middle, &GizmoPointer::default(), || Some(
                ring
            )),
            Some(ring)
        );
        assert_eq!(
            hosts_press(PointerButton::Secondary, &left_over, || None),
            None,
            "moved off the handle hovered last frame"
        );
        assert_eq!(
            hosts_press(PointerButton::Primary, &GizmoPointer::default(), || Some(
                ring
            )),
            None
        );
        let held = GizmoPointer {
            hovered: Some(ring),
            dragging: Some(ring),
            ..default()
        };
        assert_eq!(
            hosts_press(PointerButton::Secondary, &held, || Some(ring)),
            None,
            "a press during a drag is on no handle"
        );
        let ended = GizmoPointer {
            ended: Some(ring),
            ..default()
        };
        assert_eq!(
            hosts_press(PointerButton::Secondary, &ended, || Some(ring)),
            None,
            "nor while the button of a drag ended by going inert is down"
        );
    }

    /// The view takes a press only when no gizmo has it: not while the
    /// button of a drag in hand, or of one ended by going inert, is down,
    /// nor a host's press on a handle.
    #[test]
    fn the_view_takes_a_press_only_when_no_gizmo_has_it() {
        let ring = (Entity::from_raw_u32(7).unwrap(), GizmoHandle::Ring(2));
        let free = GizmoPointer::default();
        let held = GizmoPointer {
            dragging: Some(ring),
            ..default()
        };
        let ended = GizmoPointer {
            ended: Some(ring),
            ..default()
        };
        assert!(!keeps_press(PointerButton::Primary, &free, || Some(ring)));
        assert!(!keeps_press(PointerButton::Secondary, &free, || None));
        assert!(keeps_press(PointerButton::Secondary, &free, || Some(ring)));
        for button in [PointerButton::Primary, PointerButton::Secondary] {
            assert!(keeps_press(button, &held, || None), "{button:?}, held");
            assert!(keeps_press(button, &ended, || None), "{button:?}, ended");
        }
    }

    /// A press triggers no `GizmoPress` when nothing is under it, though the
    /// last frame left a handle hovered: the observer asks where the press
    /// is, rather than trusting a hover the pointer may have left since.
    #[test]
    fn a_press_over_nothing_triggers_no_gizmo_press_whatever_was_hovered() {
        use bevy::camera::NormalizedRenderTarget;
        use bevy::picking::backend::HitData;
        use bevy::picking::events::Pointer;
        use bevy::picking::pointer::{Location, PointerId};
        use bevy::window::WindowRef;

        #[derive(Resource, Default)]
        struct Heard(usize);

        let mut world = World::new();
        world.init_resource::<Heard>();
        world.add_observer(report_other_press);
        world.add_observer(|_: On<GizmoPress>, mut heard: ResMut<Heard>| heard.0 += 1);
        let window = world.spawn(Window::default()).id();
        let gizmo = world.spawn_empty().id();
        world.insert_resource(GizmoPointer {
            hovered: Some((gizmo, GizmoHandle::Arrow(0))),
            ..default()
        });
        let target =
            NormalizedRenderTarget::Window(WindowRef::Entity(window).normalize(None).unwrap());
        world.trigger(PointerPress {
            entity: window,
            pointer: Pointer::new(
                PointerId::Mouse,
                Location {
                    target,
                    position: Vec2::new(5.0, 5.0),
                },
            ),
            button: PointerButton::Secondary,
            hit: HitData::new(window, 0.0, None, None),
            count: 1,
        });
        world.flush();
        assert_eq!(world.resource::<Heard>().0, 0);
    }

    /// A gizmo is under a pointer only through a camera drawing to the
    /// pointer's window: the primary window for a camera that names it, the
    /// window itself for one that names it by entity, and never an image.
    #[test]
    fn a_gizmo_is_seen_only_in_its_camera_s_window() {
        use bevy::camera::{ImageRenderTarget, NormalizedRenderTarget, RenderTarget};
        use bevy::window::WindowRef;

        let [primary, other] = [1, 2].map(|i| Entity::from_raw_u32(i).unwrap());
        let window = |entity| {
            NormalizedRenderTarget::Window(WindowRef::Entity(entity).normalize(None).unwrap())
        };
        let to_primary = RenderTarget::Window(WindowRef::Primary);
        let to_other = RenderTarget::Window(WindowRef::Entity(other));
        assert!(draws_to(Some(&to_primary), Some(primary), &window(primary)));
        assert!(!draws_to(Some(&to_primary), Some(primary), &window(other)));
        assert!(draws_to(Some(&to_other), Some(primary), &window(other)));
        assert!(!draws_to(Some(&to_other), Some(primary), &window(primary)));
        assert!(!draws_to(None, Some(primary), &window(primary)));
        let image = RenderTarget::Image(ImageRenderTarget::from(Handle::<Image>::default()));
        assert!(!draws_to(Some(&image), Some(primary), &window(primary)));
    }

    /// A highlighted handle looks hot without the pointer; the others, and
    /// another gizmo's, look as the pointer leaves them.
    #[test]
    fn a_highlighted_handle_looks_hot() {
        let [one, other] = [1, 2].map(|i| Entity::from_raw_u32(i).unwrap());
        let mut gizmo = TransformGizmo::new(one, one);
        assert!(!looks_hot(None, one, &gizmo, GizmoHandle::Ring(1)));
        gizmo.highlight = vec![GizmoHandle::Ring(1)];
        assert!(looks_hot(None, one, &gizmo, GizmoHandle::Ring(1)));
        assert!(!looks_hot(None, one, &gizmo, GizmoHandle::Arrow(1)));
        assert!(looks_hot(
            Some((one, GizmoHandle::Arrow(1))),
            one,
            &gizmo,
            GizmoHandle::Arrow(1)
        ));
        assert!(!looks_hot(
            Some((other, GizmoHandle::Arrow(1))),
            one,
            &TransformGizmo::new(one, one),
            GizmoHandle::Arrow(1)
        ));
    }

    /// A handle reads back from the JSON it is written as, so a host can name
    /// the one it means.
    #[test]
    fn a_handle_reads_back_from_its_json() {
        for handle in [
            GizmoHandle::Arrow(0),
            GizmoHandle::Plane(1),
            GizmoHandle::Ring(2),
            GizmoHandle::Ball(0),
            GizmoHandle::Joint,
        ] {
            let json = serde_json::to_string(&handle).unwrap();
            assert_eq!(serde_json::from_str::<GizmoHandle>(&json).unwrap(), handle);
        }
    }

    /// A drag reads in its handle's own measure: an arrow's move along its
    /// axis, a plane's move along each of its two, a ring's turn about its
    /// axis either way, and a ball's stretch.
    #[test]
    fn a_drag_reads_in_its_handles_measure() {
        let start = |handle| drag(handle, Vec3::ZERO, Vec3::Y);
        let moved_to = |translation: Vec3, rotation: Quat| {
            Moved::World(GlobalTransform::from(
                Transform::from_translation(translation).with_rotation(rotation),
            ))
        };
        let amounts = |handle, moved: Moved| {
            readings(&start(handle), &moved)
                .into_iter()
                .map(|reading| (reading.axis, reading.amount, reading.unit))
                .collect::<Vec<_>>()
        };
        let offset = Vec3::new(0.3, -0.2, 0.5);
        assert_eq!(
            amounts(GizmoHandle::Arrow(0), moved_to(offset, Quat::IDENTITY)),
            vec![(Some(0), 0.3, ReadingUnit::Metres)]
        );
        assert_eq!(
            amounts(GizmoHandle::Plane(1), moved_to(offset, Quat::IDENTITY)),
            vec![
                (Some(0), 0.3, ReadingUnit::Metres),
                (Some(2), 0.5, ReadingUnit::Metres),
            ]
        );
        let turned = amounts(
            GizmoHandle::Ring(2),
            moved_to(Vec3::ZERO, Quat::from_rotation_z(-0.5)),
        );
        assert_eq!(turned.len(), 1);
        assert!(
            (turned[0].1 - (-0.5_f32).to_degrees()).abs() < 1e-3,
            "{turned:?}"
        );
        assert_eq!(
            amounts(
                GizmoHandle::Ball(0),
                Moved::Stretch(Vec3::new(1.5, 1.0, 1.0))
            ),
            vec![(Some(0), 1.5, ReadingUnit::Factor)]
        );
    }

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
            .init_resource::<Assets<HandleMaterial>>()
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
            gesture: 0,
            origin: Vec3::ZERO,
            axes: Quat::IDENTITY,
            stretch_along: axis_vector(handle.axis()),
            grabbed,
            plane,
            start_global: GlobalTransform::IDENTITY,
            start_local: Transform::IDENTITY,
            last: Transform::IDENTITY,
            reported: (Vec3::ZERO, Quat::IDENTITY),
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

    /// Unless told to face the viewer, every handle stays on its axis's
    /// positive side, whichever side the eye is on and whatever it had.
    #[test]
    fn handles_stay_positive_unless_told_to_face_the_viewer() {
        let eye = Vec3::new(-5.0, -5.0, -5.0);
        assert_eq!(
            sides(false, Vec3::NEG_ONE, Quat::IDENTITY, Vec3::ZERO, eye),
            Vec3::ONE
        );
        assert_eq!(
            sides(true, Vec3::ONE, Quat::IDENTITY, Vec3::ZERO, eye),
            Vec3::NEG_ONE
        );
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
