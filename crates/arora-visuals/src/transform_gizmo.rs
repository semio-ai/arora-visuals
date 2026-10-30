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
//! to turn, and a ball beyond each arrow to scale. A host that wants one mode
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
//! **Scale is always along the target's own axes**, whichever space the other
//! handles are in: a world axis is not one an object can be stretched along
//! once it is turned.
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
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, CompareFunction, RenderPipelineDescriptor, ShaderType,
    SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

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

/// How big a gizmo is: the reach of its arrows.
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
    /// Where the gizmo stands from the target's origin, along the gizmo's own
    /// axes. Turning a target about an offset gizmo swings it round that
    /// point.
    pub offset: Vec3,
    /// The directions the gizmo calls X, Y and Z, in the space it works in:
    /// Bevy's own by default, or [`TransformGizmo::Z_UP`] for a Z-up world.
    /// They must be a right-handed set.
    pub axes: [Dir3; 3],
    /// The X, Y and Z handles' colours.
    pub colours: [Color; 3],
    /// Write each drag onto the target. Without it the gizmo only reports.
    pub apply: bool,
}

impl TransformGizmo {
    /// A gizmo on `target`, seen through `camera`, with every mode, in world
    /// space, a hundred pixels across, in Studio's axis colours.
    pub fn new(target: Entity, camera: Entity) -> Self {
        Self {
            target,
            camera,
            modes: TransformModes::ALL,
            only: None,
            space: GizmoSpace::World,
            size: GizmoSize::Pixels(100.0),
            offset: Vec3::ZERO,
            axes: [Dir3::X, Dir3::Y, Dir3::Z],
            colours: [
                Color::srgb_u8(0xef, 0x44, 0x44),
                Color::srgb_u8(0x22, 0xc5, 0x5e),
                Color::srgb_u8(0x3b, 0x82, 0xf6),
            ],
            apply: true,
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

/// One handle of a gizmo, by what it does and about which axis. A plane's axis
/// is the one it is square to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GizmoHandle {
    Arrow(usize),
    Plane(usize),
    Ring(usize),
    Ball(usize),
}

impl GizmoHandle {
    fn mode(self) -> TransformMode {
        match self {
            GizmoHandle::Arrow(_) | GizmoHandle::Plane(_) => TransformMode::Translate,
            GizmoHandle::Ring(_) => TransformMode::Rotate,
            GizmoHandle::Ball(_) => TransformMode::Scale,
        }
    }

    fn axis(self) -> usize {
        match self {
            GizmoHandle::Arrow(axis)
            | GizmoHandle::Plane(axis)
            | GizmoHandle::Ring(axis)
            | GizmoHandle::Ball(axis) => axis,
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
const SHAFT_LENGTH: f32 = 0.8;
const CONE_LENGTH: f32 = 0.2;
const CONE_RADIUS: f32 = 0.05;
const RING_RADIUS: f32 = 0.65;
const PLANE_OFFSET: f32 = 0.243;
const PLANE_SIDE: f32 = 0.225;
const BALL_REACH: f32 = 1.2;
const BALL_RADIUS: f32 = 0.044;
/// An arrow is grabbed along this stretch of it, short of the centre, where
/// the planes and the other arrows crowd.
const ARROW_FROM: f32 = 0.25;
/// How far from a handle on screen the pointer may be and still take it.
const REACH_PX: f32 = 8.0;
/// Below this share of facing the camera a plane is too nearly edge-on to
/// drag in.
const EDGE_ON: f32 = 0.15;

/// Rest and in-hand thicknesses, as the joint controls swap thin for thick.
const SHAFT_RADIUS: [f32; 2] = [0.008, 0.016];
const RING_TUBE: [f32; 2] = [0.012, 0.04];
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
    /// Where the piece sits and how it is turned, in gizmo units along +X,
    /// before the handle's own axis is applied.
    at: Transform,
    meshes: [Handle<Mesh>; 2],
    materials: [Handle<HandleMaterial>; 2],
    /// Which look it shows now.
    hot: bool,
}

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
    /// The target's own axes when the drag began, for scaling along.
    local_axes: Quat,
    /// The gizmo's naming of the axes, so a scale handle stretches the
    /// target's own component that its label names.
    labels: Quat,
    /// Where the pointer's ray met the drag's plane when it began.
    grabbed: Vec3,
    /// The plane the drag works in.
    plane: Vec3,
    start_global: GlobalTransform,
    start_local: Transform,
}

/// The gizmo's pointer handling, in `Update`. A host orders anything that
/// shares the pointer — a camera controller above all — after it, so that
/// [`GizmoPointer`] is this frame's answer when it reads it.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct TransformGizmoSystems;

pub struct TransformGizmoPlugin;

impl Plugin for TransformGizmoPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "handle.wesl");
        app.add_plugins(MaterialPlugin::<HandleMaterial>::default())
            .init_resource::<GizmoPointer>()
            .add_systems(Update, drive.in_set(TransformGizmoSystems))
            .add_systems(
                PostUpdate,
                (build_parts, place_parts)
                    .chain()
                    .after(TransformSystems::Propagate),
            );
    }
}

/// A part to spawn: its handle, where it sits, and its rest and in-hand
/// meshes and materials.
type Piece = (
    GizmoHandle,
    Transform,
    [Handle<Mesh>; 2],
    [Handle<HandleMaterial>; 2],
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
        commands.entity(entity).insert(wanted);

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
            if !gizmo.modes.has(handle.mode()) {
                continue;
            }
            let colour = gizmo.colours[handle.axis()];
            match handle {
                GizmoHandle::Arrow(_) => {
                    let look = material(colour, OPACITY);
                    pieces.push((
                        handle,
                        along_x(SHAFT_LENGTH / 2.0),
                        SHAFT_RADIUS.map(|radius| meshes.add(Cylinder::new(radius, SHAFT_LENGTH))),
                        look.clone(),
                    ));
                    pieces.push((
                        handle,
                        along_x(SHAFT_LENGTH + CONE_LENGTH / 2.0),
                        SWELL.map(|swell| {
                            meshes.add(Cone {
                                radius: CONE_RADIUS * swell,
                                height: CONE_LENGTH,
                            })
                        }),
                        look,
                    ));
                }
                GizmoHandle::Plane(_) => {
                    let side = PLANE_SIDE / 2.0;
                    // A square in the plane square to X, out along the other
                    // two axes from the centre.
                    let at = Transform::from_xyz(0.0, PLANE_OFFSET, PLANE_OFFSET)
                        .with_rotation(Quat::from_rotation_arc(Vec3::Y, Vec3::X));
                    pieces.push((
                        handle,
                        at,
                        SWELL.map(|swell| {
                            meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(side * swell)))
                        }),
                        material(colour, PLANE_OPACITY),
                    ));
                }
                GizmoHandle::Ring(_) => {
                    let at = Transform::from_rotation(Quat::from_rotation_arc(Vec3::Y, Vec3::X));
                    pieces.push((
                        handle,
                        at,
                        RING_TUBE.map(|tube| {
                            meshes.add(Torus {
                                minor_radius: tube,
                                major_radius: RING_RADIUS,
                            })
                        }),
                        material(colour, RING_OPACITY),
                    ));
                }
                GizmoHandle::Ball(_) => {
                    pieces.push((
                        handle,
                        Transform::from_xyz(BALL_REACH, 0.0, 0.0),
                        SWELL.map(|swell| meshes.add(Sphere::new(BALL_RADIUS * swell))),
                        material(colour, OPACITY),
                    ));
                }
            }
        }
        commands.entity(entity).with_children(|gizmo| {
            for (handle, at, meshes, materials) in pieces {
                gizmo.spawn((
                    Part {
                        handle,
                        at,
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

fn along_x(distance: f32) -> Transform {
    Transform::from_xyz(distance, 0.0, 0.0).with_rotation(Quat::from_rotation_arc(Vec3::Y, Vec3::X))
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

/// Where a gizmo stands, how its axes are turned, its target's own axes, and
/// how many metres one gizmo unit is.
#[derive(Clone, Copy)]
struct Placement {
    origin: Vec3,
    axes: Quat,
    local_axes: Quat,
    unit: f32,
}

fn placement(
    gizmo: &TransformGizmo,
    target: &GlobalTransform,
    camera: (&Camera, &GlobalTransform),
) -> Option<Placement> {
    let (_, turned, position) = target.to_scale_rotation_translation();
    let labels = gizmo.labels();
    let local_axes = turned * labels;
    let axes = match gizmo.space {
        GizmoSpace::World => labels,
        GizmoSpace::Local => local_axes,
    };
    let origin = position + axes * gizmo.offset;
    let unit = match gizmo.size {
        GizmoSize::World(metres) => metres,
        GizmoSize::Pixels(pixels) => pixels * metres_per_pixel(camera, origin)?,
    };
    Some(Placement {
        origin,
        axes,
        local_axes,
        unit,
    })
}

/// How many metres one logical pixel spans at `point`, from two rays a pixel
/// apart met with the plane through `point` facing the camera. Rays rather
/// than a field of view, so orthographic cameras are measured the same way.
fn metres_per_pixel((camera, at): (&Camera, &GlobalTransform), point: Vec3) -> Option<f32> {
    let centre = camera.world_to_viewport(at, point).ok()?;
    let facing = at.forward().as_vec3();
    let meet = |pixel: Vec2| {
        let ray = camera.viewport_to_world(at, pixel).ok()?;
        meet_plane(ray, facing, point)
    };
    Some(meet(centre)?.distance(meet(centre + Vec2::X)?))
}

/// Where a ray meets the plane through `origin` square to `normal`.
fn meet_plane(ray: Ray3d, normal: Vec3, origin: Vec3) -> Option<Vec3> {
    let along = ray.direction.dot(normal);
    if along.abs() < f32::EPSILON {
        return None;
    }
    let distance = (origin - ray.origin).dot(normal) / along;
    Some(ray.origin + *ray.direction * distance)
}

/// A plane containing `axis` and turned as far toward the viewer as it can
/// be: dragging along an axis is meeting the pointer's ray with this.
fn plane_along(ray: Ray3d, axis: Vec3) -> Vec3 {
    let across = ray.direction.cross(axis);
    if across.length_squared() < f32::EPSILON {
        return axis.any_orthonormal_vector();
    }
    axis.cross(across.normalize()).normalize()
}

fn segment_distance(point: Vec2, a: Vec2, b: Vec2) -> f32 {
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
    let axes = match handle {
        GizmoHandle::Ball(_) => place.local_axes,
        _ => place.axes,
    };
    let axis = axes * axis_vector(handle.axis());
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
        GizmoHandle::Plane(index) => {
            if ray.direction.dot(axis).abs() < EDGE_ON {
                return None;
            }
            let hit = meet_plane(ray, axis, place.origin)?;
            let u = place.axes * axis_vector((index + 1) % 3);
            let v = place.axes * axis_vector((index + 2) % 3);
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

/// Finds the handle under the pointer, and takes, drags and lets go of
/// handles with the left button.
#[allow(clippy::too_many_arguments)]
fn drive(
    mut commands: Commands,
    gizmos: Query<(Entity, &TransformGizmo, &InheritedVisibility)>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    globals: Query<&GlobalTransform>,
    mut locals: Query<&mut Transform>,
    parents: Query<&ChildOf>,
    windows: Query<&Window>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut pointer: ResMut<GizmoPointer>,
    mut held: Local<Option<Drag>>,
) {
    let cursor = windows.iter().find_map(Window::cursor_position);

    if let Some(drag) = *held {
        let Ok((_, gizmo, _)) = gizmos.get(drag.gizmo) else {
            *held = None;
            pointer.dragging = None;
            return;
        };
        let released = !buttons.pressed(MouseButton::Left);
        let ray = cursor
            .zip(cameras.get(gizmo.camera).ok())
            .and_then(|(cursor, (camera, at))| camera.viewport_to_world(at, cursor).ok());
        let moved = ray.and_then(|ray| dragged_to(&drag, ray));
        if let Some(moved) = moved {
            let local = match moved {
                Moved::World(global) => {
                    let parent = parents
                        .get(gizmo.target)
                        .ok()
                        .and_then(|child_of| globals.get(child_of.parent()).ok());
                    // Moving and turning leave the scale as it was; taking it
                    // back out of the world would only add rounding to it.
                    Transform {
                        scale: drag.start_local.scale,
                        ..match parent {
                            Some(parent) => global.reparented_to(parent),
                            None => global.compute_transform(),
                        }
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
        } else if released {
            commands.trigger(GizmoDrag {
                entity: drag.gizmo,
                target: gizmo.target,
                handle: drag.handle,
                phase: DragPhase::End,
                transform: locals.get(gizmo.target).copied().unwrap_or_default(),
            });
        }
        if released {
            *held = None;
            pointer.dragging = None;
        }
        return;
    }

    // Nothing held: find what is under the pointer.
    pointer.hovered = None;
    let Some(cursor) = cursor else {
        return;
    };
    let mut nearest: Option<(f32, Entity, GizmoHandle, Placement, Ray3d)> = None;
    for (entity, gizmo, visible) in &gizmos {
        if !visible.get() {
            continue;
        }
        let (Ok(camera), Ok(target)) = (cameras.get(gizmo.camera), globals.get(gizmo.target))
        else {
            continue;
        };
        let Ok(ray) = camera.0.viewport_to_world(camera.1, cursor) else {
            continue;
        };
        let Some(place) = placement(gizmo, target, camera) else {
            continue;
        };
        for handle in GizmoHandle::ALL {
            if !gizmo.shows(handle.mode()) {
                continue;
            }
            let Some(distance) = distance_to(handle, &place, cursor, ray, camera) else {
                continue;
            };
            if distance <= REACH_PX && nearest.as_ref().is_none_or(|(best, ..)| distance < *best) {
                nearest = Some((distance, entity, handle, place, ray));
            }
        }
    }
    let Some((_, entity, handle, place, ray)) = nearest else {
        return;
    };
    pointer.hovered = Some((entity, handle));

    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let Ok((_, gizmo, _)) = gizmos.get(entity) else {
        return;
    };
    let (Ok(start_global), Ok(start_local)) = (globals.get(gizmo.target), locals.get(gizmo.target))
    else {
        return;
    };
    let axes = match handle {
        GizmoHandle::Ball(_) => place.local_axes,
        _ => place.axes,
    };
    let axis = axes * axis_vector(handle.axis());
    let plane = match handle {
        GizmoHandle::Arrow(_) | GizmoHandle::Ball(_) => plane_along(ray, axis),
        GizmoHandle::Plane(_) | GizmoHandle::Ring(_) => axis,
    };
    let Some(grabbed) = meet_plane(ray, plane, place.origin) else {
        return;
    };
    let drag = Drag {
        gizmo: entity,
        handle,
        origin: place.origin,
        axes: place.axes,
        local_axes: place.local_axes,
        labels: gizmo.labels(),
        grabbed,
        plane,
        start_global: *start_global,
        start_local: *start_local,
    };
    *held = Some(drag);
    pointer.dragging = Some((entity, handle));
    commands.trigger(GizmoDrag {
        entity,
        target: gizmo.target,
        handle,
        phase: DragPhase::Start,
        transform: *start_local,
    });
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
        GizmoHandle::Ball(index) => {
            let axis = axis(drag.local_axes, index);
            let from = (drag.grabbed - drag.origin).dot(axis);
            let to = (point - drag.origin).dot(axis);
            if from.abs() < f32::EPSILON {
                return None;
            }
            // The target's own component this handle stretches: the one its
            // label names, which under a relabelling need not be `index`.
            let component = (drag.labels * axis_vector(index)).abs();
            let stretch = Vec3::ONE + component * ((to / from).max(0.01) - 1.0);
            return Some(Moved::Stretch(stretch));
        }
    };
    Some(Moved::World(GlobalTransform::from(moved)))
}

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
fn place_parts(
    pointer: Res<GizmoPointer>,
    cameras: Query<(&Camera, &GlobalTransform), (Without<Part>, Without<TransformGizmo>)>,
    targets: Query<&GlobalTransform, (Without<Part>, Without<TransformGizmo>)>,
    mut gizmos: Query<
        (
            Entity,
            &TransformGizmo,
            &mut Transform,
            &mut GlobalTransform,
            &Children,
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
    for (entity, gizmo, mut transform, mut global, children) in &mut gizmos {
        let place = cameras
            .get(gizmo.camera)
            .ok()
            .zip(targets.get(gizmo.target).ok())
            .and_then(|(camera, target)| placement(gizmo, target, camera));
        let Some(place) = place else {
            continue;
        };
        *transform = Transform::from_translation(place.origin).with_scale(Vec3::splat(place.unit));
        *global = GlobalTransform::from(*transform);
        for child in children.iter() {
            let Ok((mut part, mut at, mut part_global, mut visibility, mut mesh, mut material)) =
                parts.get_mut(child)
            else {
                continue;
            };
            let axes = match part.handle {
                GizmoHandle::Ball(_) => place.local_axes,
                _ => place.axes,
            };
            let turn = axes * onto_axis(part.handle.axis());
            *at = Transform {
                translation: turn * part.at.translation,
                rotation: turn * part.at.rotation,
                scale: part.at.scale,
            };
            *part_global = global.mul_transform(*at);
            *visibility = if gizmo.shows(part.handle.mode()) {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            local_axes: Quat::IDENTITY,
            labels: Quat::IDENTITY,
            grabbed,
            plane,
            start_global: GlobalTransform::IDENTITY,
            start_local: Transform::IDENTITY,
        }
    }

    fn world(moved: Option<Moved>) -> GlobalTransform {
        match moved {
            Some(Moved::World(global)) => global,
            other => panic!("expected a move in the world, got {other:?}"),
        }
    }

    /// Under Z-up labels the Z arrow points along Bevy's up and the Y arrow
    /// along Bevy's −Z, and the Z ball stretches the target's own Y.
    #[test]
    fn z_up_labels_point_the_handles_their_way() {
        let gizmo = TransformGizmo {
            axes: TransformGizmo::Z_UP,
            ..TransformGizmo::new(Entity::PLACEHOLDER, Entity::PLACEHOLDER)
        };
        let labels = gizmo.labels();
        assert!((labels * axis_vector(2)).abs_diff_eq(Vec3::Y, 1e-6));
        assert!((labels * axis_vector(1)).abs_diff_eq(Vec3::NEG_Z, 1e-6));

        let d = Drag {
            axes: labels,
            local_axes: labels,
            labels,
            ..drag(GizmoHandle::Ball(2), Vec3::new(0.0, 1.0, 0.0), Vec3::Z)
        };
        let ray = Ray3d::new(Vec3::new(0.3, 3.0, 10.0), Dir3::NEG_Z);
        let Some(Moved::Stretch(stretch)) = dragged_to(&d, ray) else {
            panic!("a ball stretches");
        };
        assert!(stretch.abs_diff_eq(Vec3::new(1.0, 3.0, 1.0), 1e-5));
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
