//! The joint gizmo: a ring to turn a joint about its axis, or a bar to slide
//! it along it, within the joint's limits.
//!
//! **It is one of the gizmos, built the way the transform gizmo is.** A host
//! spawns a [`JointGizmo`] on the body a joint moves and says how that joint
//! moves; the gizmo draws itself in the transform gizmo's units and
//! thicknesses, in front of everything, and the transform gizmo's pointer
//! handling serves both kinds — whichever handle of either is nearest the
//! pointer is the one taken, and [`GizmoPointer`] reports either.
//!
//! **At rest it is a thin line in the joint's colour; in hand it is thick and
//! shows the joint's range**, ported from Studio's materials: the span already
//! travelled solid, the rest of the reachable range half-lit, and on a ring
//! what the joint cannot reach hatched, so a limited range reads as a part of a
//! whole turn.
//!
//! **A ring sits on the joint; a bar sits beside the travel.** A turning joint
//! pivots where it stands, so its ring is centred there. A sliding one carries
//! its body away, so its bar stays put, as long as the joint's travel, laid
//! alongside the axis at a ring's radius from it.
//!
//! The gizmo keeps [`JointGizmo::value`] current while it is dragged, clamped
//! to the limits, and with [`JointGizmo::apply`] poses the body from it: the
//! body's pose when the gizmo first found it is the pose at `rest`, and a value
//! turns or slides it from there. A host may set `value` itself.
//!
//! [`GizmoPointer`]: crate::transform_gizmo::GizmoPointer

use std::f32::consts::TAU;

use bevy::asset::{embedded_asset, embedded_path, AssetPath};
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::mesh::{MeshVertexBufferLayoutRef, VertexAttributeValues};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, CompareFunction, RenderPipelineDescriptor, ShaderType,
    SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::transform_gizmo::{
    meet_plane, plane_along, segment_distance, unit_size, DragPhase, GizmoPart, GizmoPointer,
    GizmoSize, HandleMaterial, HandleUniform, RING_RADIUS, TUBE,
};

/// How a joint moves its body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointMotion {
    /// About its axis, in radians.
    Turn,
    /// Along its axis, in metres.
    Slide,
}

/// A joint gizmo ([module docs](self)).
#[derive(Component, Clone, Debug)]
#[require(Transform, Visibility)]
pub struct JointGizmo {
    /// The body the joint moves.
    pub target: Entity,
    /// The camera it is used through.
    pub camera: Entity,
    pub motion: JointMotion,
    /// The axis the joint moves about or along, in the body's own frame.
    pub axis: Dir3,
    /// The joint's limits, held to when `limited`.
    pub min: f32,
    pub max: f32,
    /// Whether the joint stops at `min` and `max`. The host's to say from what
    /// the joint is, not something limits can tell alone: a revolute joint a
    /// whole turn wide still stops at its ends, and a continuous one turns
    /// without end whatever limits it carries. [`JointGizmo::new`] takes any
    /// finite limits in order as stops.
    pub limited: bool,
    /// Where the joint stands now.
    pub value: f32,
    /// The value at which the body stands in the pose it had when the gizmo
    /// found it.
    pub rest: f32,
    pub colour: Color,
    /// The gizmo's size, as the transform gizmo's is: a ring's radius is the
    /// transform gizmo's ring's at this size. Studio draws its joint controls
    /// at a fifth of a metre.
    pub size: GizmoSize,
    /// Where the ring or bar stands and how it is turned, relative to the
    /// joint: in the joint's rest frame, centred on its pivot with the body's
    /// own axes. Its rotation turns the drawing — spinning a ring's zero round
    /// its axis, say — and leaves the joint itself moving the body about or
    /// along its own axis. [`JointGizmo::default_offset`] is where `new` puts
    /// it.
    pub offset: Transform,
    /// Pose the body from `value`. Without it the gizmo only keeps `value`.
    pub apply: bool,
    /// Draw the gizmo as though it were in hand, without its being so: the
    /// look of a held control, for a still image. It takes nothing from the
    /// pointer, so the view and every other control work as ever.
    pub show_in_hand: bool,
    /// What the host calls this gizmo: the name it goes by in the
    /// [overlay](crate::overlay), where an entity id would mean nothing. Any
    /// string the host chooses — the blackboard key its value is written to,
    /// say — since this crate reads nothing into it.
    pub key: String,
}

impl JointGizmo {
    /// A gizmo on `target`'s joint, seen through `camera`, standing at `rest`,
    /// at Studio's joint-control size.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target: Entity,
        camera: Entity,
        motion: JointMotion,
        axis: Dir3,
        min: f32,
        max: f32,
        rest: f32,
        colour: Color,
    ) -> Self {
        Self {
            target,
            camera,
            motion,
            axis,
            min,
            max,
            limited: min.is_finite() && max.is_finite() && max > min,
            value: rest,
            rest,
            colour,
            size: JOINT_GIZMO_SIZE,
            offset: Self::default_offset(motion, axis),
            apply: true,
            show_in_hand: false,
            key: String::new(),
        }
    }

    /// Where a joint gizmo stands unless told otherwise: a ring on the pivot;
    /// a bar beside the travel, a ring's radius at Studio's size out from the
    /// axis, since a bar laid on the axis would sit inside the housing it
    /// slides through. Which side is fixed by the axis alone, in the body's
    /// frame, so it does not change with how the body happens to be turned.
    pub fn default_offset(motion: JointMotion, axis: Dir3) -> Transform {
        match motion {
            JointMotion::Turn => Transform::IDENTITY,
            JointMotion::Slide => {
                Transform::from_translation(axis.any_orthonormal_vector() * RING_RADIUS * 0.2)
            }
        }
    }

    /// Whether the joint stops at its limits.
    pub fn is_limited(&self) -> bool {
        self.limited
    }

    fn clamp(&self, value: f32) -> f32 {
        if self.is_limited() {
            value.clamp(self.min, self.max)
        } else {
            value
        }
    }

    /// The range as the ring draws it: measured from where the joint can
    /// start, so the shader's travelled span never wraps through zero.
    fn shown_range(&self) -> ShownRange {
        ShownRange::of(self.limited.then_some((self.min, self.max)))
    }
}

/// A joint's range as its ring draws it: measured from one end rather than
/// from the joint's own zero.
///
/// The shader fills from `min` to the current value, so the range has to be
/// **shifted** to start at zero, not wrapped into `[0, 2pi)`. Wrapping a range
/// like `[-pi/2, pi/2]` puts its start at 4.71 and its end at 1.57, which the
/// shader then reads as a range crossing zero — and fills the whole half the
/// joint has already passed.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ShownRange {
    /// The joint value the ring's own zero stands for.
    origin: f32,
    min: f32,
    max: f32,
    /// Whether progress goes round and round, as an unbounded joint's does,
    /// rather than running from one stop to the other.
    wraps: bool,
}

impl ShownRange {
    fn of(limits: Option<(f32, f32)>) -> Self {
        match limits {
            // A ring holds a turn at most, so stops a turn or more apart draw
            // as the whole ring, still read from the lower stop.
            Some((min, max)) => Self {
                origin: min,
                min: 0.0,
                max: (max - min).min(TAU),
                wraps: false,
            },
            // An unbounded joint turns forever, so its ring is a whole turn
            // read from the joint's own zero.
            None => Self {
                origin: 0.0,
                min: 0.0,
                max: TAU,
                wraps: true,
            },
        }
    }

    /// Where a joint value sits on the ring. Between stops it is the way
    /// travelled from the lower one, so the upper stop is the full span and
    /// never wraps back to nothing — stops exactly a turn apart sit at either
    /// end of the ring, not both at its start.
    fn place(&self, value: f32) -> f32 {
        if self.wraps {
            (value - self.origin).rem_euclid(TAU)
        } else {
            (value - self.origin).clamp(0.0, self.max)
        }
    }
}

/// A drag on a joint gizmo's handle, triggered on the gizmo, with the value it
/// has taken the joint to.
#[derive(EntityEvent, Clone, Copy, Debug)]
pub struct JointDrag {
    /// The gizmo.
    pub entity: Entity,
    pub target: Entity,
    pub phase: DragPhase,
    pub value: f32,
}

/// The body's pose when the gizmo found it, which `rest` names, and the body
/// it is the pose of: a gizmo pointed at another body finds that body's pose
/// afresh, rather than posing it from where the last one stood.
#[derive(Component, Clone, Copy)]
pub(crate) struct Rest(Transform, Entity);

/// What a range material needs to shade a joint's span.
#[derive(ShaderType, Debug, Clone, Default)]
pub struct RangeUniform {
    /// The range and where the joint stands in it: radians normalised to
    /// `[0, 2pi)` on a ring, metres from one end on a bar.
    pub min: f32,
    pub max: f32,
    pub current: f32,
    pub _padding: f32,
    pub colour: Vec4,
    pub background: Vec4,
}

macro_rules! range_material {
    ($name:ident, $shader:literal) => {
        #[derive(Asset, TypePath, AsBindGroup, Debug, Clone, Default)]
        pub struct $name {
            #[uniform(0)]
            pub range: RangeUniform,
        }

        impl Material for $name {
            fn fragment_shader() -> ShaderRef {
                ShaderRef::Path(
                    AssetPath::from_path_buf(embedded_path!($shader)).with_source("embedded"),
                )
            }

            fn alpha_mode(&self) -> AlphaMode {
                AlphaMode::Blend
            }

            /// Over everything, as every gizmo handle is: a joint's control
            /// sits inside the link it moves by construction.
            ///
            /// Back faces stay culled, as Bevy has them by default. With no
            /// depth test the tube's far side would draw through its near
            /// side, and the two sets of diagonal stripes would cross into a
            /// hatch.
            fn specialize(
                _pipeline: &MaterialPipeline,
                descriptor: &mut RenderPipelineDescriptor,
                _layout: &MeshVertexBufferLayoutRef,
                _key: MaterialPipelineKey<Self>,
            ) -> Result<(), SpecializedMeshPipelineError> {
                if let Some(depth) = descriptor.depth_stencil.as_mut() {
                    depth.depth_compare = Some(CompareFunction::Always);
                    depth.depth_write_enabled = Some(false);
                }
                Ok(())
            }
        }
    };
}

range_material!(RingRangeMaterial, "joint_ring.wesl");
range_material!(BarRangeMaterial, "joint_bar.wesl");

/// A joint gizmo's two looks: thin at rest, thick and shaded in hand.
#[derive(Component)]
struct JointPart {
    hot: bool,
}

/// The ring or bar's materials, so its range can be kept current.
#[derive(Component)]
enum RangeLook {
    Ring(Handle<RingRangeMaterial>),
    Bar(Handle<BarRangeMaterial>),
}

/// The settings a gizmo's parts were built from; a change rebuilds them.
#[derive(Component, PartialEq)]
struct Built {
    motion: JointMotion,
    limits: (u32, u32),
    limited: bool,
    colour: Color,
}

/// The ground the range shading is mixed toward.
const BACKGROUND: Color = Color::srgb(0.2, 0.2, 0.2);
/// A resting handle's opacity, as the transform gizmo's rings rest.
const REST_OPACITY: f32 = 0.5;

/// A joint gizmo's size unless a host sets one: Studio's joint controls, a
/// fifth of a metre in the world.
pub const JOINT_GIZMO_SIZE: GizmoSize = GizmoSize::World(0.2);

pub struct JointGizmoPlugin;

impl Plugin for JointGizmoPlugin {
    fn build(&self, app: &mut App) {
        // The pointer handling, the pointer state and the resting handle's
        // material are the transform gizmo's, shared so whichever handle of
        // either is nearest is the one taken; a joint gizmo needs them all.
        app.add_plugins(crate::transform_gizmo::TransformGizmoPlugin);
        embedded_asset!(app, "joint_ring.wesl");
        embedded_asset!(app, "joint_bar.wesl");
        app.add_plugins((
            MaterialPlugin::<RingRangeMaterial>::default(),
            MaterialPlugin::<BarRangeMaterial>::default(),
        ))
        .add_systems(
            Update,
            (
                pose_bodies.after(crate::transform_gizmo::TransformGizmoSystems),
                drop_parts,
            ),
        )
        .add_systems(
            PostUpdate,
            (build_parts, place_parts)
                .chain()
                .in_set(crate::transform_gizmo::TransformGizmoSystems)
                .after(TransformSystems::Propagate),
        );
    }
}

/// Where a joint stands in the world: its pivot, its axis, and two directions
/// across the plane a ring turns in, `u` being where the ring's zero points.
#[derive(Clone, Copy, Debug)]
pub(crate) struct JointFrame {
    pub(crate) centre: Vec3,
    pub(crate) axis: Vec3,
    pub(crate) u: Vec3,
    pub(crate) v: Vec3,
}

/// The frame of a joint on a body whose parent stands at `parent` and whose
/// rest pose is `rest`. Measured from the rest pose rather than the body's
/// current one, so the frame does not turn with the joint it describes: a
/// ring's zero stays where it is while the joint turns past it.
pub(crate) fn frame(gizmo: &JointGizmo, parent: &GlobalTransform, rest: &Transform) -> JointFrame {
    let at = parent.mul_transform(*rest);
    let (_, rotation, pivot) = at.to_scale_rotation_translation();
    // The offset in the body's frame, without the body's scale: it is a
    // distance in the world, not a share of the body's size.
    let centre = pivot + rotation * gizmo.offset.translation;
    let turned = rotation * gizmo.offset.rotation;
    let axis = (turned * *gizmo.axis).normalize_or(Vec3::Y);
    // The ring's zero, fixed in the body's frame like the bar's side, so it
    // turns with the offset rather than wherever the world puts it.
    let u = (turned * gizmo.axis.any_orthonormal_vector())
        .reject_from_normalized(axis)
        .normalize_or(axis.any_orthonormal_vector());
    JointFrame {
        centre,
        axis,
        u,
        v: axis.cross(u),
    }
}

/// Where a sliding joint's bar runs, from one end of the travel to the other.
pub(crate) fn bar_ends(gizmo: &JointGizmo, frame: &JointFrame) -> (Vec3, Vec3) {
    (
        frame.centre + frame.axis * (gizmo.min - gizmo.rest),
        frame.centre + frame.axis * (gizmo.max - gizmo.rest),
    )
}

/// What the pointer handling needs of a joint gizmo, gathered once a frame.
#[derive(Clone, Copy)]
pub(crate) struct JointReach {
    pub(crate) frame: JointFrame,
    pub(crate) unit: f32,
}

/// Gathers what hit testing needs, or nothing if the gizmo cannot be used now.
pub(crate) fn reach(
    gizmo: &JointGizmo,
    rest: Option<&Rest>,
    parent: &GlobalTransform,
    camera: (&Camera, &GlobalTransform),
) -> Option<JointReach> {
    let frame = frame(gizmo, parent, &rest?.0);
    let unit = unit_size(gizmo.size, camera, frame.centre)?;
    Some(JointReach { frame, unit })
}

/// How far the pointer is from a joint gizmo's handle on screen, in pixels.
pub(crate) fn distance(
    gizmo: &JointGizmo,
    reach: &JointReach,
    pointer: Vec2,
    (camera, at): (&Camera, &GlobalTransform),
) -> Option<f32> {
    let to_screen = |world: Vec3| camera.world_to_viewport(at, world).ok();
    match gizmo.motion {
        JointMotion::Turn => {
            const SAMPLES: usize = 64;
            let radius = RING_RADIUS * reach.unit;
            let mut nearest = f32::MAX;
            let mut previous = None;
            for i in 0..=SAMPLES {
                let angle = (i % SAMPLES) as f32 * TAU / SAMPLES as f32;
                let point = reach.frame.centre
                    + (reach.frame.u * angle.cos() + reach.frame.v * angle.sin()) * radius;
                let screen = to_screen(point);
                if let (Some(a), Some(b)) = (previous, screen) {
                    nearest = nearest.min(segment_distance(pointer, a, b));
                }
                previous = screen;
            }
            (nearest < f32::MAX).then_some(nearest)
        }
        JointMotion::Slide => {
            let (from, to) = bar_ends(gizmo, &reach.frame);
            Some(segment_distance(pointer, to_screen(from)?, to_screen(to)?))
        }
    }
}

/// What a joint drag keeps from when it began.
#[derive(Clone, Copy)]
pub(crate) struct JointHeld {
    pub(crate) gizmo: Entity,
    frame: JointFrame,
    /// The plane the pointer's ray is met with.
    plane: Vec3,
    /// The joint's value less the pointer's reading of it, when taken, so the
    /// joint does not jump to the pointer on the first frame.
    offset: f32,
}

/// The value the pointer's ray reads off a joint: an angle about its axis, or
/// a distance along it from where it rests.
fn reading(motion: JointMotion, frame: &JointFrame, plane: Vec3, ray: Ray3d) -> Option<f32> {
    let point = meet_plane(ray, plane, frame.centre)?;
    let offset = point - frame.centre;
    Some(match motion {
        JointMotion::Turn => offset.dot(frame.v).atan2(offset.dot(frame.u)),
        JointMotion::Slide => offset.dot(frame.axis),
    })
}

/// Takes hold of a joint gizmo's handle.
pub(crate) fn take(
    entity: Entity,
    gizmo: &JointGizmo,
    reach: &JointReach,
    ray: Ray3d,
) -> Option<JointHeld> {
    let plane = match gizmo.motion {
        JointMotion::Turn => reach.frame.axis,
        JointMotion::Slide => plane_along(ray, reach.frame.axis),
    };
    let read = reading(gizmo.motion, &reach.frame, plane, ray)?;
    let value = match gizmo.motion {
        JointMotion::Turn => gizmo.value,
        JointMotion::Slide => gizmo.value - gizmo.rest,
    };
    Some(JointHeld {
        gizmo: entity,
        frame: reach.frame,
        plane,
        offset: value - read,
    })
}

/// The value a drag has taken the joint to, for the pointer's ray now.
///
/// A turn reads an angle, which jumps by a whole turn as the pointer crosses
/// the ring's zero; the value taken is the one nearest the joint's current
/// value, so the joint turns on smoothly — past a whole turn, if it has no
/// limits.
pub(crate) fn dragged(gizmo: &JointGizmo, held: &JointHeld, ray: Ray3d) -> Option<f32> {
    let read = reading(gizmo.motion, &held.frame, held.plane, ray)? + held.offset;
    let wanted = match gizmo.motion {
        JointMotion::Turn => read + ((gizmo.value - read) / TAU).round() * TAU,
        JointMotion::Slide => gizmo.rest + read,
    };
    Some(gizmo.clamp(wanted))
}

/// Poses each body from its gizmo's value, finding its rest pose the first
/// time.
fn pose_bodies(
    mut commands: Commands,
    gizmos: Query<(Entity, &JointGizmo, Option<&Rest>)>,
    mut bodies: Query<&mut Transform, Without<JointGizmo>>,
) {
    for (entity, gizmo, rest) in &gizmos {
        let Ok(mut body) = bodies.get_mut(gizmo.target) else {
            continue;
        };
        let Some(rest) = rest.filter(|rest| rest.1 == gizmo.target) else {
            commands.entity(entity).insert(Rest(*body, gizmo.target));
            continue;
        };
        if !gizmo.apply {
            continue;
        }
        let moved = gizmo.value - gizmo.rest;
        let posed = match gizmo.motion {
            JointMotion::Turn => Transform {
                rotation: rest.0.rotation * Quat::from_axis_angle(*gizmo.axis, moved),
                ..rest.0
            },
            JointMotion::Slide => Transform {
                translation: rest.0.translation + rest.0.rotation * *gizmo.axis * moved,
                ..rest.0
            },
        };
        if *body != posed {
            *body = posed;
        }
    }
}

/// Takes the ring or bar away from an entity that is no longer a joint
/// gizmo, so no control is left drawn that answers nothing. The entity's
/// other children are its own.
fn drop_parts(
    mut commands: Commands,
    mut removed: RemovedComponents<JointGizmo>,
    children: Query<&Children>,
    parts: Query<(), With<JointPart>>,
) {
    for entity in removed.read() {
        let Ok(mut gizmo) = commands.get_entity(entity) else {
            continue;
        };
        gizmo.remove::<(Built, Rest)>();
        for child in children.iter_descendants(entity) {
            if parts.contains(child) {
                commands.entity(child).despawn();
            }
        }
    }
}

/// Builds the ring or bar of each gizmo that is new or whose joint changed.
fn build_parts(
    mut commands: Commands,
    gizmos: Query<(Entity, &JointGizmo, Option<&Built>, Option<&Children>)>,
    parts: Query<(), With<JointPart>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut plain: ResMut<Assets<HandleMaterial>>,
    mut rings: ResMut<Assets<RingRangeMaterial>>,
    mut bars: ResMut<Assets<BarRangeMaterial>>,
) {
    for (entity, gizmo, built, children) in &gizmos {
        let wanted = Built {
            motion: gizmo.motion,
            limits: (gizmo.min.to_bits(), gizmo.max.to_bits()),
            limited: gizmo.limited,
            colour: gizmo.colour,
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

        let resting = plain.add(HandleMaterial {
            handle: HandleUniform {
                colour: LinearRgba::from(gizmo.colour.with_alpha(REST_OPACITY)).to_vec4(),
            },
        });
        let range = gizmo.shown_range();
        let (max, current) = match gizmo.motion {
            JointMotion::Turn => (range.max, range.place(gizmo.value)),
            JointMotion::Slide => (gizmo.max - gizmo.min, gizmo.value - gizmo.min),
        };
        let uniform = RangeUniform {
            min: 0.0,
            max,
            current,
            colour: LinearRgba::from(gizmo.colour).to_vec4(),
            background: LinearRgba::from(BACKGROUND).to_vec4(),
            ..default()
        };
        // Unit-sized meshes, scaled to the gizmo's unit when placed, so a
        // gizmo sized in pixels keeps them in step with the view.
        let (meshes_by_look, look) = match gizmo.motion {
            JointMotion::Turn => (
                TUBE.map(|tube| {
                    meshes.add(Torus {
                        minor_radius: tube,
                        major_radius: RING_RADIUS,
                    })
                }),
                RangeLook::Ring(rings.add(RingRangeMaterial { range: uniform })),
            ),
            JointMotion::Slide => (
                // A unit long; stretched to the travel when placed.
                TUBE.map(|tube| meshes.add(bar_mesh(tube))),
                RangeLook::Bar(bars.add(BarRangeMaterial { range: uniform })),
            ),
        };
        commands.entity(entity).with_children(|parent| {
            let common = || {
                (
                    GizmoPart,
                    Pickable::IGNORE,
                    NotShadowCaster,
                    NotShadowReceiver,
                    NoFrustumCulling,
                    Transform::default(),
                )
            };
            parent.spawn((
                JointPart { hot: false },
                Mesh3d(meshes_by_look[0].clone()),
                MeshMaterial3d(resting),
                common(),
            ));
            let mut hot = parent.spawn((
                JointPart { hot: true },
                Mesh3d(meshes_by_look[1].clone()),
                Visibility::Hidden,
                common(),
            ));
            match &look {
                RangeLook::Ring(material) => hot.insert(MeshMaterial3d(material.clone())),
                RangeLook::Bar(material) => hot.insert(MeshMaterial3d(material.clone())),
            };
            hot.insert(look);
        });
    }
}

/// Stands each gizmo's ring or bar where its joint is, shows the look for
/// whether it is in hand, and keeps its range shading current.
///
/// Writes world transforms itself, after propagation, for the same reason the
/// transform gizmo does.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn place_parts(
    pointer: Res<GizmoPointer>,
    cameras: Query<(&Camera, &GlobalTransform), (Without<JointPart>, Without<JointGizmo>)>,
    parents: Query<&ChildOf, Without<JointPart>>,
    globals: Query<&GlobalTransform, (Without<JointPart>, Without<JointGizmo>)>,
    mut gizmos: Query<
        (
            Entity,
            &JointGizmo,
            Option<&Rest>,
            &mut Transform,
            &mut GlobalTransform,
            &Children,
        ),
        Without<JointPart>,
    >,
    mut parts: Query<
        (
            &JointPart,
            &mut Transform,
            &mut GlobalTransform,
            &mut Visibility,
            Option<&RangeLook>,
        ),
        Without<JointGizmo>,
    >,
    mut rings: ResMut<Assets<RingRangeMaterial>>,
    mut bars: ResMut<Assets<BarRangeMaterial>>,
) {
    let hot = pointer
        .dragging
        .or(pointer.hovered)
        .map(|(entity, _)| entity);
    for (entity, gizmo, rest, mut transform, mut global, children) in &mut gizmos {
        let parent = parent_global(gizmo.target, &parents, &globals);
        let (Some(rest), Ok(camera)) = (rest, cameras.get(gizmo.camera)) else {
            continue;
        };
        let frame = frame(gizmo, &parent, &rest.0);
        let Some(unit) = unit_size(gizmo.size, camera, frame.centre) else {
            continue;
        };
        *transform = Transform::IDENTITY;
        *global = GlobalTransform::IDENTITY;

        let range = gizmo.shown_range();
        let placed = match gizmo.motion {
            JointMotion::Turn => Transform {
                translation: frame.centre,
                rotation: ring_turn(&frame, range.origin),
                scale: Vec3::splat(unit),
            },
            JointMotion::Slide => {
                let (from, to) = bar_ends(gizmo, &frame);
                Transform {
                    translation: (from + to) / 2.0,
                    rotation: Quat::from_rotation_arc(Vec3::Y, frame.axis),
                    scale: Vec3::new(unit, (to - from).length().max(1e-4), unit),
                }
            }
        };
        let in_hand = hot == Some(entity) || gizmo.show_in_hand;
        for child in children.iter() {
            let Ok((part, mut at, mut part_global, mut visibility, look)) = parts.get_mut(child)
            else {
                continue;
            };
            *at = placed;
            *part_global = GlobalTransform::from(placed);
            let shown = part.hot == in_hand;
            let wanted = if shown {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
            if *visibility != wanted {
                *visibility = wanted;
            }
            let current = match gizmo.motion {
                JointMotion::Turn => range.place(gizmo.value),
                JointMotion::Slide => gizmo.value - gizmo.min,
            };
            // Read before writing: taking an asset mutably marks it changed,
            // which re-uploads it every frame whether or not anything moved.
            match look {
                Some(RangeLook::Ring(material))
                    if rings
                        .get(material)
                        .is_some_and(|m| m.range.current != current) =>
                {
                    if let Some(mut material) = rings.get_mut(material) {
                        material.range.current = current;
                    }
                }
                Some(RangeLook::Bar(material))
                    if bars
                        .get(material)
                        .is_some_and(|m| m.range.current != current) =>
                {
                    if let Some(mut material) = bars.get_mut(material) {
                        material.range.current = current;
                    }
                }
                _ => {}
            }
        }
    }
}

/// The world transform of `target`'s parent, or the world's origin for a
/// target with none.
fn parent_global(
    target: Entity,
    parents: &Query<&ChildOf, Without<JointPart>>,
    globals: &Query<&GlobalTransform, (Without<JointPart>, Without<JointGizmo>)>,
) -> GlobalTransform {
    parents
        .get(target)
        .ok()
        .and_then(|child_of| globals.get(child_of.parent()).ok())
        .copied()
        .unwrap_or(GlobalTransform::IDENTITY)
}

/// A bar a unit long along Y, whose `uv.y` everywhere says how far along the
/// travel a point is, as `joint_bar.wesl` reads it.
///
/// A cylinder's sides already run `uv.y` from nought at its −Y end to one at
/// its +Y end, but its end caps carry coordinates of their own, laid out as a
/// disc, which the shader would read as somewhere along the travel — so a cap
/// would show the travelled colour wherever its disc falls below the joint's
/// progress. Each cap is given the coordinate of the end it closes.
fn bar_mesh(radius: f32) -> Mesh {
    let mut mesh = Mesh::from(Cylinder::new(radius, 1.0));
    let normals: Vec<[f32; 3]> = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
        Some(VertexAttributeValues::Float32x3(normals)) => normals.clone(),
        _ => return mesh,
    };
    if let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute_mut(Mesh::ATTRIBUTE_UV_0) {
        for (uv, normal) in uvs.iter_mut().zip(&normals) {
            if normal[1] > 0.5 {
                uv[1] = 1.0;
            } else if normal[1] < -0.5 {
                uv[1] = 0.0;
            }
        }
    }
    mesh
}

/// Turns a torus built about +Y so its own angle runs the way the frame
/// measures angles, its zero at `origin`.
///
/// The mesh's up is the axis **negated**, because a torus winds the opposite
/// way about its own axis from the way the frame measures — without that the
/// ring fills against the drag. A torus is symmetric about its plane, so
/// flipping its up changes the winding and nothing about how it looks.
fn ring_turn(frame: &JointFrame, origin: f32) -> Quat {
    let start = frame.u * origin.cos() + frame.v * origin.sin();
    Quat::from_mat3(&Mat3::from_cols(
        start,
        -frame.axis,
        frame.axis.cross(start),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joint(motion: JointMotion, min: f32, max: f32) -> JointGizmo {
        JointGizmo::new(
            Entity::PLACEHOLDER,
            Entity::PLACEHOLDER,
            motion,
            Dir3::Y,
            min,
            max,
            0.0,
            Color::WHITE,
        )
    }

    /// A joint that stops holds its ends however wide they are, and one that
    /// does not passes them, whatever limits it carries.
    #[test]
    fn whether_a_joint_stops_is_its_own() {
        use std::f32::consts::PI;
        let mut stopping = joint(JointMotion::Turn, -PI, PI);
        assert!(stopping.is_limited(), "a whole turn wide, and still stops");
        assert_eq!(stopping.clamp(4.0), PI);
        stopping.limited = false;
        assert_eq!(stopping.clamp(4.0), 4.0);
    }

    /// Stops a whole turn apart sit at either end of the ring, with the way
    /// between them travelled in order; an unbounded joint goes round.
    #[test]
    fn stops_a_turn_apart_run_end_to_end() {
        use std::f32::consts::PI;
        let range = ShownRange::of(Some((-PI, PI)));
        assert_eq!(range.max, TAU);
        assert!(range.place(-PI).abs() < 1e-6, "the lower stop at the start");
        assert!((range.place(0.0) - PI).abs() < 1e-6, "halfway at halfway");
        assert!(
            (range.place(PI) - TAU).abs() < 1e-6,
            "the upper stop at the end"
        );
        let round = ShownRange::of(None);
        assert!(
            round.place(TAU + 1.0) < TAU,
            "an unbounded joint goes round"
        );
    }

    /// Taking the gizmo away takes its ring with it and leaves the entity's
    /// other children.
    #[test]
    fn removing_the_gizmo_removes_its_parts() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_systems(Update, drop_parts);
        let gizmo = app
            .world_mut()
            .spawn(joint(JointMotion::Turn, -1.0, 1.0))
            .id();
        let part = app
            .world_mut()
            .spawn((JointPart { hot: false }, ChildOf(gizmo)))
            .id();
        let other = app.world_mut().spawn(ChildOf(gizmo)).id();
        app.update();
        app.world_mut().entity_mut(gizmo).remove::<JointGizmo>();
        app.update();
        assert!(app.world().get_entity(part).is_err(), "the ring is gone");
        assert!(app.world().get_entity(other).is_ok(), "other children stay");
    }

    /// A gizmo pointed at another body poses that body from its own rest, not
    /// from where the first one stood.
    #[test]
    fn a_retargeted_gizmo_poses_its_new_body_from_its_own_rest() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_systems(Update, pose_bodies);
        let first = app
            .world_mut()
            .spawn(Transform::from_xyz(1.0, 0.0, 0.0))
            .id();
        let second = app
            .world_mut()
            .spawn(Transform::from_xyz(0.0, 5.0, 0.0))
            .id();
        let mut gizmo = joint(JointMotion::Slide, -1.0, 1.0);
        gizmo.target = first;
        gizmo.axis = Dir3::X;
        gizmo.value = 0.5;
        let gizmo = app.world_mut().spawn(gizmo).id();
        app.update();
        app.update();
        let at = |app: &App, body: Entity| app.world().get::<Transform>(body).unwrap().translation;
        assert!(at(&app, first).abs_diff_eq(Vec3::new(1.5, 0.0, 0.0), 1e-5));

        app.world_mut().get_mut::<JointGizmo>(gizmo).unwrap().target = second;
        app.update();
        app.update();
        assert!(
            at(&app, second).abs_diff_eq(Vec3::new(0.5, 5.0, 0.0), 1e-5),
            "{:?}",
            at(&app, second)
        );
    }

    /// A range shifts to start at zero rather than wrapping, so a range across
    /// the joint's own zero still fills from its start.
    #[test]
    fn a_range_is_shifted_not_wrapped() {
        let range = ShownRange::of(Some((-1.0, 1.0)));
        assert_eq!((range.min, range.max), (0.0, 2.0));
        assert!((range.place(0.0) - 1.0).abs() < 1e-6);
        assert_eq!(ShownRange::of(None).max, TAU);
    }

    fn frame_about_y() -> JointFrame {
        JointFrame {
            centre: Vec3::ZERO,
            axis: Vec3::Y,
            u: Vec3::X,
            v: Vec3::Y.cross(Vec3::X),
        }
    }

    fn ray_down_at(point: Vec3) -> Ray3d {
        Ray3d::new(point + Vec3::Y * 10.0, Dir3::NEG_Y)
    }

    /// A turn follows the pointer's angle from where it was taken, and holds
    /// at a limit.
    #[test]
    fn a_turn_follows_the_pointer_and_holds_at_its_limit() {
        let mut gizmo = joint(JointMotion::Turn, -1.0, 1.0);
        let reach = JointReach {
            frame: frame_about_y(),
            unit: 1.0,
        };
        let held = take(Entity::PLACEHOLDER, &gizmo, &reach, ray_down_at(Vec3::X)).unwrap();
        // v is Y × X = −Z, so a point toward −Z is a quarter turn on.
        let quarter = dragged(&gizmo, &held, ray_down_at(Vec3::NEG_Z)).unwrap();
        assert!(
            (quarter - 1.0).abs() < 1e-5,
            "a quarter turn holds at the limit of 1"
        );
        gizmo.value = 0.5;
        let back = dragged(&gizmo, &held, ray_down_at(Vec3::new(1.0, 0.0, -0.2))).unwrap();
        assert!((back - 0.2f32.atan2(1.0)).abs() < 1e-4);
    }

    /// Every point of a bar's cap says the end of the travel it closes, and
    /// the sides run from nought at −Y to one at +Y.
    #[test]
    fn a_bar_reads_its_travel_from_end_to_end() {
        let mesh = bar_mesh(0.1);
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("a bar has positions");
        };
        let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
        else {
            panic!("a bar has coordinates");
        };
        for (position, uv) in positions.iter().zip(uvs) {
            let expected = position[1] + 0.5;
            assert!((uv[1] - expected).abs() < 1e-5, "{position:?} reads {uv:?}");
        }
    }

    /// An endless turn keeps going past a whole turn rather than jumping back
    /// as the pointer crosses the ring's zero.
    #[test]
    fn an_endless_turn_carries_on_past_a_whole_turn() {
        let mut gizmo = joint(JointMotion::Turn, f32::NEG_INFINITY, f32::INFINITY);
        let reach = JointReach {
            frame: frame_about_y(),
            unit: 1.0,
        };
        let held = take(Entity::PLACEHOLDER, &gizmo, &reach, ray_down_at(Vec3::X)).unwrap();
        gizmo.value = TAU - 0.1;
        let past = dragged(&gizmo, &held, ray_down_at(Vec3::new(1.0, 0.0, -0.1))).unwrap();
        assert!((past - (TAU + 0.1f32.atan2(1.0))).abs() < 1e-4, "{past}");
    }

    /// The offset places and turns the drawing in the joint's rest frame: a
    /// shift moves the centre along the body's own axes, and a turn about the
    /// joint's axis spins the ring's zero round it without tilting the axis.
    #[test]
    fn an_offset_moves_and_spins_the_drawing() {
        let mut gizmo = joint(JointMotion::Turn, -1.0, 1.0);
        let body = Transform::from_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2));
        let plain = frame(&gizmo, &GlobalTransform::IDENTITY, &body);
        gizmo.offset = Transform::from_xyz(1.0, 0.0, 0.0)
            .with_rotation(Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2));
        let moved = frame(&gizmo, &GlobalTransform::IDENTITY, &body);
        // The body's X is the world's Y after its quarter turn about Z.
        assert!(
            moved.centre.abs_diff_eq(Vec3::Y, 1e-5),
            "{:?}",
            moved.centre
        );
        assert!(moved.axis.abs_diff_eq(plain.axis, 1e-5));
        assert!(
            moved.u.dot(plain.u).abs() < 1e-5,
            "the zero turned a quarter"
        );
    }

    /// A slide's bar stands beside its axis by default, and its offset moves
    /// it.
    #[test]
    fn a_bar_stands_beside_its_axis_unless_moved() {
        let mut gizmo = joint(JointMotion::Slide, -0.1, 0.1);
        let beside = frame(&gizmo, &GlobalTransform::IDENTITY, &Transform::IDENTITY).centre;
        assert!((beside.length() - RING_RADIUS * 0.2).abs() < 1e-5);
        assert!(beside.dot(*gizmo.axis).abs() < 1e-5);
        gizmo.offset = Transform::IDENTITY;
        let on = frame(&gizmo, &GlobalTransform::IDENTITY, &Transform::IDENTITY).centre;
        assert!(on.abs_diff_eq(Vec3::ZERO, 1e-6));
    }

    /// A slide follows the pointer along its axis and holds at its ends.
    #[test]
    fn a_slide_follows_its_axis_and_holds_at_its_ends() {
        let gizmo = joint(JointMotion::Slide, -0.1, 0.1);
        let reach = JointReach {
            frame: frame_about_y(),
            unit: 1.0,
        };
        let ray = |y: f32| Ray3d::new(Vec3::new(0.0, y, 10.0), Dir3::NEG_Z);
        let held = take(Entity::PLACEHOLDER, &gizmo, &reach, ray(0.0)).unwrap();
        assert!((dragged(&gizmo, &held, ray(0.05)).unwrap() - 0.05).abs() < 1e-5);
        assert!((dragged(&gizmo, &held, ray(0.5)).unwrap() - 0.1).abs() < 1e-5);
    }
}
