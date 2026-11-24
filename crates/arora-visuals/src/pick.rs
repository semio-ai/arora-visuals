//! Picks: a pointer press on an instance, resolved to the element under it or
//! to a miss.
//!
//! **The cast is over the mesh as drawn, not as loaded.** Each element's
//! vertices have its morph targets applied at their current weights, in world
//! space, which is what three.js casts over. Bevy's own mesh picking casts the
//! base vertices, so it misses an element a morph has moved — an opened mouth
//! — and lands on whatever is behind it. Any asset whose shape is driven at
//! runtime needs the morph-aware cast, which is why this is shared rather than
//! one renderer's workaround.
//!
//! Bevy's picking is used only to deliver the press: with no mesh backend
//! registered, every press on the scene reaches the window entity and is
//! resolved here. **A press that lands anywhere else is not the scene's.** A UI
//! backend delivers a press on a panel to the panel's node, and resolving that
//! one too would select whatever element happens to sit behind the panel.
//!
//! The pointer goes to the instances whose rectangle holds it, the one drawn
//! last first, and the nearest element its ray meets is the one under it.
//! Only an instance whose camera draws to the window the pointer is in can be
//! under it: a camera filling a screen's texture is a picture on a mesh, not
//! a view the pointer is over.
//!
//! **A pick is a [`Picked`] message.** Two consumers sharing an app each read
//! every pick with a reader of their own, and tell theirs by
//! [`Picked::instance`], so neither takes the other's.
//!
//! **A pick is a click, decided as Studio's scene decides one.** The press
//! notes every element under the pointer; the release picks the element under
//! it then, if that element was among them — however far the pointer moved
//! in between, so a press and release on one element picks it. A release over
//! no element is a miss of the topmost instance, but only if the pointer
//! moved [`MISS_SLOP`] pixels or less: a drag that turns the view and ends
//! over empty space is not a request to clear the selection. A release over a
//! different element than any the press found picks nothing. A press or
//! release outside every drawn rectangle belongs to no instance.
//!
//! **A press and its release in one frame are a click too.** Bevy sends a
//! press to what the pointer is over this frame but a release to what it was
//! over the frame before, so a pointer that reached the window in the frame
//! it pressed and let go — a quick tap on a slow frame, a touch that begins
//! and ends between frames — releases onto nothing. Such a pointer's frame
//! is replayed in input order, each release deciding with the press before
//! it as the window's release would have, where the window is what the
//! pointer is over; a release over anything else lets its press go.

use bevy::camera::primitives::Aabb;
use bevy::ecs::system::SystemParam;
use bevy::math::Affine3A;
use bevy::mesh::morph::{MeshMorphWeights, MorphAttributes, MorphWeights};
use bevy::mesh::{Indices, PrimitiveTopology};
use std::collections::{HashMap, HashSet};

use bevy::picking::events::{PointerPress, PointerRelease};
use bevy::picking::hover::{HoverMap, PreviousHoverMap};
use bevy::picking::mesh_picking::ray_cast::{ray_mesh_intersection, Backfaces};
use bevy::picking::pointer::{Location, PointerAction, PointerInput};
use bevy::picking::pointer::{PointerButton, PointerId};
use bevy::picking::PickingSystems;
use bevy::prelude::*;

use crate::binding::ElementTargets;
use crate::instance::{EntityInstance, Superseded};
use crate::transform_gizmo::{draws_to, GizmoPointer};
use bevy::camera::{NormalizedRenderTarget, RenderTarget};
use bevy::window::PrimaryWindow;

/// A click on an instance: the element under it, by the ids the asset's own
/// `RobotData` declares, or `None` for a click that hit no element.
///
/// Like any Bevy message it is kept for two frames, so a consumer reads picks
/// every frame, from a system, and keeps any it needs longer itself.
#[derive(Message, Debug, Clone, PartialEq, Eq)]
pub struct Picked {
    /// The instance entity, which a consumer tells its own picks by.
    pub instance: Entity,
    /// What the host calls the instance ([`EntityInstance::id`]).
    pub instance_id: String,
    pub element_id: Option<String>,
}

/// How far, in logical pixels, the pointer may move between press and release
/// and a release over nothing still count as a miss.
pub const MISS_SLOP: f32 = 2.0;

/// What each pointer's primary press found, kept until its release: a
/// second finger or a mouse beside a touch keeps a press of its own, so one
/// pointer's release never decides with another's press.
#[derive(Resource, Default)]
struct Pressed(HashMap<PointerId, Press>);

/// The presses held as the frame's pointer events began, for a frame whose
/// releases reached nothing to be replayed from.
#[derive(Resource, Default)]
struct HeldBefore(HashMap<PointerId, Press>);

#[derive(Clone)]
struct Press {
    position: Vec2,
    /// Every `(instance, element id)` under the pointer at the press.
    under: HashSet<(Entity, String)>,
}

/// Picks and misses, decided on release as a click ([module docs](self)).
pub struct PickPlugin;

impl Plugin for PickPlugin {
    /// Both consumers sharing an app may add it; it installs once.
    fn is_unique(&self) -> bool {
        false
    }

    fn build(&self, app: &mut App) {
        if !crate::installed::first_install::<Self>(app) {
            return;
        }
        app.add_message::<Picked>()
            .init_resource::<Pressed>()
            .init_resource::<ReleasedElsewhere>()
            .init_resource::<HeldBefore>()
            .add_observer(on_press)
            .add_observer(on_release)
            .add_systems(
                PreUpdate,
                (
                    note_held_presses.before(PickingSystems::Hover),
                    release_reaching_nothing.after(PickingSystems::Hover),
                ),
            )
            .add_systems(PostUpdate, forget_unreleased_press);
    }
}

/// What the pointer is over at `position`: the element nearest it, every
/// element under it, and the topmost instance whose rectangle holds it.
struct Under {
    nearest: Option<(Entity, String)>,
    all: HashSet<(Entity, String)>,
    topmost: Option<Entity>,
}

/// The cameras a pointer can be over, and the window that counts as primary.
#[derive(SystemParam)]
struct Views<'w, 's> {
    cameras: Query<
        'w,
        's,
        (
            &'static Camera,
            &'static GlobalTransform,
            Option<&'static RenderTarget>,
        ),
    >,
    primary: Query<'w, 's, Entity, With<PrimaryWindow>>,
}

fn under(
    position: Vec2,
    target: &NormalizedRenderTarget,
    instances: &Query<(Entity, &EntityInstance), Without<Superseded>>,
    views: &Views,
    elements: &ElementMeshes,
) -> Under {
    let primary = views.primary.iter().next();
    let mut held: Vec<(Entity, &EntityInstance, &Camera, &GlobalTransform)> = instances
        .iter()
        .filter(|(_, instance)| instance.ready)
        .filter_map(|(entity, instance)| {
            let (camera, transform, drawn) = views.cameras.get(instance.camera).ok()?;
            if !draws_to(drawn, primary, target) {
                return None;
            }
            camera
                .is_active
                .then(|| camera.logical_viewport_rect())
                .flatten()?
                .contains(position)
                .then_some((entity, instance, camera, transform))
        })
        .collect();
    // Every element of every instance under the pointer, with the order of
    // the camera it was seen through. Collected across instances rather than
    // resolved one instance at a time: where several share a camera they
    // share a ray, and the nearest of all of them is the one under it.
    let mut hits: Vec<(isize, f32)> = Vec::new();
    let mut hit_of: Vec<(Entity, String)> = Vec::new();
    for (entity, instance, camera, transform) in &held {
        let Ok(ray) = camera.viewport_to_world(transform, position) else {
            continue;
        };
        for (id, targets) in &instance.bindings.elements {
            let Some(distance) = elements.cast(targets, ray) else {
                continue;
            };
            hits.push((camera.order, distance));
            hit_of.push((*entity, id.clone()));
        }
    }
    held.sort_by_key(|(_, _, camera, _)| std::cmp::Reverse(camera.order));
    Under {
        nearest: winner(&hits).map(|won| hit_of[won].clone()),
        all: hit_of.into_iter().collect(),
        topmost: held.first().map(|(entity, ..)| *entity),
    }
}

/// Whether a pointer event is the scene's: it bubbles up from its target, so
/// it is taken once, and only where it landed on the window rather than on
/// something drawn over the scene. The one rule every pointer handler here
/// shares — picks, the camera's gestures, a gizmo's grab — so a press on a
/// panel is never the scene's to any of them.
///
/// It leans on Bevy's picking reaching the window, which needs its pointer
/// input and its window picking enabled, and on nothing else in the scene
/// being pickable under the pointer: a mesh picking backend that picks this
/// crate's handles or a host's meshes takes the press off the window.
pub(crate) fn on_the_scene(
    entity: Entity,
    original: Entity,
    windows: &Query<(), With<Window>>,
) -> bool {
    entity == original && windows.contains(entity)
}

/// Notes what a press found, for its release to decide by.
fn on_press(
    press: On<PointerPress>,
    instances: Query<(Entity, &EntityInstance), Without<Superseded>>,
    views: Views,
    elements: ElementMeshes,
    windows: Query<(), With<Window>>,
    mut pressed: ResMut<Pressed>,
) {
    // A click is the primary button's, or a touch's.
    if press.button != PointerButton::Primary
        || !on_the_scene(press.entity, press.original_event_target(), &windows)
    {
        return;
    }
    // Whether the press takes a gizmo's handle is not known yet — the gizmo
    // decides in `Update`, from where the pointer is now, after this — so it
    // is decided at the release, by whether a gizmo drag is in hand then.
    let position = press.pointer.position;
    pressed.0.insert(
        press.pointer.id,
        Press {
            position,
            under: under(
                position,
                &press.pointer.target,
                &instances,
                &views,
                &elements,
            )
            .all,
        },
    );
}

/// Decides a release as a pick, a miss or nothing ([module docs](self)).
#[allow(clippy::too_many_arguments)]
fn on_release(
    release: On<PointerRelease>,
    instances: Query<(Entity, &EntityInstance), Without<Superseded>>,
    views: Views,
    elements: ElementMeshes,
    windows: Query<(), With<Window>>,
    gizmos: Option<Res<GizmoPointer>>,
    mut pressed: ResMut<Pressed>,
    mut elsewhere: ResMut<ReleasedElsewhere>,
    mut picks: MessageWriter<Picked>,
) {
    // A release bubbles from where it landed; take it once. A click is the
    // primary button's, or a touch's.
    if release.entity != release.original_event_target() || release.button != PointerButton::Primary
    {
        return;
    }
    let pointer = release.pointer.id;
    // Only the window's release decides a pick. Bevy sends a release to
    // everything hovered, in no set order — a label over the scene among
    // them — so a release anywhere else leaves the press for the window's,
    // and only notes that the press has been let go; see
    // `forget_unreleased_press`.
    if !windows.contains(release.entity) {
        elsewhere.0.insert(pointer);
        return;
    }
    let Some(press) = pressed.0.remove(&pointer) else {
        return;
    };
    released(
        &press,
        &release.pointer.location(),
        &instances,
        &views,
        &elements,
        gizmos.as_deref(),
        &mut picks,
    );
}

/// Notes the presses held before this frame's pointer events, for
/// `release_reaching_nothing` to replay from.
fn note_held_presses(pressed: Res<Pressed>, mut before: ResMut<HeldBefore>) {
    before.0.clone_from(&pressed.0);
}

/// Decides the releases Bevy sent nowhere, because the pointer was over
/// nothing the frame before ([module docs](self)).
///
/// The pointer's inputs this frame are replayed in order from the press it
/// held before them, so each release pairs with the press that came before
/// it: the presses' observers have all run by now, and what they left holds
/// only the last. The window is what the pointer is over for every input of
/// the frame, as it is for the presses' observers: a press counts and a
/// release decides only over it, and a release anywhere else lets its press
/// go, as one released onto a panel is let go.
#[allow(clippy::too_many_arguments)]
fn release_reaching_nothing(
    mut inputs: MessageReader<PointerInput>,
    hovered: Option<Res<HoverMap>>,
    previously: Option<Res<PreviousHoverMap>>,
    instances: Query<(Entity, &EntityInstance), Without<Superseded>>,
    views: Views,
    elements: ElementMeshes,
    windows: Query<(), With<Window>>,
    gizmos: Option<Res<GizmoPointer>>,
    before: Res<HeldBefore>,
    mut pressed: ResMut<Pressed>,
    mut picks: MessageWriter<Picked>,
) {
    let (Some(hovered), Some(previously)) = (hovered, previously) else {
        return;
    };
    let mut replayed: HashMap<PointerId, Option<Press>> = HashMap::new();
    for input in inputs.read() {
        let pointer = input.pointer_id;
        // A release Bevy sent somewhere was decided there, or let go by
        // `forget_unreleased_press`.
        if previously
            .get(&pointer)
            .is_some_and(|over| !over.is_empty())
        {
            continue;
        }
        let over_window = hovered
            .get(&pointer)
            .is_some_and(|over| over.keys().any(|entity| windows.contains(*entity)));
        match input.action {
            PointerAction::Press(PointerButton::Primary) => {
                if !over_window {
                    continue;
                }
                let press = Press {
                    position: input.location.position,
                    under: under(
                        input.location.position,
                        &input.location.target,
                        &instances,
                        &views,
                        &elements,
                    )
                    .all,
                };
                replayed.insert(pointer, Some(press));
            }
            PointerAction::Release(PointerButton::Primary) => {
                let held = replayed
                    .entry(pointer)
                    .or_insert_with(|| before.0.get(&pointer).cloned())
                    .take();
                if let (Some(press), true) = (held, over_window) {
                    released(
                        &press,
                        &input.location,
                        &instances,
                        &views,
                        &elements,
                        gizmos.as_deref(),
                        &mut picks,
                    );
                }
            }
            _ => {}
        }
    }
    for (pointer, held) in replayed {
        match held {
            Some(press) => pressed.0.insert(pointer, press),
            None => pressed.0.remove(&pointer),
        };
    }
}

/// Decides the window's release of `press` at `location` as a pick, a miss
/// or nothing, and sends the pick.
fn released(
    press: &Press,
    location: &Location,
    instances: &Query<(Entity, &EntityInstance), Without<Superseded>>,
    views: &Views,
    elements: &ElementMeshes,
    gizmos: Option<&GizmoPointer>,
    picks: &mut MessageWriter<Picked>,
) {
    // A press that took a gizmo's handle is the gizmo's, and so is its
    // release: the button is still the gizmo's as the release arrives, since
    // the gizmo lets go of it only in `Update`, after this — whether its drag
    // is in hand or was ended by the gizmo going inert or being taken away.
    if gizmos.is_some_and(|gizmos| gizmos.owns_button()) {
        return;
    }
    let position = location.position;
    let found = under(position, &location.target, instances, views, elements);
    let Some((instance, element_id)) = decide(press, found, position) else {
        return;
    };
    let Ok((_, held)) = instances.get(instance) else {
        return;
    };
    let picked = Picked {
        instance,
        instance_id: held.id.clone(),
        element_id,
    };
    match &picked.element_id {
        Some(element) => log::info!("{}: picked element {element}", picked.instance_id),
        None => log::info!("{}: a click on no element", picked.instance_id),
    }
    picks.write(picked);
}

/// The pointers whose release this frame landed somewhere other than the
/// window.
#[derive(Resource, Default)]
struct ReleasedElsewhere(HashSet<PointerId>);

/// Forgets a press that was let go this frame without a release on the
/// window taking it — a press released over a panel. Left, it would pair with
/// a later release back over the scene and pick something no click chose.
/// Read from the release, not from mouse buttons, so a touch, which sets no
/// mouse button, keeps its press while it is held. A release on the window is
/// decided in `PreUpdate`, before this runs.
fn forget_unreleased_press(mut elsewhere: ResMut<ReleasedElsewhere>, mut pressed: ResMut<Pressed>) {
    for pointer in elsewhere.0.drain() {
        pressed.0.remove(&pointer);
    }
}

/// What a release decides, given what its press found and what is under the
/// pointer now ([module docs](self)): the instance clicked, and the element,
/// or none for a miss.
fn decide(press: &Press, found: Under, position: Vec2) -> Option<(Entity, Option<String>)> {
    match found.nearest {
        Some(nearest) if press.under.contains(&nearest) => {
            let (instance, element_id) = nearest;
            Some((instance, Some(element_id)))
        }
        // Over an element the press did not find: not a click on anything.
        Some(_) => None,
        None => {
            if position.distance(press.position) > MISS_SLOP {
                return None;
            }
            Some((found.topmost?, None))
        }
    }
}

/// Which hit wins, among `(camera order, distance along that camera's ray)`.
///
/// **Two instances sharing a camera share a ray, so their distances are
/// comparable and the nearest wins. Two on different cameras are not** — the
/// distances run from different origins along different rays, so comparing
/// them is meaningless and the higher camera order decides instead, as it
/// does on screen.
///
/// A view that gives every instance its own camera therefore resolves by
/// order alone, which is the same answer it got before this distinguished the
/// two cases. A view with one camera over a whole world resolves by depth,
/// which is the only answer that is right there.
fn winner(hits: &[(isize, f32)]) -> Option<usize> {
    hits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)))
        .map(|(index, _)| index)
}

/// The element meshes as drawn: what a pick casts over and a selection
/// outline traces.
#[derive(SystemParam)]
pub struct ElementMeshes<'w, 's> {
    meshes: Res<'w, Assets<Mesh>>,
    handles: Query<'w, 's, &'static Mesh3d>,
    transforms: Query<'w, 's, &'static GlobalTransform>,
    weights: Query<'w, 's, &'static MorphWeights>,
    mesh_weights: Query<'w, 's, &'static MeshMorphWeights>,
    bounds: Query<'w, 's, &'static Aabb>,
}

impl ElementMeshes<'_, '_> {
    /// The element's own mesh asset, `None` for a group or a mesh that is
    /// not a triangle list.
    pub fn mesh(&self, targets: &ElementTargets) -> Option<(AssetId<Mesh>, &Mesh)> {
        let handle = &self.handles.get(targets.mesh?).ok()?.0;
        let mesh = self.meshes.get(handle)?;
        (mesh.primitive_topology() == PrimitiveTopology::TriangleList)
            .then_some((handle.id(), mesh))
    }

    /// Whether a morph is moving the element: any of its weights is not
    /// zero, so its drawn vertices are not its base mesh's.
    pub fn morphing(&self, targets: &ElementTargets) -> bool {
        targets
            .morph
            .and_then(|entity| self.weights.get(entity).ok())
            .is_some_and(|weights| weights.weights().iter().any(|weight| *weight != 0.0))
    }

    /// The corners, in the world, of a box sure to hold one mesh as it is
    /// drawn: its bounding box where no morph moves it, and that box widened
    /// by each morph target's offsets at that target's weight where one does.
    ///
    /// **Not its drawn vertices' box.** Building those costs every vertex of
    /// every target, which a view paying it each frame for a speaking face
    /// cannot afford. Each target's offsets are boxed once per mesh, in
    /// `offsets`, and widening by them costs one step per target. The box is
    /// conservative — each target widens it as far as that target alone could
    /// — so it may be larger than the drawn shape, never smaller. A mesh's
    /// weights are its own or its node's, as the mesh names them. `None` for
    /// a mesh with no bounding box.
    pub fn mesh_corners(&self, mesh: Entity, offsets: &mut MorphOffsets) -> Option<[Vec3; 8]> {
        let aabb = self.bounds.get(mesh).ok()?;
        let world = self.transforms.get(mesh).ok()?.affine();
        let centre = Vec3::from(aabb.center);
        let half = Vec3::from(aabb.half_extents);
        let base = [centre - half, centre + half];
        let weights: &[f32] = match self.mesh_weights.get(mesh) {
            Ok(MeshMorphWeights::Value { weights }) => weights,
            Ok(MeshMorphWeights::Reference(node)) => self
                .weights
                .get(*node)
                .map(MorphWeights::weights)
                .unwrap_or_default(),
            Err(_) => &[],
        };
        let moving = weights.iter().any(|weight| *weight != 0.0);
        let asset = self
            .handles
            .get(mesh)
            .ok()
            .and_then(|handle| Some((handle.0.id(), self.meshes.get(&handle.0)?)));
        let [low, high] = match (moving, asset) {
            (true, Some((id, asset))) => match asset.try_morph_targets().ok() {
                Some(deltas) => {
                    let vertices = asset.count_vertices();
                    let held = offsets
                        .0
                        .entry(id)
                        .and_modify(|held| {
                            if held.0 != (vertices, deltas.len()) {
                                *held =
                                    ((vertices, deltas.len()), target_offsets(deltas, vertices));
                            }
                        })
                        .or_insert_with(|| {
                            ((vertices, deltas.len()), target_offsets(deltas, vertices))
                        });
                    widened(base, &held.1, weights)
                }
                None => base,
            },
            _ => base,
        };
        Some(std::array::from_fn(|corner| {
            world.transform_point3(Vec3::new(
                if corner & 1 == 0 { low.x } else { high.x },
                if corner & 2 == 0 { low.y } else { high.y },
                if corner & 4 == 0 { low.z } else { high.z },
            ))
        }))
    }

    /// The distance along `ray` to the nearest triangle of the element as it
    /// is drawn, either side.
    ///
    /// **An element no morph is moving is cast where it stands**: a ray that
    /// misses its bounding box misses it, and one that does not is met with
    /// its own vertices under its transform, with nothing copied. A cast is
    /// paid on every press, over every element, so building the whole
    /// robot's vertices again each time is the cost to avoid. **An element a
    /// morph is moving is cast as drawn**
    /// ([`ElementMeshes::drawn`]), with no box first: the box is the base
    /// mesh's, and a morph can carry vertices past it — an opened mouth
    /// below a closed one's box — where a box test would miss a real hit.
    pub fn cast(&self, targets: &ElementTargets, ray: Ray3d) -> Option<f32> {
        if self.morphing(targets) {
            return self.drawn(targets)?.hit(ray);
        }
        let entity = targets.mesh?;
        let (_, mesh) = self.mesh(targets)?;
        let world = self.transforms.get(entity).ok()?.affine();
        if let Ok(aabb) = self.bounds.get(entity) {
            if !ray_meets_box(ray, &world, aabb) {
                return None;
            }
        }
        let positions = mesh
            .try_attribute(Mesh::ATTRIBUTE_POSITION)
            .ok()?
            .as_float3()?;
        cast_triangles(ray, &world, positions, mesh.try_indices().ok())
    }

    /// The element's mesh as it is drawn this frame: its vertices with the
    /// morph targets applied at their current weights, in world space.
    pub fn drawn(&self, targets: &ElementTargets) -> Option<DrawnMesh<'_>> {
        let (_, mesh) = self.mesh(targets)?;
        let base = mesh
            .try_attribute(Mesh::ATTRIBUTE_POSITION)
            .ok()?
            .as_float3()?;
        let weights = targets
            .morph
            .and_then(|entity| self.weights.get(entity).ok())
            .map(MorphWeights::weights)
            .unwrap_or_default();
        let deltas = mesh.try_morph_targets().ok().map(Vec::as_slice);
        let world = self.transforms.get(targets.mesh?).ok()?.affine();
        let positions = morphed(base, deltas, weights)
            .into_iter()
            .map(|p| world.transform_point3(Vec3::from(p)).to_array())
            .collect();
        Some(DrawnMesh {
            positions,
            indices: mesh.try_indices().ok(),
        })
    }
}

/// Each mesh's morph targets' offset boxes, kept between frames by whoever
/// widens boxes by them, and remade when a mesh's vertex or offset count
/// changes.
/// One mesh's held offset boxes, with the vertex and offset counts they were
/// made from.
type HeldOffsets = ((usize, usize), Vec<[Vec3; 2]>);

#[derive(Default)]
pub struct MorphOffsets(HashMap<AssetId<Mesh>, HeldOffsets>);

/// The box of each morph target's offsets: `deltas` is every target's
/// offsets in turn, `vertices` to a target.
fn target_offsets(deltas: &[MorphAttributes], vertices: usize) -> Vec<[Vec3; 2]> {
    if vertices == 0 {
        return Vec::new();
    }
    deltas
        .chunks(vertices)
        .map(|target| {
            target
                .iter()
                .fold([Vec3::ZERO, Vec3::ZERO], |[low, high], delta| {
                    [low.min(delta.position), high.max(delta.position)]
                })
        })
        .collect()
}

/// `base` widened by each target's offset box at its weight: as far as that
/// target could carry any vertex, either way, for a negative weight too.
fn widened(base: [Vec3; 2], offsets: &[[Vec3; 2]], weights: &[f32]) -> [Vec3; 2] {
    offsets
        .iter()
        .zip(weights)
        .fold(base, |[low, high], ([least, most], weight)| {
            let (a, b) = (*least * *weight, *most * *weight);
            [low + a.min(b), high + a.max(b)]
        })
}

/// A mesh's triangles as drawn, in world space.
pub struct DrawnMesh<'a> {
    pub positions: Vec<[f32; 3]>,
    pub indices: Option<&'a Indices>,
}

impl DrawnMesh<'_> {
    /// The distance along `ray` to the nearest triangle it meets, either
    /// side, since an asset's materials may be double-sided.
    pub(crate) fn hit(&self, ray: Ray3d) -> Option<f32> {
        cast_triangles(ray, &Affine3A::IDENTITY, &self.positions, self.indices)
    }
}

/// The distance along `ray` to the nearest triangle of these, placed by
/// `world`, either side, since an asset's materials may be double-sided.
fn cast_triangles(
    ray: Ray3d,
    world: &Affine3A,
    positions: &[[f32; 3]],
    indices: Option<&Indices>,
) -> Option<f32> {
    let cull = Backfaces::Include;
    match indices {
        Some(Indices::U16(i)) => {
            ray_mesh_intersection(ray, world, positions, None, Some(i), None, cull)
        }
        Some(Indices::U32(i)) => {
            ray_mesh_intersection(ray, world, positions, None, Some(i), None, cull)
        }
        None => ray_mesh_intersection::<u32>(ray, world, positions, None, None, None, cull),
    }
    .map(|hit| hit.distance)
}

/// Whether `ray` meets a mesh's bounding box, the box in the mesh's own space
/// and placed by `world`: the ray is carried into that space rather than the
/// box out of it, so the box stays a box.
fn ray_meets_box(ray: Ray3d, world: &Affine3A, aabb: &Aabb) -> bool {
    let to_local = world.inverse();
    let origin = to_local.transform_point3(ray.origin);
    let direction = to_local.transform_vector3(*ray.direction);
    let (min, max) = (
        Vec3::from(aabb.center - aabb.half_extents),
        Vec3::from(aabb.center + aabb.half_extents),
    );
    // The slab test: the ray is inside the box between where it has entered
    // every pair of faces and before it has left any.
    let (mut near, mut far) = (0.0_f32, f32::INFINITY);
    for axis in 0..3 {
        if direction[axis].abs() < f32::EPSILON {
            if origin[axis] < min[axis] || origin[axis] > max[axis] {
                return false;
            }
            continue;
        }
        let a = (min[axis] - origin[axis]) / direction[axis];
        let b = (max[axis] - origin[axis]) / direction[axis];
        near = near.max(a.min(b));
        far = far.min(a.max(b));
        if near > far {
            return false;
        }
    }
    true
}

/// `base` positions moved by each morph target at its weight. `deltas` holds
/// the targets one after the other, `base.len()` displacements each, as
/// Bevy's glTF loader lays them out.
pub fn morphed(
    base: &[[f32; 3]],
    deltas: Option<&[MorphAttributes]>,
    weights: &[f32],
) -> Vec<[f32; 3]> {
    let mut positions = base.to_vec();
    let (Some(deltas), n) = (deltas, base.len()) else {
        return positions;
    };
    for (target, weight) in weights.iter().enumerate() {
        if *weight == 0.0 {
            continue;
        }
        let Some(target) = deltas.get(target * n..(target + 1) * n) else {
            break;
        };
        for (position, delta) in positions.iter_mut().zip(target) {
            position[0] += weight * delta.position.x;
            position[1] += weight * delta.position.y;
            position[2] += weight * delta.position.z;
        }
    }
    positions
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A box widened by a target at a weight holds wherever that target can
    /// carry a vertex at it, a negative weight included, and a target at no
    /// weight leaves it alone.
    #[test]
    fn a_widened_box_holds_what_the_morph_can_reach() {
        let base = [Vec3::splat(-1.0), Vec3::splat(1.0)];
        // One target that opens a mouth: down by up to 0.5, and no further up.
        let offsets = [[Vec3::new(0.0, -0.5, 0.0), Vec3::ZERO]];
        let [low, high] = widened(base, &offsets, &[1.0]);
        assert_eq!(low, Vec3::new(-1.0, -1.5, -1.0));
        assert_eq!(high, Vec3::splat(1.0));
        let [low, high] = widened(base, &offsets, &[-2.0]);
        assert_eq!(low, Vec3::splat(-1.0));
        assert_eq!(high, Vec3::new(1.0, 2.0, 1.0));
        assert_eq!(widened(base, &offsets, &[0.0]), base);
    }

    /// Each target's offsets are boxed apart from the others'.
    #[test]
    fn each_target_is_boxed_on_its_own() {
        let offset = |y: f32| MorphAttributes::new(Vec3::new(0.0, y, 0.0), Vec3::ZERO, Vec3::ZERO);
        let deltas = [offset(1.0), offset(0.5), offset(-2.0), offset(0.0)];
        let boxes = target_offsets(&deltas, 2);
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0], [Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)]);
        assert_eq!(boxes[1], [Vec3::new(0.0, -2.0, 0.0), Vec3::ZERO]);
    }

    /// A ray meets a placed box when it passes through it, and not when it
    /// passes beside it or the box is behind it.
    #[test]
    fn a_ray_meets_a_box_it_passes_through() {
        let aabb = Aabb::from_min_max(Vec3::splat(-1.0), Vec3::splat(1.0));
        let world = Affine3A::from_translation(Vec3::new(5.0, 0.0, 0.0));
        let toward = Ray3d::new(Vec3::ZERO, Dir3::X);
        assert!(ray_meets_box(toward, &world, &aabb));
        let beside = Ray3d::new(Vec3::new(0.0, 3.0, 0.0), Dir3::X);
        assert!(!ray_meets_box(beside, &world, &aabb));
        let away = Ray3d::new(Vec3::ZERO, Dir3::NEG_X);
        assert!(!ray_meets_box(away, &world, &aabb));
    }

    /// A release whose button a gizmo still owns picks nothing — a drag in
    /// hand, or one its gizmo ended by going inert — and the same release
    /// with the button free is a click.
    #[test]
    fn a_release_a_gizmo_owns_picks_nothing() {
        use crate::transform_gizmo::GizmoHandle;

        let picks_for = |gizmos: GizmoPointer| picks_through(gizmos, true);
        let grip = (Entity::from_raw_u32(9).unwrap(), GizmoHandle::Arrow(0));
        assert_eq!(picks_for(GizmoPointer::default()), 1, "a click on nothing");
        let held = GizmoPointer {
            dragging: Some(grip),
            ..default()
        };
        assert_eq!(picks_for(held), 0, "the release of a drag in hand");
        let ended = GizmoPointer {
            ended: Some(grip),
            ..default()
        };
        assert_eq!(
            picks_for(ended),
            0,
            "the release of a drag ended by going inert"
        );
    }

    /// A camera filling an image — a screen's texture — is a picture on a
    /// mesh, not a view the pointer is over: a click where its viewport would
    /// be is on no instance.
    #[test]
    fn an_instance_drawn_into_an_image_is_not_under_the_pointer() {
        assert_eq!(picks_through(GizmoPointer::default(), true), 1);
        assert_eq!(picks_through(GizmoPointer::default(), false), 0);
    }

    /// The picks a release over a robot with no elements makes, given what
    /// the gizmos hold, through a camera drawing to the window the release is
    /// in or into an image.
    fn picks_through(gizmos: GizmoPointer, to_window: bool) -> usize {
        use bevy::camera::{
            ComputedCameraValues, ImageRenderTarget, NormalizedRenderTarget, RenderTarget,
            RenderTargetInfo,
        };
        use bevy::picking::backend::HitData;
        use bevy::picking::events::Pointer;
        use bevy::picking::pointer::Location;
        use bevy::window::WindowRef;

        {
            let mut world = World::new();
            world.init_resource::<Assets<Mesh>>();
            world.init_resource::<Pressed>();
            world.init_resource::<ReleasedElsewhere>();
            world.init_resource::<Messages<Picked>>();
            world.insert_resource(gizmos);
            world.add_observer(on_release);
            let window = world.spawn(Window::default()).id();
            // A robot with no elements, seen through a camera the release is
            // over: a click there is on the robot and on nothing in it.
            let camera = world
                .spawn((
                    Camera {
                        computed: ComputedCameraValues {
                            target_info: Some(RenderTargetInfo {
                                physical_size: UVec2::splat(400),
                                scale_factor: 1.0,
                            }),
                            ..default()
                        },
                        ..default()
                    },
                    GlobalTransform::IDENTITY,
                    if to_window {
                        RenderTarget::Window(WindowRef::Entity(window))
                    } else {
                        RenderTarget::Image(ImageRenderTarget::from(Handle::<Image>::default()))
                    },
                ))
                .id();
            world.spawn(EntityInstance {
                id: "robot".to_string(),
                asset_path: String::new(),
                meta: Default::default(),
                bindings: Default::default(),
                camera,
                draw_order: 0,
                ready: true,
            });
            world.resource_mut::<Pressed>().0.insert(
                PointerId::Mouse,
                Press {
                    position: Vec2::splat(200.0),
                    under: HashSet::new(),
                },
            );
            let target =
                NormalizedRenderTarget::Window(WindowRef::Entity(window).normalize(None).unwrap());
            world.trigger(PointerRelease {
                entity: window,
                pointer: Pointer::new(
                    PointerId::Mouse,
                    Location {
                        target,
                        position: Vec2::splat(200.0),
                    },
                ),
                button: PointerButton::Primary,
                hit: HitData::new(window, 0.0, None, None),
            });
            world.flush();
            world.resource::<Messages<Picked>>().len()
        }
    }

    /// An app picking as a host's does, through Bevy's own pointer events,
    /// with a robot of no elements seen through a camera drawing to the
    /// window: a click anywhere on it is a miss of that robot.
    fn picking_app() -> (App, Entity) {
        use bevy::camera::{ComputedCameraValues, RenderTarget, RenderTargetInfo};
        use bevy::picking::{InteractionPlugin, PickingPlugin};
        use bevy::window::{PrimaryWindow, WindowRef};

        let mut app = App::new();
        app.add_plugins((MinimalPlugins, PickingPlugin, InteractionPlugin, PickPlugin))
            .init_resource::<Assets<Mesh>>();
        let window = app
            .world_mut()
            .spawn((Window::default(), PrimaryWindow))
            .id();
        let camera = app
            .world_mut()
            .spawn((
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: UVec2::splat(400),
                            scale_factor: 1.0,
                        }),
                        ..default()
                    },
                    ..default()
                },
                GlobalTransform::IDENTITY,
                RenderTarget::Window(WindowRef::Entity(window)),
            ))
            .id();
        app.world_mut().spawn(EntityInstance {
            id: "robot".to_string(),
            asset_path: String::new(),
            meta: Default::default(),
            bindings: Default::default(),
            camera,
            draw_order: 0,
            ready: true,
        });
        app.world_mut().spawn(PointerId::Mouse);
        app.update();
        (app, window)
    }

    /// Sends the mouse's inputs, at `x` across the window, all in one frame,
    /// and counts the picks that frame decided.
    fn in_one_frame(app: &mut App, window: Entity, inputs: &[(f32, PointerAction)]) -> usize {
        use bevy::camera::NormalizedRenderTarget;
        use bevy::window::WindowRef;

        let target =
            NormalizedRenderTarget::Window(WindowRef::Entity(window).normalize(None).unwrap());
        for (x, action) in inputs {
            app.world_mut().write_message(PointerInput {
                pointer_id: PointerId::Mouse,
                location: Location {
                    target: target.clone(),
                    position: Vec2::new(*x, 200.0),
                },
                action: *action,
            });
        }
        app.update();
        app.world_mut()
            .resource_mut::<Messages<Picked>>()
            .drain()
            .count()
    }

    fn moved() -> PointerAction {
        PointerAction::Move { delta: Vec2::ONE }
    }

    /// A pointer that reaches the window, presses and lets go all in one
    /// frame clicks: Bevy sends that release nowhere, as the pointer was
    /// over nothing the frame before.
    #[test]
    fn a_press_and_release_in_the_frame_the_pointer_arrives_is_a_click() {
        let (mut app, window) = picking_app();
        let press = PointerAction::Press(PointerButton::Primary);
        let release = PointerAction::Release(PointerButton::Primary);
        assert_eq!(
            in_one_frame(
                &mut app,
                window,
                &[(200.0, moved()), (200.0, press), (200.0, release)]
            ),
            1
        );
    }

    /// A click whose press and release fall in one frame, or in frames of
    /// their own, picks once either way; one that moves past the slop over
    /// nothing in that frame is a drag, and picks nothing.
    #[test]
    fn a_click_picks_once_however_its_frames_fall() {
        let press = || PointerAction::Press(PointerButton::Primary);
        let release = || PointerAction::Release(PointerButton::Primary);

        let (mut app, window) = picking_app();
        assert_eq!(in_one_frame(&mut app, window, &[(200.0, moved())]), 0);
        assert_eq!(
            in_one_frame(&mut app, window, &[(200.0, press()), (200.0, release())]),
            1,
            "pressed and let go in one frame, over the window the frame before"
        );
        assert_eq!(in_one_frame(&mut app, window, &[(200.0, press())]), 0);
        assert_eq!(
            in_one_frame(&mut app, window, &[(200.0, release())]),
            1,
            "pressed and let go in frames of their own"
        );

        let (mut app, window) = picking_app();
        let far = 200.0 + MISS_SLOP + 10.0;
        assert_eq!(
            in_one_frame(
                &mut app,
                window,
                &[
                    (200.0, moved()),
                    (200.0, press()),
                    (far, moved()),
                    (far, release())
                ]
            ),
            0,
            "a drag over nothing, in the frame the pointer arrives"
        );
    }

    /// Something drawn over the scene, under the mouse while it holds an
    /// entity: a panel, as a picking backend reports one.
    #[derive(Resource, Default)]
    struct Panel(Option<Entity>);

    fn panel_hits(panel: Res<Panel>, mut hits: MessageWriter<bevy::picking::backend::PointerHits>) {
        use bevy::picking::backend::{HitData, PointerHits};

        if let Some(panel) = panel.0 {
            hits.write(PointerHits::new(
                PointerId::Mouse,
                vec![(panel, HitData::new(panel, 0.0, None, None))],
                1.0,
            ));
        }
    }

    /// A release decided where the pointer is now is the window's only when
    /// the window is what the pointer is over, so none of these picks: a tap
    /// over a panel the pointer reached that frame; a tap the frame after the
    /// pointer was over a panel, whose release Bevy sent to the panel; a
    /// press on the scene let go over a panel after the pointer was over
    /// nothing.
    #[test]
    fn a_quick_tap_over_a_panel_is_not_the_scene_s() {
        let press = || PointerAction::Press(PointerButton::Primary);
        let release = || PointerAction::Release(PointerButton::Primary);
        let with_panel = || {
            let (mut app, window) = picking_app();
            app.init_resource::<Panel>().add_systems(
                PreUpdate,
                panel_hits.in_set(bevy::picking::PickingSystems::Backend),
            );
            let panel = app.world_mut().spawn_empty().id();
            (app, window, panel)
        };

        let (mut app, window, panel) = with_panel();
        app.world_mut().resource_mut::<Panel>().0 = Some(panel);
        assert_eq!(
            in_one_frame(
                &mut app,
                window,
                &[(200.0, moved()), (200.0, press()), (200.0, release())]
            ),
            0,
            "over the panel now"
        );

        let (mut app, window, panel) = with_panel();
        app.world_mut().resource_mut::<Panel>().0 = Some(panel);
        assert_eq!(in_one_frame(&mut app, window, &[(200.0, moved())]), 0);
        app.world_mut().resource_mut::<Panel>().0 = None;
        assert_eq!(
            in_one_frame(&mut app, window, &[(200.0, press()), (200.0, release())]),
            0,
            "over the panel the frame before, so let go there"
        );

        // Pressed on the scene, then off every view — over nothing, so the
        // release reaches nothing — then back over a panel to let go.
        let (mut app, window, panel) = with_panel();
        assert_eq!(
            in_one_frame(&mut app, window, &[(200.0, moved()), (200.0, press())]),
            0
        );
        assert_eq!(in_one_frame(&mut app, window, &[(600.0, moved())]), 0);
        app.world_mut().resource_mut::<Panel>().0 = Some(panel);
        assert_eq!(
            in_one_frame(&mut app, window, &[(200.0, moved()), (200.0, release())]),
            0,
            "let go over the panel, from over nothing"
        );
        assert!(
            !app.world()
                .resource::<Pressed>()
                .0
                .contains_key(&PointerId::Mouse),
            "and the press is let go"
        );
        // So a later release over the scene, from over nothing, has no press
        // to pick with.
        app.world_mut().resource_mut::<Panel>().0 = None;
        assert_eq!(in_one_frame(&mut app, window, &[(600.0, moved())]), 0);
        assert_eq!(
            in_one_frame(&mut app, window, &[(200.0, moved()), (200.0, release())]),
            0,
            "no press left to pick with"
        );
    }

    /// A release in a frame replayed from over nothing pairs with the press
    /// held before that frame when no press comes before it, though a later
    /// press in the frame has taken its place by the time it is decided; and
    /// a press outside every view is no press at all.
    #[test]
    fn a_replayed_release_pairs_with_the_press_before_it() {
        let press = || PointerAction::Press(PointerButton::Primary);
        let release = || PointerAction::Release(PointerButton::Primary);

        let (mut app, window) = picking_app();
        assert_eq!(
            in_one_frame(&mut app, window, &[(200.0, moved()), (200.0, press())]),
            0
        );
        assert_eq!(in_one_frame(&mut app, window, &[(600.0, moved())]), 0);
        let far = 200.0 + MISS_SLOP + 50.0;
        assert_eq!(
            in_one_frame(
                &mut app,
                window,
                &[
                    (200.0, moved()),
                    (200.0, release()),
                    (far, moved()),
                    (far, press())
                ]
            ),
            1,
            "the release where its own press was"
        );

        let (mut app, window) = picking_app();
        assert_eq!(
            in_one_frame(&mut app, window, &[(401.0, moved()), (401.0, press())]),
            0
        );
        assert_eq!(
            in_one_frame(&mut app, window, &[(399.5, moved()), (399.5, release())]),
            0,
            "pressed outside every view"
        );
    }

    /// Clicks queued in the frame the pointer arrives each pick once, every
    /// release with the press before it.
    #[test]
    fn clicks_queued_in_one_frame_each_pick() {
        let press = || PointerAction::Press(PointerButton::Primary);
        let release = || PointerAction::Release(PointerButton::Primary);

        let (mut app, window) = picking_app();
        assert_eq!(
            in_one_frame(
                &mut app,
                window,
                &[
                    (200.0, moved()),
                    (200.0, press()),
                    (200.0, release()),
                    (200.0, press()),
                    (200.0, release())
                ]
            ),
            2,
            "two clicks in one frame"
        );

        let (mut app, window) = picking_app();
        assert_eq!(
            in_one_frame(
                &mut app,
                window,
                &[
                    (200.0, moved()),
                    (200.0, press()),
                    (200.0, release()),
                    (200.0, press())
                ]
            ),
            1,
            "the first click"
        );
        assert_eq!(
            in_one_frame(&mut app, window, &[(200.0, release())]),
            1,
            "the second press, let go the next frame"
        );
    }

    /// A press is kept while it is held — by a touch too, which sets no
    /// mouse button — and forgotten once let go somewhere other than the
    /// window.
    #[test]
    fn a_press_released_elsewhere_is_forgotten() {
        let mut app = App::new();
        app.init_resource::<ReleasedElsewhere>()
            .init_resource::<Pressed>()
            .add_systems(Update, forget_unreleased_press);
        let held = |app: &App, pointer: PointerId| {
            app.world().resource::<Pressed>().0.contains_key(&pointer)
        };
        let touch = PointerId::Touch(1);
        for pointer in [PointerId::Mouse, touch] {
            app.world_mut().resource_mut::<Pressed>().0.insert(
                pointer,
                Press {
                    position: Vec2::ZERO,
                    under: HashSet::new(),
                },
            );
        }
        app.update();
        app.update();
        assert!(
            held(&app, touch),
            "held while not let go, with no mouse button down"
        );
        app.world_mut()
            .resource_mut::<ReleasedElsewhere>()
            .0
            .insert(touch);
        app.update();
        assert!(!held(&app, touch), "forgotten once let go elsewhere");
        assert!(
            held(&app, PointerId::Mouse),
            "another pointer's press is its own"
        );
    }

    const ROBOT: Entity = Entity::PLACEHOLDER;

    fn pair(element: &str) -> (Entity, String) {
        (ROBOT, element.to_string())
    }

    fn press_on(elements: &[&str], position: Vec2) -> Press {
        Press {
            position,
            under: elements.iter().map(|element| pair(element)).collect(),
        }
    }

    fn found(nearest: Option<&str>) -> Under {
        Under {
            nearest: nearest.map(pair),
            all: HashSet::new(),
            topmost: Some(ROBOT),
        }
    }

    /// A release on an element the press found picks it, however far the
    /// pointer moved between the two — as a click in Studio's scene does.
    #[test]
    fn a_release_on_an_element_the_press_found_picks_it() {
        let press = press_on(&["head", "body"], Vec2::ZERO);
        let picked = decide(&press, found(Some("body")), Vec2::new(80.0, 0.0)).unwrap();
        assert_eq!(picked, (ROBOT, Some("body".to_string())));
    }

    /// A release on an element the press never found is not a click on it.
    #[test]
    fn a_release_on_another_element_picks_nothing() {
        let press = press_on(&["head"], Vec2::ZERO);
        assert!(decide(&press, found(Some("arm")), Vec2::ZERO).is_none());
    }

    /// A release over nothing is a miss only within the slop: a drag that
    /// turned the view and ended over empty space clears nothing.
    #[test]
    fn a_release_over_nothing_misses_only_within_the_slop() {
        let press = press_on(&[], Vec2::ZERO);
        let miss = decide(&press, found(None), Vec2::new(MISS_SLOP, 0.0)).unwrap();
        assert_eq!(miss, (ROBOT, None));
        assert!(decide(&press, found(None), Vec2::new(MISS_SLOP + 1.0, 0.0)).is_none());
    }

    /// Instances on one camera resolve by depth, which is what a world with a
    /// single camera over it needs and what resolving one instance at a time
    /// cannot do.
    #[test]
    fn sharing_a_camera_the_nearest_wins() {
        assert_eq!(winner(&[(0, 9.0), (0, 2.0), (0, 5.0)]), Some(1));
        // Order within the slice must not decide it: the same hits in the
        // other order pick the same one.
        assert_eq!(winner(&[(0, 2.0), (0, 9.0), (0, 5.0)]), Some(0));
    }

    /// Across cameras the order decides, because the distances are measured
    /// along different rays and cannot be compared at all.
    #[test]
    fn across_cameras_the_higher_order_wins_however_far() {
        assert_eq!(
            winner(&[(0, 0.1), (3, 100.0)]),
            Some(1),
            "a far hit on a camera drawn over a near one still wins"
        );
        // The nearest of the topmost camera's hits, and not merely the last
        // of them: order alone would answer 1 here.
        assert_eq!(winner(&[(5, 1.0), (5, 4.0), (2, 0.01)]), Some(0));
    }

    #[test]
    fn nothing_hit_is_no_pick() {
        assert_eq!(winner(&[]), None);
    }

    fn delta(x: f32, y: f32) -> MorphAttributes {
        MorphAttributes::new(Vec3::new(x, y, 0.0), Vec3::ZERO, Vec3::ZERO)
    }

    #[test]
    fn morph_targets_move_the_vertices_by_their_weights() {
        let base = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        // Two targets: the first moves the second vertex up, the second
        // moves the first vertex right.
        let deltas = [
            delta(0.0, 0.0),
            delta(0.0, 2.0),
            delta(1.0, 0.0),
            delta(0.0, 0.0),
        ];
        assert_eq!(morphed(&base, Some(&deltas), &[]), base.to_vec());
        assert_eq!(
            morphed(&base, Some(&deltas), &[0.5, 1.0]),
            vec![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        );
        assert_eq!(morphed(&base, None, &[1.0]), base.to_vec());
    }

    /// A thin mesh a morph opens is hit where it is drawn open, and missed
    /// where its base vertices lay.
    #[test]
    fn a_pick_meets_the_mesh_where_its_morphs_put_it() {
        // A sliver at y ∈ [0, 0.01] whose morph opens it to y ∈ [0, 1].
        let base = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.01, 0.0],
            [0.0, 0.01, 0.0],
        ];
        let deltas = [
            delta(0.0, 0.0),
            delta(0.0, 0.0),
            delta(0.0, 0.99),
            delta(0.0, 0.99),
        ];
        let indices = Indices::U16(vec![0, 1, 2, 0, 2, 3]);
        let ray = Ray3d::new(Vec3::new(0.5, 0.5, 10.0), Dir3::NEG_Z);
        let drawn = |weight: f32| DrawnMesh {
            positions: morphed(&base, Some(&deltas), &[weight]),
            indices: Some(&indices),
        };
        assert_eq!(
            drawn(0.0).hit(ray),
            None,
            "the base sliver is below the ray"
        );
        assert_eq!(drawn(1.0).hit(ray), Some(10.0), "the opened mesh is hit");
    }
}
