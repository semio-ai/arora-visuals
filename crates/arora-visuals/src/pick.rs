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

use bevy::ecs::system::SystemParam;
use bevy::math::Affine3A;
use bevy::mesh::morph::{MorphAttributes, MorphWeights};
use bevy::mesh::{Indices, PrimitiveTopology};
use std::collections::HashSet;

use bevy::picking::events::{PointerPress, PointerRelease};
use bevy::picking::mesh_picking::ray_cast::{ray_mesh_intersection, Backfaces};
use bevy::prelude::*;

use crate::binding::ElementTargets;
use crate::instance::{EntityInstance, Superseded};
use crate::transform_gizmo::GizmoPointer;

/// A pointer press on an instance: the element under it, by the ids the
/// asset's own `RobotData` declares, or `None` for a press that hit no
/// element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Picked {
    pub instance_id: String,
    pub element_id: Option<String>,
}

/// The picks since a consumer last drained them.
#[derive(Resource, Default)]
pub struct Picks(pub Vec<Picked>);

/// How far, in logical pixels, the pointer may move between press and release
/// and a release over nothing still count as a miss.
pub const MISS_SLOP: f32 = 2.0;

/// What a press found, kept until its release.
#[derive(Resource, Default)]
struct Pressed(Option<Press>);

struct Press {
    position: Vec2,
    /// Every `(instance id, element id)` under the pointer at the press.
    under: HashSet<(String, String)>,
    /// The press took a gizmo's handle, so its release is the gizmo's.
    taken: bool,
}

/// Picks and misses, decided on release as a click ([module docs](self)).
pub struct PickPlugin;

impl Plugin for PickPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Picks>()
            .init_resource::<Pressed>()
            .add_observer(on_press)
            .add_observer(on_release);
    }
}

/// What the pointer is over at `position`: the element nearest it, every
/// element under it, and the topmost instance whose rectangle holds it.
struct Under {
    nearest: Option<(String, String)>,
    all: HashSet<(String, String)>,
    topmost: Option<String>,
}

fn under(
    position: Vec2,
    instances: &Query<&EntityInstance, Without<Superseded>>,
    cameras: &Query<(&Camera, &GlobalTransform)>,
    elements: &ElementMeshes,
) -> Under {
    let mut held: Vec<(&EntityInstance, &Camera, &GlobalTransform)> = instances
        .iter()
        .filter(|instance| instance.ready)
        .filter_map(|instance| {
            let (camera, transform) = cameras.get(instance.camera).ok()?;
            camera
                .is_active
                .then(|| camera.logical_viewport_rect())
                .flatten()?
                .contains(position)
                .then_some((instance, camera, transform))
        })
        .collect();
    // Every element of every instance under the pointer, with the order of
    // the camera it was seen through. Collected across instances rather than
    // resolved one instance at a time: where several share a camera they
    // share a ray, and the nearest of all of them is the one under it.
    let mut hits: Vec<(isize, f32)> = Vec::new();
    let mut hit_of: Vec<(String, String)> = Vec::new();
    for (instance, camera, transform) in &held {
        let Ok(ray) = camera.viewport_to_world(transform, position) else {
            continue;
        };
        for (id, targets) in &instance.bindings.elements {
            let Some(distance) = elements.drawn(targets).and_then(|mesh| mesh.hit(ray)) else {
                continue;
            };
            hits.push((camera.order, distance));
            hit_of.push((instance.id.clone(), id.clone()));
        }
    }
    held.sort_by_key(|(_, camera, _)| std::cmp::Reverse(camera.order));
    Under {
        nearest: winner(&hits).map(|won| hit_of[won].clone()),
        all: hit_of.into_iter().collect(),
        topmost: held.first().map(|(instance, ..)| instance.id.clone()),
    }
}

/// Whether a pointer event is the scene's: it bubbles up from its target, so
/// it is taken once, and only where it landed on the window rather than on
/// something drawn over the scene.
fn on_the_scene(entity: Entity, original: Entity, windows: &Query<(), With<Window>>) -> bool {
    entity == original && windows.contains(entity)
}

/// Notes what a press found, for its release to decide by.
fn on_press(
    press: On<PointerPress>,
    instances: Query<&EntityInstance, Without<Superseded>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    elements: ElementMeshes,
    windows: Query<(), With<Window>>,
    gizmos: Option<Res<GizmoPointer>>,
    mut pressed: ResMut<Pressed>,
) {
    if !on_the_scene(press.entity, press.original_event_target(), &windows) {
        return;
    }
    let position = press.pointer.position;
    // A gizmo's handles are drawn over the scene but reach the window like
    // any press on it, so one that takes a handle is the gizmo's, and so is
    // its release.
    let taken = gizmos.is_some_and(|gizmos| gizmos.engaged());
    let all = if taken {
        HashSet::new()
    } else {
        under(position, &instances, &cameras, &elements).all
    };
    pressed.0 = Some(Press {
        position,
        under: all,
        taken,
    });
}

/// Decides a release as a pick, a miss or nothing ([module docs](self)).
#[allow(clippy::too_many_arguments)]
fn on_release(
    release: On<PointerRelease>,
    instances: Query<&EntityInstance, Without<Superseded>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    elements: ElementMeshes,
    windows: Query<(), With<Window>>,
    mut pressed: ResMut<Pressed>,
    mut picks: ResMut<Picks>,
) {
    if !on_the_scene(release.entity, release.original_event_target(), &windows) {
        return;
    }
    let Some(press) = pressed.0.take() else {
        return;
    };
    if press.taken {
        return;
    }
    let position = release.pointer.position;
    let found = under(position, &instances, &cameras, &elements);
    if let Some(picked) = decide(&press, found, position) {
        match &picked.element_id {
            Some(element) => log::info!("{}: picked element {element}", picked.instance_id),
            None => log::info!("{}: a click on no element", picked.instance_id),
        }
        picks.0.push(picked);
    }
}

/// What a release decides, given what its press found and what is under the
/// pointer now ([module docs](self)).
fn decide(press: &Press, found: Under, position: Vec2) -> Option<Picked> {
    match found.nearest {
        Some(nearest) if press.under.contains(&nearest) => {
            let (instance_id, element_id) = nearest;
            Some(Picked {
                instance_id,
                element_id: Some(element_id),
            })
        }
        // Over an element the press did not find: not a click on anything.
        Some(_) => None,
        None => {
            if position.distance(press.position) > MISS_SLOP {
                return None;
            }
            Some(Picked {
                instance_id: found.topmost?,
                element_id: None,
            })
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

/// A mesh's triangles as drawn, in world space.
pub struct DrawnMesh<'a> {
    pub positions: Vec<[f32; 3]>,
    pub indices: Option<&'a Indices>,
}

impl DrawnMesh<'_> {
    /// The distance along `ray` to the nearest triangle it meets, either
    /// side, since an asset's materials may be double-sided.
    pub(crate) fn hit(&self, ray: Ray3d) -> Option<f32> {
        let world = Affine3A::IDENTITY;
        let cull = Backfaces::Include;
        let positions = &self.positions;
        match self.indices {
            Some(Indices::U16(i)) => {
                ray_mesh_intersection(ray, &world, positions, None, Some(i), None, cull)
            }
            Some(Indices::U32(i)) => {
                ray_mesh_intersection(ray, &world, positions, None, Some(i), None, cull)
            }
            None => ray_mesh_intersection::<u32>(ray, &world, positions, None, None, None, cull),
        }
        .map(|hit| hit.distance)
    }
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

    fn pair(element: &str) -> (String, String) {
        ("robot".to_string(), element.to_string())
    }

    fn press_on(elements: &[&str], position: Vec2) -> Press {
        Press {
            position,
            under: elements.iter().map(|element| pair(element)).collect(),
            taken: false,
        }
    }

    fn found(nearest: Option<&str>) -> Under {
        Under {
            nearest: nearest.map(pair),
            all: HashSet::new(),
            topmost: Some("robot".to_string()),
        }
    }

    /// A release on an element the press found picks it, however far the
    /// pointer moved between the two — as a click in Studio's scene does.
    #[test]
    fn a_release_on_an_element_the_press_found_picks_it() {
        let press = press_on(&["head", "body"], Vec2::ZERO);
        let picked = decide(&press, found(Some("body")), Vec2::new(80.0, 0.0)).unwrap();
        assert_eq!(picked.element_id.as_deref(), Some("body"));
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
        assert!(miss.element_id.is_none());
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
