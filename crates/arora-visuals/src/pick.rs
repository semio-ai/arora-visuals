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
//! registered, every press reaches the window entity and is resolved here.
//!
//! The press goes to the instances whose rectangle holds the pointer, the one
//! drawn last first. The nearest element its ray meets is the pick; a press
//! meeting no element of any of them is a miss of the topmost; a press outside
//! every drawn rectangle belongs to no instance.

use bevy::ecs::system::SystemParam;
use bevy::math::Affine3A;
use bevy::mesh::morph::{MorphAttributes, MorphWeights};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::picking::events::{Pointer, Press};
use bevy::picking::mesh_picking::ray_cast::{ray_mesh_intersection, Backfaces};
use bevy::prelude::*;

use crate::binding::ElementTargets;
use crate::instance::{EntityInstance, Superseded};

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

/// Resolve a press to a pick or a miss ([module docs](self)).
pub fn on_press(
    press: On<Pointer<Press>>,
    instances: Query<&EntityInstance, Without<Superseded>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    elements: ElementMeshes,
    mut picks: ResMut<Picks>,
) {
    // The press bubbles up from its target; resolve it once.
    if press.entity != press.original_event_target() {
        return;
    }
    let position = press.pointer_location.position;
    let mut under: Vec<(&EntityInstance, &Camera, &GlobalTransform)> = instances
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
    under.sort_by_key(|(_, camera, _)| std::cmp::Reverse(camera.order));
    for (instance, camera, transform) in &under {
        let Ok(ray) = camera.viewport_to_world(transform, position) else {
            continue;
        };
        let nearest = instance
            .bindings
            .elements
            .iter()
            .filter_map(|(id, targets)| Some((id, elements.drawn(targets)?.hit(ray)?)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((element_id, _)) = nearest {
            log::info!("{}: picked element {element_id}", instance.id);
            picks.0.push(Picked {
                instance_id: instance.id.clone(),
                element_id: Some(element_id.clone()),
            });
            return;
        }
    }
    if let Some((instance, _, _)) = under.first() {
        log::info!("{}: a press on no element", instance.id);
        picks.0.push(Picked {
            instance_id: instance.id.clone(),
            element_id: None,
        });
    }
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
    fn hit(&self, ray: Ray3d) -> Option<f32> {
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
