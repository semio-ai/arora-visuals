//! Feature-edge selection marking: each selected element traced in line gizmos
//! over the instance it belongs to.
//!
//! The edges are the element mesh's, as three.js's `EdgesGeometry` finds them:
//! the boundary, plus the creases where two adjacent triangles turn by more
//! than [`CREASE_DEGREES`]. They are computed once per mesh from its base
//! triangles and cached; each frame their ends are placed where the mesh
//! actually draws them, morphs applied, for the same reason a pick casts over
//! the drawn mesh. Selecting a group marks every element mesh under it.
//!
//! # This is not a silhouette outline, and the difference is not cosmetic
//!
//! **Feature edges approximate a silhouette only on geometry that is nearly
//! flat and has few creases.** On the 2D shapes of a face they land on the
//! shape's edge and read as an outline. On a curved 3D body every crease above
//! the threshold is drawn too, and the result reads as a wireframe over the
//! object rather than a line around it.
//!
//! A silhouette outline — one line around the whole of what is selected, which
//! is what the Studio scene draws — is a different mechanism: render the
//! selection to a mask, dilate it, composite the difference. It is not here.
//! A consumer that wants it needs to write it, and should not reach for this
//! expecting it.
//!
//! Line gizmos are used because WebGL2 has no wireframe, which settles *how*
//! edges get drawn. It does not make edges the right thing to draw.

use std::collections::HashMap;
use std::sync::Arc;

use bevy::gizmos::config::{GizmoConfig, GizmoConfigGroup, GizmoLineConfig};
use bevy::mesh::Indices;
use bevy::prelude::*;

use crate::binding::ElementTargets;
use crate::instance::{EntityInstance, Superseded};
use crate::pick::ElementMeshes;

/// The angle two adjacent triangles turn by for their shared edge to be
/// outlined — the authoring app's glow threshold.
const CREASE_DEGREES: f32 = 2.0;

/// The outline's colour: the authoring app's selection red, nearly opaque.
const GLOW: Color = Color::srgba(1.0, 16.0 / 255.0, 16.0 / 255.0, 0.9);

/// The outline's line width, in logical pixels.
const GLOW_WIDTH: f32 = 3.0;

pub struct GlowPlugin;

impl Plugin for GlowPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Selections>()
            .init_resource::<FeatureEdges>()
            .insert_gizmo_config(
                SelectionGlow,
                GizmoConfig {
                    line: GizmoLineConfig {
                        width: GLOW_WIDTH,
                        ..default()
                    },
                    // In front of every layer the instance draws into.
                    depth_bias: -1.0,
                    ..default()
                },
            )
            .add_systems(
                PostUpdate,
                draw_glow.after(bevy::transform::TransformSystems::Propagate),
            );
    }
}

/// The gizmo group the outline draws in.
#[derive(Default, Reflect, GizmoConfigGroup)]
struct SelectionGlow;

/// The selected elements of each instance, by instance id.
///
/// Kept apart from the instances so a selection survives an instance's
/// reload: what a person selected is their state, not the asset's.
#[derive(Resource, Default)]
pub struct Selections(pub HashMap<String, Vec<String>>);

/// Each mesh's feature edges, found on first selection.
#[derive(Resource, Default)]
struct FeatureEdges(HashMap<AssetId<Mesh>, Arc<Vec<[u32; 2]>>>);

/// Outline every selected element of every indexed instance.
fn draw_glow(
    selections: Res<Selections>,
    instances: Query<&EntityInstance, Without<Superseded>>,
    parents: Query<&ChildOf>,
    elements: ElementMeshes,
    mut edges: ResMut<FeatureEdges>,
    mut gizmos: Gizmos<SelectionGlow>,
) {
    for instance in &instances {
        if !instance.bindings.indexed {
            continue;
        }
        let Some(selected) = selections.0.get(&instance.id) else {
            continue;
        };
        for id in selected {
            let Some(targets) = instance.bindings.elements.get(id) else {
                continue;
            };
            let outlined: Vec<&ElementTargets> = if targets.mesh.is_some() {
                vec![targets]
            } else {
                instance.bindings
                    .elements
                    .values()
                    .filter(|element| {
                        element.mesh.is_some() && descends(element.node, targets.node, &parents)
                    })
                    .collect()
            };
            for element in outlined {
                let Some((mesh_id, mesh)) = elements.mesh(element) else {
                    continue;
                };
                let outline = edges
                    .0
                    .entry(mesh_id)
                    .or_insert_with(|| {
                        let base = mesh
                            .try_attribute(Mesh::ATTRIBUTE_POSITION)
                            .ok()
                            .and_then(|positions| positions.as_float3())
                            .unwrap_or_default();
                        Arc::new(feature_edges(base, mesh.try_indices().ok(), CREASE_DEGREES))
                    })
                    .clone();
                let Some(drawn) = elements.drawn(element) else {
                    continue;
                };
                for [a, b] in outline.iter() {
                    let (Some(a), Some(b)) = (
                        drawn.positions.get(*a as usize),
                        drawn.positions.get(*b as usize),
                    ) else {
                        continue;
                    };
                    gizmos.line(Vec3::from(*a), Vec3::from(*b), GLOW);
                }
            }
        }
    }
}

/// Whether `entity` is `ancestor` or lies under it.
fn descends(mut entity: Entity, ancestor: Entity, parents: &Query<&ChildOf>) -> bool {
    loop {
        if entity == ancestor {
            return true;
        }
        match parents.get(entity) {
            Ok(parent) => entity = parent.parent(),
            Err(_) => return false,
        }
    }
}

/// A triangle mesh's feature edges, as vertex index pairs: the edges of one
/// triangle only (the boundary), and the edges two triangles share at an
/// angle above `crease_degrees`. Vertices at one position (to 1e-4) are one
/// vertex, so a seam the mesh splits for its normals or UVs is no boundary;
/// degenerate triangles are skipped. three.js's `EdgesGeometry`.
fn feature_edges(
    positions: &[[f32; 3]],
    indices: Option<&Indices>,
    crease_degrees: f32,
) -> Vec<[u32; 2]> {
    let corners: Vec<u32> = match indices {
        Some(Indices::U16(i)) => i.iter().map(|&i| u32::from(i)).collect(),
        Some(Indices::U32(i)) => i.clone(),
        None => (0..positions.len() as u32).collect(),
    };
    let key = |p: &[f32; 3]| p.map(|c| (c * 1e4).round() as i64);
    let mut welded: HashMap<[i64; 3], u32> = HashMap::new();
    let canonical: Vec<u32> = positions
        .iter()
        .enumerate()
        .map(|(i, p)| *welded.entry(key(p)).or_insert(i as u32))
        .collect();
    let crease = crease_degrees.to_radians().cos();
    let mut open: HashMap<(u32, u32), (Vec3, [u32; 2])> = HashMap::new();
    let mut edges = Vec::new();
    for &[i, j, k] in corners.as_chunks::<3>().0 {
        let (Some(&i), Some(&j), Some(&k)) = (
            canonical.get(i as usize),
            canonical.get(j as usize),
            canonical.get(k as usize),
        ) else {
            continue;
        };
        let vertices = [i, j, k];
        let [a, b, c] = vertices.map(|v| Vec3::from(positions[v as usize]));
        let Some(normal) = (b - a).cross(c - a).try_normalize() else {
            continue;
        };
        for (u, v) in [(0, 1), (1, 2), (2, 0)] {
            let (u, v) = (vertices[u], vertices[v]);
            if u == v {
                continue;
            }
            let shared = (u.min(v), u.max(v));
            match open.remove(&shared) {
                Some((other, edge)) => {
                    if normal.dot(other) <= crease {
                        edges.push(edge);
                    }
                }
                None => {
                    open.insert(shared, (normal, [u, v]));
                }
            }
        }
    }
    edges.extend(open.into_values().map(|(_, edge)| edge));
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(mut edges: Vec<[u32; 2]>) -> Vec<[u32; 2]> {
        for edge in &mut edges {
            edge.sort();
        }
        edges.sort();
        edges
    }

    /// A flat quad's outline is its four sides, not its diagonal.
    #[test]
    fn a_flat_quad_is_outlined_by_its_boundary() {
        let quad = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = Indices::U16(vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(
            sorted(feature_edges(&quad, Some(&indices), CREASE_DEGREES)),
            vec![[0, 1], [0, 3], [1, 2], [2, 3]]
        );
    }

    /// Unindexed triangles meeting at the same positions share their edge.
    #[test]
    fn coincident_vertices_are_one_vertex() {
        let soup = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        assert_eq!(feature_edges(&soup, None, CREASE_DEGREES).len(), 4);
    }

    /// A fold is outlined along its crease; a turn below the threshold is not.
    #[test]
    fn a_crease_is_outlined_above_the_threshold() {
        let fold = |z: f32| {
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, z],
            ]
        };
        let indices = Indices::U16(vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(
            feature_edges(&fold(1.0), Some(&indices), CREASE_DEGREES).len(),
            5
        );
        assert_eq!(
            feature_edges(&fold(0.001), Some(&indices), CREASE_DEGREES).len(),
            4
        );
    }
}
