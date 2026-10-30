//! The selection outline: one line around the whole of what is selected,
//! drawn over everything in front of it — the outline the Studio scene draws.
//!
//! A host marks a camera with [`OutlineView`], and that camera's view gets the
//! outline of every [`Selections`] entry.
//!
//! # How it is drawn
//!
//! **The selected meshes are drawn a second time, by a mask camera that sees
//! nothing else**, into a texture the size of the view. A quad held just in
//! front of the view camera reads that mask and paints wherever the mask is
//! empty but some of the selection lies within the outline's width.
//!
//! Because the mask holds only the selection, the line follows the
//! selection's own silhouette even where other geometry stands in front of
//! it, and it is painted over that geometry. That is deliberate and it is
//! Studio's: its outline draws hidden edges in the same colour as visible ones.
//! Because the mask camera draws each mesh as the GPU does, a mesh a morph has
//! moved is outlined where it now is.
//!
//! The mask camera finds the selection through a render layer of its own,
//! [`OUTLINE_LAYER`], which a selected mesh joins and leaves as the selection
//! changes. Whatever layers the mesh had before are kept, so it stays visible
//! everywhere it was.
//!
//! [`glow`](crate::glow) marks a selection differently — with its feature
//! edges — and the two read the same selections.

use std::collections::HashSet;

use bevy::asset::{embedded_asset, embedded_path, AssetPath};
use bevy::camera::visibility::RenderLayers;
use bevy::camera::RenderTarget;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, CompareFunction, Extent3d, RenderPipelineDescriptor, ShaderType,
    SpecializedMeshPipelineError, TextureFormat,
};
use bevy::shader::ShaderRef;

use crate::instance::{EntityInstance, Superseded};
use crate::select::{covered, Selections};

/// The render layer the mask camera draws, and only selected meshes are on.
pub const OUTLINE_LAYER: usize = 30;

/// Outline this camera's view of the selection.
#[derive(Component, Clone, Copy, Debug)]
pub struct OutlineView {
    pub colour: Color,
    /// The line's width, in logical pixels.
    pub width: f32,
}

impl Default for OutlineView {
    /// White, as Studio draws a selection that brings no colour of its own.
    fn default() -> Self {
        Self {
            colour: Color::WHITE,
            width: 3.0,
        }
    }
}

/// The quad the outline is painted on. A host that counts or frames the
/// meshes in its scene should leave this out: it rides the camera and is not
/// part of what the camera is looking at.
#[derive(Component)]
pub struct OutlineOverlay;

/// Marks the camera that draws the selection mask.
#[derive(Component)]
struct OutlineMask;

/// What an outlined view owns: its mask camera, the mask, and the quad.
#[derive(Component)]
struct OutlineParts {
    mask_camera: Entity,
    overlay: Entity,
    mask: Handle<Image>,
    material: Handle<OutlineMaterial>,
}

/// On a mesh currently in the mask: the layers it had before it joined.
#[derive(Component)]
struct Outlined {
    before: Option<RenderLayers>,
}

#[derive(ShaderType, Clone, Debug, Default)]
pub struct OutlineUniform {
    pub colour: Vec4,
    /// In physical pixels.
    pub width: f32,
    pub _padding: Vec3,
}

/// The overlay's material; see `outline.wesl`.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct OutlineMaterial {
    #[uniform(0)]
    pub outline: OutlineUniform,
    #[texture(1)]
    #[sampler(2)]
    pub mask: Handle<Image>,
}

impl Material for OutlineMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(embedded_path!("outline.wesl")).with_source("embedded"),
        )
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    /// Painted over everything, never hidden by it: the quad sits just in
    /// front of the camera, and the line it carries belongs on top.
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

pub struct OutlinePlugin;

impl Plugin for OutlinePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "outline.wesl");
        app.add_plugins(MaterialPlugin::<OutlineMaterial>::default())
            .init_resource::<Selections>()
            .add_systems(
                Update,
                (attach_outline, mark_selection, fit_outline).chain(),
            );
    }
}

/// Gives each newly outlined view its mask camera, mask and overlay.
fn attach_outline(
    mut commands: Commands,
    views: Query<(Entity, &Camera, &OutlineView), Without<OutlineParts>>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<OutlineMaterial>>,
) {
    for (view, camera, settings) in &views {
        // Resized to the view as soon as the view knows its size.
        let mask = images.add(Image::new_target_texture(
            1,
            1,
            TextureFormat::Rgba8Unorm,
            None,
        ));
        let material = materials.add(OutlineMaterial {
            outline: uniform(settings, 1.0),
            mask: mask.clone(),
        });
        let mask_camera = commands
            .spawn((
                Camera3d::default(),
                Camera {
                    // Before the view, so the mask is drawn before it is read.
                    order: camera.order - 1,
                    clear_color: ClearColorConfig::Custom(Color::NONE),
                    ..default()
                },
                RenderTarget::Image(mask.clone().into()),
                RenderLayers::layer(OUTLINE_LAYER),
                Msaa::Off,
                Tonemapping::None,
                Transform::IDENTITY,
                ChildOf(view),
                OutlineMask,
            ))
            .id();
        let overlay = commands
            .spawn((
                Mesh3d(meshes.add(Rectangle::new(1.0, 1.0))),
                MeshMaterial3d(material.clone()),
                Transform::IDENTITY,
                NotShadowCaster,
                NotShadowReceiver,
                ChildOf(view),
                OutlineOverlay,
            ))
            .id();
        commands.entity(view).insert(OutlineParts {
            mask_camera,
            overlay,
            mask,
            material,
        });
    }
}

fn uniform(settings: &OutlineView, scale: f32) -> OutlineUniform {
    OutlineUniform {
        colour: LinearRgba::from(settings.colour).to_vec4(),
        width: settings.width * scale,
        ..default()
    }
}

/// Moves the selection's meshes into the mask and everything else out of it.
fn mark_selection(
    mut commands: Commands,
    selections: Res<Selections>,
    instances: Query<&EntityInstance, Without<Superseded>>,
    parents: Query<&ChildOf>,
    layers: Query<Option<&RenderLayers>>,
    outlined: Query<(Entity, &Outlined)>,
) {
    let mut wanted = HashSet::new();
    for instance in &instances {
        if !instance.bindings.indexed {
            continue;
        }
        let Some(selected) = selections.0.get(&instance.id) else {
            continue;
        };
        for id in selected {
            wanted.extend(
                covered(instance, id, &parents)
                    .into_iter()
                    .filter_map(|element| element.mesh),
            );
        }
    }

    for (mesh, marked) in &outlined {
        if wanted.remove(&mesh) {
            continue;
        }
        let mut entity = commands.entity(mesh);
        match &marked.before {
            Some(before) => entity.insert(before.clone()),
            None => entity.remove::<RenderLayers>(),
        };
        entity.remove::<Outlined>();
    }
    for mesh in wanted {
        let Ok(before) = layers.get(mesh) else {
            continue;
        };
        let joined = before.cloned().unwrap_or_default().with(OUTLINE_LAYER);
        commands.entity(mesh).insert((
            joined,
            Outlined {
                before: before.cloned(),
            },
        ));
    }
}

/// Keeps the mask the size of the view, the mask camera looking through the
/// view's lens, and the overlay exactly covering the view.
fn fit_outline(
    views: Query<(&Camera, &Projection, &OutlineView, &OutlineParts)>,
    mut cameras: Query<&mut Projection, (With<OutlineMask>, Without<OutlineParts>)>,
    mut overlays: Query<&mut Transform, With<OutlineOverlay>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<OutlineMaterial>>,
) {
    for (camera, projection, settings, parts) in &views {
        let Some(size) = camera.physical_viewport_size() else {
            continue;
        };
        let extent = Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        };
        let resized = images
            .get(&parts.mask)
            .is_some_and(|image| image.texture_descriptor.size != extent);
        if resized {
            if let Some(mut image) = images.get_mut(&parts.mask) {
                image.resize(extent);
            }
        }
        if let Ok(mut lens) = cameras.get_mut(parts.mask_camera) {
            *lens = projection.clone();
        }
        // Only when something changed: touching a material re-uploads it. A
        // resize counts, because it replaces the mask the material binds.
        let wanted = uniform(settings, camera.target_scaling_factor().unwrap_or(1.0));
        let stale = materials.get(&parts.material).is_some_and(|material| {
            material.outline.colour != wanted.colour || material.outline.width != wanted.width
        });
        if stale || resized {
            if let Some(mut material) = materials.get_mut(&parts.material) {
                material.outline = wanted;
            }
        }
        // Just past the near plane, sized to fill the view exactly there.
        if let (Projection::Perspective(lens), Ok(mut transform)) =
            (projection, overlays.get_mut(parts.overlay))
        {
            let distance = lens.near * 2.0;
            let height = 2.0 * distance * (lens.fov * 0.5).tan();
            *transform = Transform::from_xyz(0.0, 0.0, -distance).with_scale(Vec3::new(
                height * lens.aspect_ratio,
                height,
                1.0,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy::ecs::system::RunSystemOnce;

    use super::*;
    use crate::binding::{ElementTargets, SceneBindings};
    use crate::meta::AssetMeta;

    /// A world with one indexed instance of two elements, each with a mesh:
    /// one already on a layer of its own, one on no layer at all.
    fn world() -> (World, Entity, Entity) {
        let mut world = World::new();
        world.init_resource::<Selections>();
        let layered = world.spawn(RenderLayers::layer(2)).id();
        let bare = world.spawn_empty().id();
        let camera = world.spawn_empty().id();

        let mut bindings = SceneBindings::default();
        for (id, mesh) in [("layered", layered), ("bare", bare)] {
            bindings.elements.insert(
                id.to_string(),
                ElementTargets {
                    node: mesh,
                    mesh: Some(mesh),
                    morph: None,
                },
            );
        }
        bindings.indexed = true;
        world.spawn(EntityInstance {
            id: "robot".to_string(),
            asset_path: String::new(),
            meta: AssetMeta::default(),
            bindings,
            camera,
            draw_order: 0,
            ready: true,
        });
        (world, layered, bare)
    }

    fn select(world: &mut World, ids: &[&str]) {
        world.resource_mut::<Selections>().0.insert(
            "robot".to_string(),
            ids.iter().map(|id| id.to_string()).collect(),
        );
        world.run_system_once(mark_selection).unwrap();
    }

    /// Joining the mask must not take a mesh off any layer it was on, and
    /// leaving it must put back exactly what was there — including nothing.
    /// A mesh left on the wrong layers either vanishes from views it belonged
    /// to or stays outlined after it is deselected, and neither shows as an
    /// error.
    #[test]
    fn a_selected_mesh_joins_the_mask_and_leaves_it_as_it_was() {
        let (mut world, layered, bare) = world();

        select(&mut world, &["layered", "bare"]);
        assert_eq!(
            world.get::<RenderLayers>(layered),
            Some(&RenderLayers::layer(2).with(OUTLINE_LAYER)),
        );
        assert_eq!(
            world.get::<RenderLayers>(bare),
            Some(&RenderLayers::default().with(OUTLINE_LAYER)),
            "a mesh on no layer is on the default one, and keeps it",
        );

        select(&mut world, &["bare"]);
        assert_eq!(
            world.get::<RenderLayers>(layered),
            Some(&RenderLayers::layer(2)),
            "deselected, it is back on its own layer alone",
        );
        assert!(world.get::<Outlined>(layered).is_none());
        assert!(world.get::<Outlined>(bare).is_some(), "still selected");

        select(&mut world, &[]);
        assert!(
            world.get::<RenderLayers>(bare).is_none(),
            "a mesh that had no layers has none again",
        );
    }

    /// Reselecting what is already selected changes nothing, so a selection
    /// that holds still costs no work and never flickers.
    #[test]
    fn a_held_selection_is_left_alone() {
        let (mut world, layered, _) = world();
        select(&mut world, &["layered"]);
        let before = world.get::<RenderLayers>(layered).cloned();
        select(&mut world, &["layered"]);
        assert_eq!(world.get::<RenderLayers>(layered).cloned(), before);
        assert!(world.get::<Outlined>(layered).is_some());
    }
}
