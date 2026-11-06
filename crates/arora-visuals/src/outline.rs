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
use bevy::camera::{CameraUpdateSystems, ImageRenderTarget, RenderTarget};
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

use crate::active_area::FramingSystems;
use crate::instance::{EntityInstance, Superseded};
#[cfg(test)]
use crate::select::Selection;
use crate::select::{covered, Highlight, Selections};

/// The render layer the mask camera draws, and only selected meshes are on.
pub const OUTLINE_LAYER: usize = 30;

/// Outline this camera's view of the selection.
#[derive(Component, Clone, Copy, Debug)]
pub struct OutlineView {
    /// The colour of a selection that brings none of its own; see
    /// [`Selections::highlight`] for when one does. The same holds for the
    /// width and the fill.
    pub colour: Color,
    /// The line's width, in logical pixels.
    pub width: f32,
    /// How strongly the selection itself is filled with the outline colour,
    /// from not at all to solid, for when a line alone is not emphasis enough.
    /// Like the line, the fill shows through whatever stands in front of the
    /// selection.
    pub fill: f32,
}

impl Default for OutlineView {
    /// A white line and no fill, as Studio draws a selection that brings no
    /// colour of its own.
    fn default() -> Self {
        Self {
            colour: Color::WHITE,
            width: 3.0,
            fill: 0.0,
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

/// What an outlined view owns: its mask camera, the mask, the quad, and the
/// render layer the quad is on, which the view joins beside its own.
#[derive(Component)]
struct OutlineParts {
    mask_camera: Entity,
    overlay: Entity,
    mask: Handle<Image>,
    material: Handle<OutlineMaterial>,
    layer: usize,
}

/// The first of the render layers outline quads are drawn on, one per
/// outlined view, upward from here. **A host must not draw its own entities
/// on these**: an outlined view joins its quad's layer, so anything else on
/// it would be drawn into that view.
pub const OVERLAY_LAYERS_FROM: usize = OUTLINE_LAYER + 1;

/// Which views hold which quad layers: slot `i` is layer
/// `OVERLAY_LAYERS_FROM + i`. A quad rides its view's camera, so on a layer
/// another camera drew it would show up in that camera's picture as a stray
/// rectangle; a layer of its own, which only its view joins, keeps it to the
/// view it belongs to, whatever other layers that view draws.
#[derive(Resource, Default)]
struct OverlayLayers(Vec<Option<Entity>>);

impl OverlayLayers {
    /// A free layer for `view`, reclaiming any slot whose view is gone.
    fn take(&mut self, view: Entity, alive: impl Fn(Entity) -> bool) -> usize {
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
        OVERLAY_LAYERS_FROM + slot
    }

    fn free(&mut self, layer: usize) {
        if let Some(held) = self.0.get_mut(layer - OVERLAY_LAYERS_FROM) {
            *held = None;
        }
    }
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
    pub fill: f32,
    pub _padding: Vec2,
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
            .init_resource::<OverlayLayers>()
            .add_systems(
                Update,
                (detach_outline, attach_outline, mark_selection).chain(),
            )
            // The mask follows the view's framing before Bevy computes either
            // camera's projection; the quad is fitted to the projection Bevy
            // then computed, sub-view and all.
            .add_systems(
                PostUpdate,
                (
                    sync_mask.after(FramingSystems).before(CameraUpdateSystems),
                    fit_overlay
                        .after(CameraUpdateSystems)
                        .after(TransformSystems::Propagate),
                ),
            );
    }
}

/// Gives each newly outlined view its mask camera, mask and overlay.
#[allow(clippy::type_complexity)]
fn attach_outline(
    mut commands: Commands,
    views: Query<(Entity, &Camera, &OutlineView, Option<&RenderLayers>), Without<OutlineParts>>,
    owners: Query<(), Or<(With<OutlineView>, With<OutlineParts>)>>,
    mut layers: ResMut<OverlayLayers>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<OutlineMaterial>>,
) {
    for (view, camera, settings, view_layers) in &views {
        // A holder is alive while it still asks for an outline, so a view
        // attaching earlier in this same pass, whose parts are not yet
        // inserted, keeps its layer.
        let layer = layers.take(view, |holder| owners.contains(holder));
        // Resized to the view as soon as the view knows its size.
        let mask = images.add(Image::new_target_texture(
            1,
            1,
            TextureFormat::Rgba8Unorm,
            None,
        ));
        let material = materials.add(OutlineMaterial {
            outline: uniform(settings, None, 1.0),
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
                // At the view's scale factor, so a projection Bevy sizes
                // from its target — an orthographic one sized to the window —
                // sizes the mask as it sizes the view.
                RenderTarget::Image(ImageRenderTarget {
                    handle: mask.clone(),
                    scale_factor: camera.target_scaling_factor().unwrap_or(1.0),
                }),
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
                RenderLayers::layer(layer),
                NotShadowCaster,
                NotShadowReceiver,
                ChildOf(view),
                OutlineOverlay,
            ))
            .id();
        commands.entity(view).insert((
            view_layers.cloned().unwrap_or_default().with(layer),
            OutlineParts {
                mask_camera,
                overlay,
                mask,
                material,
                layer,
            },
        ));
    }
}

/// Takes an outline away from a view no longer asking for one: its mask
/// camera and quad go, its layer is freed, and the view leaves that layer
/// alone — whatever other layers it has now, the host's to keep, including
/// any set since the outline came.
fn detach_outline(
    mut commands: Commands,
    views: Query<(Entity, &OutlineParts, Option<&RenderLayers>), Without<OutlineView>>,
    mut layers: ResMut<OverlayLayers>,
) {
    for (view, parts, view_layers) in &views {
        commands.entity(parts.mask_camera).despawn();
        commands.entity(parts.overlay).despawn();
        layers.free(parts.layer);
        let left = view_layers
            .cloned()
            .unwrap_or_default()
            .without(parts.layer);
        commands.entity(view).insert(left).remove::<OutlineParts>();
    }
}

/// What the shader is given: whatever the selection's highlight sets, the
/// view's own for whatever it leaves unset, and the width scaled to physical
/// pixels.
fn uniform(settings: &OutlineView, highlight: Option<Highlight>, scale: f32) -> OutlineUniform {
    let highlight = highlight.unwrap_or_default();
    OutlineUniform {
        colour: LinearRgba::from(highlight.colour.unwrap_or(settings.colour)).to_vec4(),
        width: highlight.width.unwrap_or(settings.width) * scale,
        fill: highlight.fill.unwrap_or(settings.fill),
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
        for selection in selected {
            wanted.extend(
                covered(instance, &selection.element, &parents)
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

/// Keeps the mask the size of the view and the mask camera looking through
/// the view's lens — its projection and its sub-view, so a view framed beside
/// a panel masks what it draws — and keeps the view on its quad's layer if
/// the host has since replaced the view's layers.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn sync_mask(
    mut commands: Commands,
    selections: Res<Selections>,
    views: Query<(
        Entity,
        &Camera,
        &Projection,
        &OutlineView,
        &OutlineParts,
        Option<&RenderLayers>,
    )>,
    mut cameras: Query<
        (&mut Camera, &mut Projection, &mut RenderTarget),
        (With<OutlineMask>, Without<OutlineParts>),
    >,
    mut overlays: Query<&mut Visibility, With<OutlineOverlay>>,
    outlined: Query<(), With<Outlined>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<OutlineMaterial>>,
) {
    // With nothing selected there is nothing to outline, and the mask pass
    // and the quad's dozens of samples a pixel are paid for nothing; both
    // stand down until a selection appears.
    let selecting = !outlined.is_empty();
    for (view, camera, projection, settings, parts, view_layers) in &views {
        if let Ok(mut visibility) = overlays.get_mut(parts.overlay) {
            visibility.set_if_neq(if selecting {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            });
        }
        let joined = RenderLayers::layer(parts.layer);
        if !view_layers.is_some_and(|layers| layers.intersects(&joined)) {
            commands
                .entity(view)
                .insert(view_layers.cloned().unwrap_or_default().with(parts.layer));
        }
        if let Ok((mut mask_camera, mut lens, mut target)) = cameras.get_mut(parts.mask_camera) {
            let scale = camera.target_scaling_factor().unwrap_or(1.0);
            if let RenderTarget::Image(image) = &mut *target {
                if image.scale_factor != scale {
                    image.scale_factor = scale;
                }
            }
            *lens = projection.clone();
            if mask_camera.is_active != selecting {
                mask_camera.is_active = selecting;
            }
            // Just before the view, wherever the view now stands, so the mask
            // is drawn before the view reads it.
            if mask_camera.order != camera.order - 1 {
                mask_camera.order = camera.order - 1;
            }
            if mask_camera.sub_camera_view != camera.sub_camera_view {
                mask_camera.sub_camera_view = camera.sub_camera_view;
            }
        }
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
        // Only when something changed: touching a material re-uploads it. A
        // resize counts, because it replaces the mask the material binds.
        let wanted = uniform(
            settings,
            selections.highlight(),
            camera.target_scaling_factor().unwrap_or(1.0),
        );
        let stale = materials.get(&parts.material).is_some_and(|material| {
            let held = &material.outline;
            held.colour != wanted.colour || held.width != wanted.width || held.fill != wanted.fill
        });
        if stale || resized {
            if let Some(mut material) = materials.get_mut(&parts.material) {
                material.outline = wanted;
            }
        }
    }
}

/// How far into the view the quad stands, in normalised device depth: with a
/// reversed depth range, halfway — twice the near plane's distance under a
/// perspective, midway between the planes under an orthographic projection.
/// The quad is drawn over everything whatever its depth; it only has to be
/// inside the view.
const OVERLAY_DEPTH: f32 = 0.5;

/// Fits the quad over the whole view, from the projection Bevy computed for
/// it this frame: the corners of the view at one depth, carried back into the
/// camera's space. That holds for an orthographic projection as for a
/// perspective one, and for an off-centre sub-view, where the view's centre
/// is not the camera's axis. The shader reads the mask by screen position,
/// so the quad need only cover the view; it is made a little larger, so
/// rounding never leaves an edge uncovered.
///
/// Fitted after propagation, and its world transform set here too, so the
/// quad stands where this frame's projection says rather than a frame
/// behind it.
#[allow(clippy::type_complexity)]
fn fit_overlay(
    views: Query<(&Camera, &GlobalTransform, &OutlineParts)>,
    mut overlays: Query<
        (&mut Transform, &mut GlobalTransform),
        (With<OutlineOverlay>, Without<OutlineParts>),
    >,
) {
    for (camera, at, parts) in &views {
        let Ok((mut transform, mut global)) = overlays.get_mut(parts.overlay) else {
            continue;
        };
        let wanted = overlay_over(camera.clip_from_view());
        if *transform != wanted {
            *transform = wanted;
        }
        *global = at.mul_transform(wanted);
    }
}

/// The quad's place in the camera's space under `clip_from_view`.
fn overlay_over(clip_from_view: Mat4) -> Transform {
    let view_from_clip = clip_from_view.inverse();
    let corner = |x: f32, y: f32| view_from_clip.project_point3(Vec3::new(x, y, OVERLAY_DEPTH));
    let (low, high) = (corner(-1.0, -1.0), corner(1.0, 1.0));
    let centre = (low + high) / 2.0;
    let size = (high - low).abs() * 1.02;
    Transform::from_translation(centre).with_scale(Vec3::new(size.x, size.y, 1.0))
}

#[cfg(test)]
mod tests {
    use bevy::ecs::system::RunSystemOnce;

    use super::*;
    use crate::binding::{ElementTargets, SceneBindings};
    use crate::meta::AssetMeta;
    use bevy::camera::{CameraProjection, SubCameraView};

    fn outline_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Image>()
            .init_asset::<Mesh>()
            .init_asset::<OutlineMaterial>()
            .init_resource::<OverlayLayers>()
            .add_systems(Update, (detach_outline, attach_outline).chain());
        app
    }

    fn quad_layer(app: &mut App, view: Entity) -> usize {
        app.world()
            .get::<OutlineParts>(view)
            .expect("outlined")
            .layer
    }

    /// Views outlined in the same frame each get a quad layer of their own.
    #[test]
    fn views_outlined_together_get_layers_of_their_own() {
        let mut app = outline_app();
        let first = app
            .world_mut()
            .spawn((Camera::default(), OutlineView::default()))
            .id();
        let second = app
            .world_mut()
            .spawn((Camera::default(), OutlineView::default()))
            .id();
        app.update();
        assert_ne!(quad_layer(&mut app, first), quad_layer(&mut app, second));
    }

    /// Taking an outline away leaves the view every layer but the quad's,
    /// including one the host gave it after the outline came.
    #[test]
    fn removing_an_outline_keeps_the_hosts_layers() {
        let mut app = outline_app();
        let view = app
            .world_mut()
            .spawn((
                Camera::default(),
                OutlineView::default(),
                RenderLayers::layer(0),
            ))
            .id();
        app.update();
        let quad = quad_layer(&mut app, view);
        let given = app
            .world()
            .get::<RenderLayers>(view)
            .cloned()
            .unwrap()
            .with(5);
        app.world_mut().entity_mut(view).insert(given);
        app.world_mut().entity_mut(view).remove::<OutlineView>();
        app.update();
        let left = app.world().get::<RenderLayers>(view).cloned().unwrap();
        assert_eq!(
            left,
            RenderLayers::from_layers(&[0, 5]),
            "quad layer {quad}"
        );
        assert!(app.world().get::<OutlineParts>(view).is_none());
    }

    /// The quad's corners, carried through `clip_from_view`, land just past
    /// the view's edges on every side: for a perspective, for an orthographic
    /// projection, and for a view framed off-centre by a sub-view.
    #[test]
    fn the_quad_covers_the_view_under_any_projection() {
        let mut perspective = PerspectiveProjection::default();
        perspective.update(1280.0, 800.0);
        let mut orthographic = OrthographicProjection::default_3d();
        orthographic.update(1280.0, 800.0);
        let beside_a_panel = SubCameraView {
            full_size: UVec2::new(900, 800),
            offset: Vec2::new(0.0, 0.0),
            size: UVec2::new(1280, 800),
        };
        for (name, clip_from_view) in [
            ("perspective", perspective.get_clip_from_view()),
            ("orthographic", orthographic.get_clip_from_view()),
            (
                "perspective, off-centre",
                perspective.get_clip_from_view_for_sub(&beside_a_panel),
            ),
            (
                "orthographic, off-centre",
                orthographic.get_clip_from_view_for_sub(&beside_a_panel),
            ),
        ] {
            let quad = overlay_over(clip_from_view);
            for (x, y) in [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)] {
                let corner =
                    clip_from_view.project_point3(quad.transform_point(Vec3::new(x, y, 0.0)));
                let wanted = Vec2::new(x, y) * 2.0 * 1.02;
                assert!(
                    corner.truncate().abs_diff_eq(wanted, 1e-3),
                    "{name}: corner {x},{y} lands at {corner:?}"
                );
            }
        }
    }

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
            ids.iter().map(|id| Selection::of(*id)).collect(),
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

    /// A highlight that sets only a colour keeps the view's width and fill:
    /// what a selection leaves unset is the view's, not a default of its own.
    #[test]
    fn a_highlight_overrides_only_what_it_sets() {
        let view = OutlineView {
            colour: Color::WHITE,
            width: 3.0,
            fill: 0.2,
        };
        let red = Color::srgb(1.0, 0.0, 0.0);
        let shown = uniform(
            &view,
            Some(Highlight {
                colour: Some(red),
                ..default()
            }),
            2.0,
        );
        assert_eq!(shown.colour, LinearRgba::from(red).to_vec4());
        assert_eq!(
            shown.width, 6.0,
            "the view's width, scaled to physical pixels"
        );
        assert_eq!(shown.fill, 0.2);

        let own = uniform(&view, None, 1.0);
        assert_eq!(own.colour, LinearRgba::WHITE.to_vec4());
        assert_eq!(own.width, 3.0);
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
