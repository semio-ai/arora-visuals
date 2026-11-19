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
//! The mask camera finds the selection through its view's `mask` layer
//! ([`ViewLayers`]), which a selected mesh joins and leaves as the selection
//! changes. A mesh joins the mask of each outlined view that draws it, and
//! no other, so a view outlines only what it shows even when another view's
//! content stands in the same place. Whatever layers the mesh had before are
//! kept, so it stays visible everywhere it was.
//!
//! A view's outline takes its colour, width and fill from the selection it
//! shows: a highlight one of those selected elements brings applies to that
//! view alone.
//!
//! [`glow`](crate::glow) marks a selection differently — with its feature
//! edges — and the two read the same selections.

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
use crate::select::{covered, highlight_of, Highlight, Selection, Selections};
use crate::view_layer::{
    content_layers, ViewLayerPlugin, ViewLayerSystems, ViewLayers, WantsViewLayers,
};

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

/// What an outlined view owns: its mask camera, the mask and the quad, which
/// is on the view's `aids` layer so that no other camera draws it. And what
/// the view's selection currently is: whether it holds anything, and the
/// highlight it asks for.
#[derive(Component)]
struct OutlineParts {
    mask_camera: Entity,
    overlay: Entity,
    mask: Handle<Image>,
    material: Handle<OutlineMaterial>,
    selecting: bool,
    highlight: Option<Highlight>,
}

/// On a mesh currently in some view's mask: the layers it had before it
/// joined.
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
        app.add_plugins((
            MaterialPlugin::<OutlineMaterial>::default(),
            ViewLayerPlugin,
        ))
        .init_resource::<Selections>()
        .add_systems(
            Update,
            (
                ask_for_layers,
                detach_outline,
                attach_outline,
                mark_selection,
            )
                .chain(),
        )
        // The mask follows the view's framing before Bevy computes either
        // camera's projection; the quad is fitted to the projection Bevy
        // then computed, sub-view and all.
        .add_systems(
            PostUpdate,
            (
                sync_mask
                    .after(FramingSystems)
                    .after(ViewLayerSystems)
                    .before(CameraUpdateSystems),
                fit_overlay
                    .after(CameraUpdateSystems)
                    .after(TransformSystems::Propagate),
            ),
        );
    }
}

/// Asks for the layers of each outlined view that has none yet.
#[allow(clippy::type_complexity)]
fn ask_for_layers(
    mut commands: Commands,
    views: Query<
        Entity,
        (
            With<OutlineView>,
            Without<ViewLayers>,
            Without<WantsViewLayers>,
        ),
    >,
) {
    for view in &views {
        commands.entity(view).insert(WantsViewLayers);
    }
}

/// Gives each newly outlined view its mask camera, mask and overlay, once
/// the view has its layers.
#[allow(clippy::type_complexity)]
fn attach_outline(
    mut commands: Commands,
    views: Query<(Entity, &Camera, &OutlineView, &ViewLayers), Without<OutlineParts>>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<OutlineMaterial>>,
) {
    for (view, camera, settings, given) in &views {
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
                RenderLayers::layer(given.mask),
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
                RenderLayers::layer(given.aids),
                NotShadowCaster,
                NotShadowReceiver,
                ChildOf(view),
                OutlineOverlay,
                // It covers the whole view just past the near plane; a mesh
                // picking backend that could pick it would take every press
                // off the scene behind it.
                Pickable::IGNORE,
            ))
            .id();
        commands.entity(view).insert(OutlineParts {
            mask_camera,
            overlay,
            mask,
            material,
            selecting: false,
            highlight: None,
        });
    }
}

/// Takes an outline away from a view no longer asking for one: its mask
/// camera and quad go. The view keeps its layers, which anything else drawn
/// for it may still be on.
fn detach_outline(
    mut commands: Commands,
    views: Query<(Entity, &OutlineParts), Without<OutlineView>>,
) {
    for (view, parts) in &views {
        commands.entity(parts.mask_camera).despawn();
        commands.entity(parts.overlay).despawn();
        commands.entity(view).remove::<OutlineParts>();
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

/// Moves each selected mesh into the mask of every outlined view that draws
/// it, and everything else out of the masks; and records, per view, whether
/// it has anything to outline and the highlight its selection asks for.
fn mark_selection(
    mut commands: Commands,
    selections: Res<Selections>,
    instances: Query<&EntityInstance, Without<Superseded>>,
    parents: Query<&ChildOf>,
    mut views: Query<(&ViewLayers, Option<&RenderLayers>, &mut OutlineParts)>,
    layers: Query<(Option<&RenderLayers>, Option<&Outlined>)>,
    outlined: Query<(Entity, &Outlined)>,
) {
    // Each selected mesh, with what it was on before any mask, and the
    // selection it is selected through.
    let mut selected: Vec<(Entity, Option<RenderLayers>, &Selection)> = Vec::new();
    for instance in &instances {
        if !instance.bindings.indexed {
            continue;
        }
        let Some(chosen) = selections.0.get(&instance.id) else {
            continue;
        };
        for selection in chosen {
            for mesh in covered(instance, &selection.element, &parents)
                .into_iter()
                .filter_map(|element| element.mesh)
            {
                let Ok((now, marked)) = layers.get(mesh) else {
                    continue;
                };
                let own = match marked {
                    Some(marked) => marked.before.clone(),
                    None => now.cloned(),
                };
                selected.push((mesh, own, selection));
            }
        }
    }

    let views: Vec<_> = views
        .iter_mut()
        .map(|(given, layers, parts)| (*given, content_layers(layers, Some(given)), parts))
        .collect();
    let mut wanted: Vec<(Entity, Option<RenderLayers>, RenderLayers)> = Vec::new();
    let mut shown: Vec<Vec<&Selection>> = vec![Vec::new(); views.len()];
    for (mesh, own, selection) in &selected {
        let drawn_on = own.clone().unwrap_or_default();
        let mut joined = drawn_on.clone();
        for (index, (given, content, _)) in views.iter().enumerate() {
            if drawn_on.intersects(content) {
                joined = joined.with(given.mask);
                if !shown[index]
                    .iter()
                    .any(|held| std::ptr::eq(*held, *selection))
                {
                    shown[index].push(selection);
                }
            }
        }
        if joined != drawn_on && !wanted.iter().any(|(held, ..)| held == mesh) {
            wanted.push((*mesh, own.clone(), joined));
        }
    }
    for ((_, _, mut parts), shown) in views.into_iter().zip(shown) {
        let selecting = !shown.is_empty();
        let highlight = highlight_of(shown);
        if parts.selecting != selecting {
            parts.selecting = selecting;
        }
        if parts.highlight != highlight {
            parts.highlight = highlight;
        }
    }

    for (mesh, marked) in &outlined {
        if wanted.iter().any(|(held, ..)| *held == mesh) {
            continue;
        }
        let mut entity = commands.entity(mesh);
        match &marked.before {
            Some(before) => entity.insert(before.clone()),
            None => entity.remove::<RenderLayers>(),
        };
        entity.remove::<Outlined>();
    }
    for (mesh, before, joined) in wanted {
        let Ok((now, marked)) = layers.get(mesh) else {
            continue;
        };
        if marked.is_some() && now == Some(&joined) {
            continue;
        }
        commands.entity(mesh).insert((joined, Outlined { before }));
    }
}

/// Keeps the mask the size of the view and the mask camera looking through
/// the view's lens — its projection and its sub-view, so a view framed beside
/// a panel masks what it draws.
#[allow(clippy::type_complexity)]
fn sync_mask(
    views: Query<(&Camera, &Projection, &OutlineView, &OutlineParts)>,
    mut cameras: Query<
        (&mut Camera, &mut Projection, &mut RenderTarget),
        (With<OutlineMask>, Without<OutlineParts>),
    >,
    mut overlays: Query<&mut Visibility, With<OutlineOverlay>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<OutlineMaterial>>,
) {
    for (camera, projection, settings, parts) in &views {
        // With nothing of its own selected a view has nothing to outline, and
        // the mask pass and the quad's dozens of samples a pixel would be
        // paid for nothing; both stand down until a selection appears.
        let selecting = parts.selecting;
        if let Ok(mut visibility) = overlays.get_mut(parts.overlay) {
            visibility.set_if_neq(if selecting {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            });
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
            parts.highlight,
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
        global.set_if_neq(at.mul_transform(wanted));
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
        app.add_plugins((MinimalPlugins, AssetPlugin::default(), ViewLayerPlugin))
            .init_asset::<Image>()
            .init_asset::<Mesh>()
            .init_asset::<OutlineMaterial>()
            .add_systems(
                Update,
                (ask_for_layers, detach_outline, attach_outline).chain(),
            );
        app
    }

    /// An outlined view's quad is on its `aids` layer and its mask camera
    /// draws its `mask` layer, so neither reaches another view.
    #[test]
    fn an_outlined_views_parts_are_on_its_own_layers() {
        let mut app = outline_app();
        let view = app
            .world_mut()
            .spawn((Camera::default(), OutlineView::default()))
            .id();
        // Layers are given in one frame and the outline attached in the next.
        app.update();
        app.update();
        let given = *app.world().get::<ViewLayers>(view).expect("layers");
        let parts = app.world().get::<OutlineParts>(view).expect("outlined");
        let (overlay, mask_camera) = (parts.overlay, parts.mask_camera);
        assert_eq!(
            app.world().get::<RenderLayers>(overlay),
            Some(&RenderLayers::layer(given.aids)),
        );
        assert_eq!(
            app.world().get::<RenderLayers>(mask_camera),
            Some(&RenderLayers::layer(given.mask)),
        );
    }

    /// Taking an outline away leaves the view the layers it has, the host's
    /// included, and the parts go.
    #[test]
    fn removing_an_outline_keeps_the_views_layers() {
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
        app.update();
        let overlay = app
            .world()
            .get::<OutlineParts>(view)
            .expect("outlined")
            .overlay;
        let given = app
            .world()
            .get::<RenderLayers>(view)
            .cloned()
            .unwrap()
            .with(5);
        app.world_mut().entity_mut(view).insert(given.clone());
        app.world_mut().entity_mut(view).remove::<OutlineView>();
        app.update();
        assert_eq!(app.world().get::<RenderLayers>(view), Some(&given));
        assert!(app.world().get::<OutlineParts>(view).is_none());
        assert!(app.world().get_entity(overlay).is_err(), "the quad went");
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

    /// The layers of the two views in [`world`]: one drawing the default
    /// layer, one drawing layer 2.
    const ON_DEFAULT: ViewLayers = ViewLayers { aids: 31, mask: 32 };
    const ON_TWO: ViewLayers = ViewLayers { aids: 33, mask: 34 };

    fn parts() -> OutlineParts {
        OutlineParts {
            mask_camera: Entity::PLACEHOLDER,
            overlay: Entity::PLACEHOLDER,
            mask: Handle::default(),
            material: Handle::default(),
            selecting: false,
            highlight: None,
        }
    }

    /// A world with one indexed instance of two elements, each with a mesh:
    /// one on layer 2, one on no layer at all. And two outlined views, one
    /// drawing the default layer and one drawing layer 2, as two consumers
    /// sharing an app would, their content standing in the same place.
    fn world() -> (World, Entity, Entity, [Entity; 2]) {
        let mut world = World::new();
        world.init_resource::<Selections>();
        let layered = world.spawn(RenderLayers::layer(2)).id();
        let bare = world.spawn_empty().id();
        let camera = world.spawn_empty().id();
        let views = [
            world
                .spawn((
                    ON_DEFAULT,
                    RenderLayers::layer(0).with(ON_DEFAULT.aids),
                    parts(),
                ))
                .id(),
            world
                .spawn((ON_TWO, RenderLayers::layer(2).with(ON_TWO.aids), parts()))
                .id(),
        ];

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
        (world, layered, bare, views)
    }

    fn select(world: &mut World, chosen: Vec<Selection>) {
        world
            .resource_mut::<Selections>()
            .0
            .insert("robot".to_string(), chosen);
        world.run_system_once(mark_selection).unwrap();
    }

    fn of(ids: &[&str]) -> Vec<Selection> {
        ids.iter().map(|id| Selection::of(*id)).collect()
    }

    /// A selected mesh joins the mask of the view that draws it and no
    /// other, and leaving puts back exactly what was there — including
    /// nothing. A mesh left on the wrong layers either vanishes from views it
    /// belonged to, stays outlined after it is deselected, or is outlined in
    /// a view that does not show it, and none of those shows as an error.
    #[test]
    fn a_selected_mesh_joins_only_the_masks_of_the_views_that_draw_it() {
        let (mut world, layered, bare, _) = world();

        select(&mut world, of(&["layered", "bare"]));
        assert_eq!(
            world.get::<RenderLayers>(layered),
            Some(&RenderLayers::layer(2).with(ON_TWO.mask)),
        );
        assert_eq!(
            world.get::<RenderLayers>(bare),
            Some(&RenderLayers::default().with(ON_DEFAULT.mask)),
            "a mesh on no layer is on the default one, and keeps it",
        );

        select(&mut world, of(&["bare"]));
        assert_eq!(
            world.get::<RenderLayers>(layered),
            Some(&RenderLayers::layer(2)),
            "deselected, it is back on its own layer alone",
        );
        assert!(world.get::<Outlined>(layered).is_none());
        assert!(world.get::<Outlined>(bare).is_some(), "still selected");

        select(&mut world, of(&[]));
        assert!(
            world.get::<RenderLayers>(bare).is_none(),
            "a mesh that had no layers has none again",
        );
    }

    /// Each view outlines what it shows, with the highlight of what it shows:
    /// a highlight brought by an element one view draws does not recolour
    /// another view's outline.
    #[test]
    fn each_view_takes_the_highlight_of_its_own_selection() {
        let (mut world, _, _, [on_default, on_two]) = world();
        let red = Highlight {
            colour: Some(Color::srgb(1.0, 0.0, 0.0)),
            ..default()
        };
        select(
            &mut world,
            vec![
                Selection {
                    element: "layered".into(),
                    highlight: Some(red),
                },
                Selection::of("bare"),
            ],
        );
        let shown = |world: &World, view| {
            let parts = world.get::<OutlineParts>(view).unwrap();
            (parts.selecting, parts.highlight)
        };
        assert_eq!(shown(&world, on_two), (true, Some(red)));
        assert_eq!(shown(&world, on_default), (true, None));

        select(&mut world, of(&["layered"]));
        assert_eq!(
            shown(&world, on_default),
            (false, None),
            "a view with none of the selection has nothing to outline",
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
        let (mut world, layered, _, _) = world();
        select(&mut world, of(&["layered"]));
        let before = world.get::<RenderLayers>(layered).cloned();
        let tick = world.change_tick();
        select(&mut world, of(&["layered"]));
        assert_eq!(world.get::<RenderLayers>(layered).cloned(), before);
        assert!(world.get::<Outlined>(layered).is_some());
        assert!(
            !world
                .entity(layered)
                .get_ref::<RenderLayers>()
                .unwrap()
                .last_changed()
                .is_newer_than(tick, world.change_tick()),
            "the mesh's layers were not written again",
        );
    }
}
