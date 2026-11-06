//! Ghosting: drawing everything beneath an entity as a ghost of itself, as
//! Studio draws the instance showing a robot's measured state beside the one
//! it is commanded to.
//!
//! **A ghost is a layer over the element's own material, not a replacement
//! for it.** [`Ghost`] on an entity draws every mesh beneath it — including
//! ones that load later — with a [`GhostMaterial`], and keeps the mesh's own
//! `StandardMaterial` as [`GhostOf`]. That material stays the source of truth:
//! its colour and image are copied into the ghost as they change, so a colour
//! written to the element still tints its ghost, and removing [`Ghost`]
//! restores it exactly. Anything writing an element's material writes that
//! asset, which a ghosted mesh still holds in [`GhostOf`].
//!
//! **The look is Studio's** (`materials/ghost.tsx`, see `ghost.wesl`): the
//! element's colour washed halfway toward white and unlit, nearly clear facing
//! the camera and near-opaque at the silhouette, an image at one opacity, and
//! drawn over everything without testing depth, so the robot it stands beside
//! never hides it.
//!
//! **How strongly a ghost shows is the host's** ([`GhostLook`]): the colour it
//! is washed toward, white by default as Studio has it, and how opaque its
//! rim is, Studio's by default. A wash reads only against a ground unlike it,
//! so a host drawing on a light ground washes toward a dark colour instead,
//! and may want the rim stronger there to hold its outline.
//!
//! **A ghost casts no shadow.** It is the trace of where something is, not a
//! second solid object standing in the light.
//!
//! What else changes on a ghosted instance — Studio hides its joint controls,
//! for instance — is the host's to decide.

use bevy::asset::{embedded_asset, embedded_path, AssetPath};
use bevy::light::NotShadowCaster;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, CompareFunction, RenderPipelineDescriptor, ShaderType,
    SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

/// Draw everything beneath this entity as a ghost ([module docs](self)).
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Ghost;

/// A ghosted mesh's own material, kept while it is drawn as a ghost.
#[derive(Component, Clone, Debug)]
pub struct GhostOf {
    pub original: Handle<StandardMaterial>,
    /// Whether the mesh cast no shadow before it was ghosted, so restoring it
    /// does not make it start.
    had_no_shadow: bool,
}

/// How far a ghost's colour is from its wash toward the element's: Studio's.
pub const GHOST_TINT: f32 = 0.5;

/// How every ghost shows ([module docs](self)).
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct GhostLook {
    /// The colour a ghost is washed toward: white, as Studio has it.
    pub wash: Color,
    /// How opaque a ghost's rim is, against Studio's: one is Studio's, two
    /// twice as opaque, up to fully so.
    pub rim: f32,
}

impl Default for GhostLook {
    fn default() -> Self {
        Self {
            wash: Color::WHITE,
            rim: 1.0,
        }
    }
}

#[derive(ShaderType, Clone, Debug, Default)]
pub struct GhostUniform {
    pub colour: Vec4,
    pub wash: Vec4,
    pub tint: f32,
    pub power: f32,
    pub coefficient: f32,
    pub image: f32,
    pub rim: f32,
    pub _padding: Vec3,
}

/// A ghost's material; see `ghost.wesl`.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug, Default)]
pub struct GhostMaterial {
    #[uniform(0)]
    pub ghost: GhostUniform,
    #[texture(1)]
    #[sampler(2)]
    pub image: Option<Handle<Image>>,
}

impl GhostMaterial {
    /// The ghost of a material: its colour and image, shown as `look` says, in
    /// Studio's ghost look.
    pub fn of(material: &StandardMaterial, look: &GhostLook) -> Self {
        Self {
            ghost: GhostUniform {
                colour: LinearRgba::from(material.base_color).to_vec4(),
                wash: LinearRgba::from(look.wash).to_vec4(),
                rim: look.rim,
                tint: GHOST_TINT,
                power: 8.0,
                coefficient: 1.2,
                image: if material.base_color_texture.is_some() {
                    1.0
                } else {
                    0.0
                },
                ..default()
            },
            image: material.base_color_texture.clone(),
        }
    }
}

impl Material for GhostMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(embedded_path!("ghost.wesl")).with_source("embedded"),
        )
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    /// Over everything, as Studio draws a ghost: it stands where the solid
    /// robot does, and depth would hide it inside that robot.
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

pub struct GhostPlugin;

impl Plugin for GhostPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "ghost.wesl");
        app.add_plugins(MaterialPlugin::<GhostMaterial>::default())
            .init_resource::<GhostLook>()
            .add_systems(Update, (ghost_meshes, restore_meshes, follow_originals));
    }
}

/// Ghosts every mesh beneath a [`Ghost`] that is not yet ghosted.
fn ghost_meshes(
    mut commands: Commands,
    roots: Query<Entity, With<Ghost>>,
    children: Query<&Children>,
    meshes: Query<(&MeshMaterial3d<StandardMaterial>, Has<NotShadowCaster>), Without<GhostOf>>,
    standard: Res<Assets<StandardMaterial>>,
    look: Res<GhostLook>,
    mut ghosts: ResMut<Assets<GhostMaterial>>,
) {
    for root in &roots {
        for entity in std::iter::once(root).chain(children.iter_descendants(root)) {
            let Ok((material, had_no_shadow)) = meshes.get(entity) else {
                continue;
            };
            let Some(original) = standard.get(&material.0) else {
                continue;
            };
            let ghost = ghosts.add(GhostMaterial::of(original, &look));
            commands
                .entity(entity)
                .insert((
                    GhostOf {
                        original: material.0.clone(),
                        had_no_shadow,
                    },
                    MeshMaterial3d(ghost),
                    NotShadowCaster,
                ))
                .remove::<MeshMaterial3d<StandardMaterial>>();
        }
    }
}

/// Restores every ghosted mesh no longer beneath a [`Ghost`].
fn restore_meshes(
    mut commands: Commands,
    ghosted: Query<(Entity, &GhostOf)>,
    roots: Query<(), With<Ghost>>,
    parents: Query<&ChildOf>,
) {
    for (entity, of) in &ghosted {
        let still = std::iter::once(entity)
            .chain(parents.iter_ancestors(entity))
            .any(|ancestor| roots.contains(ancestor));
        if still {
            continue;
        }
        let mut mesh = commands.entity(entity);
        mesh.insert(MeshMaterial3d(of.original.clone()))
            .remove::<(GhostOf, MeshMaterial3d<GhostMaterial>)>();
        if !of.had_no_shadow {
            mesh.remove::<NotShadowCaster>();
        }
    }
}

/// Copies each ghosted mesh's own colour and image, and the look, into its
/// ghost as they change, reading before writing so an unchanged ghost is not
/// re-uploaded.
fn follow_originals(
    ghosted: Query<(&GhostOf, &MeshMaterial3d<GhostMaterial>)>,
    standard: Res<Assets<StandardMaterial>>,
    look: Res<GhostLook>,
    mut ghosts: ResMut<Assets<GhostMaterial>>,
) {
    for (of, handle) in &ghosted {
        let Some(original) = standard.get(&of.original) else {
            continue;
        };
        let wanted = GhostMaterial::of(original, &look);
        let stale = ghosts.get(&handle.0).is_some_and(|ghost| {
            ghost.ghost.colour != wanted.ghost.colour
                || ghost.ghost.wash != wanted.ghost.wash
                || ghost.ghost.rim != wanted.ghost.rim
                || ghost.image != wanted.image
        });
        if stale {
            if let Some(mut ghost) = ghosts.get_mut(&handle.0) {
                ghost.ghost = wanted.ghost;
                ghost.image = wanted.image;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ghost takes its element's colour and image, in Studio's tint and
    /// falloff.
    #[test]
    fn a_ghost_takes_its_elements_colour_and_image() {
        let plain = GhostMaterial::of(
            &StandardMaterial {
                base_color: Color::srgb(1.0, 0.0, 0.0),
                ..default()
            },
            &GhostLook {
                wash: Color::BLACK,
                rim: 2.0,
            },
        );
        assert_eq!(plain.ghost.colour, LinearRgba::RED.to_vec4());
        assert_eq!(plain.ghost.wash, LinearRgba::BLACK.to_vec4());
        assert_eq!(plain.ghost.rim, 2.0);
        assert_eq!(
            (plain.ghost.tint, plain.ghost.power, plain.ghost.coefficient),
            (0.5, 8.0, 1.2)
        );
        assert_eq!(plain.ghost.image, 0.0);

        let image = Handle::<Image>::default();
        let pictured = GhostMaterial::of(
            &StandardMaterial {
                base_color_texture: Some(image.clone()),
                ..default()
            },
            &GhostLook::default(),
        );
        assert_eq!(pictured.ghost.image, 1.0);
        assert_eq!(pictured.image, Some(image));
    }

    /// Ghosting swaps a mesh's material for its ghost and keeps the original;
    /// removing the ghost puts the original back, shadow and all.
    #[test]
    fn a_ghost_is_undone_exactly() {
        let mut app = App::new();
        app.init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<GhostMaterial>>()
            .init_resource::<GhostLook>()
            .add_systems(Update, (ghost_meshes, restore_meshes).chain());
        let original = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        let root = app.world_mut().spawn(Ghost).id();
        let mesh = app
            .world_mut()
            .spawn((MeshMaterial3d(original.clone()), ChildOf(root)))
            .id();

        app.update();
        let world = app.world();
        assert!(world
            .get::<MeshMaterial3d<StandardMaterial>>(mesh)
            .is_none());
        assert!(world.get::<MeshMaterial3d<GhostMaterial>>(mesh).is_some());
        assert!(world.get::<NotShadowCaster>(mesh).is_some());

        app.world_mut().entity_mut(root).remove::<Ghost>();
        app.update();
        let world = app.world();
        assert_eq!(
            world
                .get::<MeshMaterial3d<StandardMaterial>>(mesh)
                .unwrap()
                .0,
            original
        );
        assert!(world.get::<MeshMaterial3d<GhostMaterial>>(mesh).is_none());
        assert!(world.get::<GhostOf>(mesh).is_none());
        assert!(
            world.get::<NotShadowCaster>(mesh).is_none(),
            "it cast a shadow before"
        );
    }
}
