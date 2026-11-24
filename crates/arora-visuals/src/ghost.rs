//! Ghosting: drawing everything beneath an entity as a ghost of itself, as
//! Studio draws the instance showing a robot's measured state beside the one
//! it is commanded to.
//!
//! **A ghost is a layer over the element's own material, not a replacement
//! for it.** [`Ghost`] on an entity draws every mesh beneath it — including
//! ones that load later — with a [`GhostMaterial`], and keeps the mesh's own
//! `StandardMaterial` as [`GhostOf`]. That material stays the source of truth:
//! its colour, opacity and image are copied into the ghost as they change, so
//! a colour written to the element still tints its ghost, and removing
//! [`Ghost`] restores it exactly. A writer that swaps a mesh's material
//! reaches it through [`SolidMaterial`], which finds it whether the mesh is
//! drawn solid or as a ghost; a material inserted on a ghosted mesh some
//! other way is taken as its new original rather than drawn beside the ghost.
//!
//! **The look is Studio's** (`materials/ghost.tsx`, see `ghost.wesl`): the
//! element's colour washed halfway toward white and unlit, nearly clear facing
//! the camera and near-opaque at the silhouette, an image at one opacity, and
//! drawn over everything without testing depth, so the robot it stands beside
//! never hides it.
//!
//! **How strongly a ghost shows is the host's.** [`GhostLook`] sets it for
//! every ghost: the colour a ghost is washed toward, white by default as
//! Studio has it, and how opaque its rim is, Studio's by default. A wash reads
//! only against a ground unlike it, so a host drawing on a light ground washes
//! toward a dark colour instead, and may want the rim stronger there to hold
//! its outline. [`GhostStyle`] beside a [`Ghost`] changes one ghost: its
//! colour, wash, rim, opacity, and whether it tests depth — so one ghost can
//! be faded or recoloured over time, and a trace of many ghosts can hide
//! behind the scene and one another instead of all showing at once.
//!
//! **Ghosts share their materials.** Every mesh showing the same original
//! material under the same look shares one [`GhostMaterial`]: one per
//! original for ghosts without a [`GhostStyle`], one per original and styled
//! ghost for those with one. A change to an original, a style or the look
//! updates those shared materials in place, so animating a ghost re-uploads
//! one uniform per material it shows rather than making new ones.
//!
//! **A ghost casts no shadow.** It is the trace of where something is, not a
//! second solid object standing in the light.
//!
//! What else changes on a ghosted instance — Studio hides its joint controls,
//! for instance — is the host's to decide.

use std::collections::HashMap;

use bevy::asset::{embedded_asset, embedded_path, AssetPath};
use bevy::ecs::query::QueryData;
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

/// How one [`Ghost`] shows, beside it on the same entity; what it leaves
/// unset comes from [`GhostLook`]. Changing it changes the ghost from the next
/// frame, so a host animates a ghost by writing it.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct GhostStyle {
    /// The colour every element is drawn in, in place of its own. Its alpha
    /// is the elements' opacity, as their own colour's would be.
    pub colour: Option<Color>,
    /// The colour the ghost is washed toward, in place of the look's.
    pub wash: Option<Color>,
    /// How opaque the rim is against Studio's, in place of the look's.
    pub rim: Option<f32>,
    /// How opaque the whole ghost is, from nought to one: what a host fades
    /// it by.
    pub opacity: f32,
    /// Whether the ghost hides behind what stands in front of it.
    pub depth: GhostDepth,
}

impl Default for GhostStyle {
    fn default() -> Self {
        Self {
            colour: None,
            wash: None,
            rim: None,
            opacity: 1.0,
            depth: GhostDepth::default(),
        }
    }
}

/// Whether a ghost tests depth.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GhostDepth {
    /// Drawn over everything, as Studio draws a ghost: it stands where the
    /// solid robot does, and depth would hide it inside that robot.
    #[default]
    OverEverything,
    /// Hidden by what stands in front of it, other ghosts of this kind
    /// included, as the poses of a trace want to be.
    Tested,
}

/// A ghosted mesh's own material, kept while it is drawn as a ghost.
///
/// `original` is the material the mesh shows when solid. Assigning another
/// to it — a writer's own copy, say — re-derives the ghost from that one.
#[derive(Component, Clone, Debug)]
pub struct GhostOf {
    pub original: Handle<StandardMaterial>,
    /// Whether the mesh cast no shadow before it was ghosted, so restoring it
    /// does not make it start.
    had_no_shadow: bool,
    /// The [`Ghost`] it is drawn under, whose [`GhostStyle`] it takes.
    root: Entity,
    /// Which shared ghost material it shows.
    key: GhostKey,
}

/// The material a mesh shows when solid, whether it is drawn solid now or as
/// a ghost: what a writer of a mesh's material reads and swaps, so a write to
/// a ghosted mesh reaches the material its ghost follows and the mesh shows
/// again once the ghost goes.
#[derive(QueryData)]
#[query_data(mutable)]
pub struct SolidMaterial {
    solid: Option<&'static mut MeshMaterial3d<StandardMaterial>>,
    ghost: Option<&'static mut GhostOf>,
}

impl SolidMaterialItem<'_, '_> {
    /// The mesh's own material, if it has one.
    pub fn handle(&self) -> Option<&Handle<StandardMaterial>> {
        match (&self.solid, &self.ghost) {
            (Some(solid), _) => Some(&solid.0),
            (None, Some(ghost)) => Some(&ghost.original),
            (None, None) => None,
        }
    }

    /// Gives the mesh `material` as its own, solid or ghosted.
    pub fn set(&mut self, material: Handle<StandardMaterial>) {
        if let Some(solid) = self.solid.as_mut() {
            solid.0 = material;
        } else if let Some(ghost) = self.ghost.as_mut() {
            ghost.original = material;
        }
    }
}

impl SolidMaterialReadOnlyItem<'_, '_> {
    /// The mesh's own material, if it has one.
    pub fn handle(&self) -> Option<&Handle<StandardMaterial>> {
        match (self.solid, self.ghost) {
            (Some(solid), _) => Some(&solid.0),
            (None, Some(ghost)) => Some(&ghost.original),
            (None, None) => None,
        }
    }
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

#[derive(ShaderType, Clone, Debug, Default, PartialEq)]
pub struct GhostUniform {
    pub colour: Vec4,
    pub wash: Vec4,
    pub tint: f32,
    pub power: f32,
    pub coefficient: f32,
    pub image: f32,
    pub rim: f32,
    pub opacity: f32,
    pub _padding: Vec2,
}

/// A ghost's material; see `ghost.wesl`.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug, Default)]
#[bind_group_data(GhostMaterialKey)]
pub struct GhostMaterial {
    #[uniform(0)]
    pub ghost: GhostUniform,
    #[texture(1)]
    #[sampler(2)]
    pub image: Option<Handle<Image>>,
    /// Whether it tests depth: a pipeline's, not a uniform's, to choose.
    pub depth: GhostDepth,
}

/// What a ghost material's pipeline depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GhostMaterialKey {
    depth: GhostDepth,
}

impl From<&GhostMaterial> for GhostMaterialKey {
    fn from(material: &GhostMaterial) -> Self {
        Self {
            depth: material.depth,
        }
    }
}

impl GhostMaterial {
    /// The ghost of a material: its colour, opacity and image, shown as
    /// `look` says, in Studio's ghost look.
    pub fn of(material: &StandardMaterial, look: &GhostLook) -> Self {
        Self::styled(material, look, None)
    }

    /// The ghost of a material as one ghost's `style` changes `look`.
    pub fn styled(
        material: &StandardMaterial,
        look: &GhostLook,
        style: Option<&GhostStyle>,
    ) -> Self {
        let colour = style
            .and_then(|style| style.colour)
            .unwrap_or(material.base_color);
        Self {
            ghost: GhostUniform {
                colour: LinearRgba::from(colour).to_vec4(),
                wash: LinearRgba::from(style.and_then(|s| s.wash).unwrap_or(look.wash)).to_vec4(),
                rim: style.and_then(|s| s.rim).unwrap_or(look.rim),
                opacity: style.map_or(1.0, |s| s.opacity.clamp(0.0, 1.0)),
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
            depth: style.map_or(GhostDepth::default(), |s| s.depth),
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

    /// Over everything unless the ghost tests depth ([`GhostDepth`]). A
    /// ghost that tests depth also writes it, so of the poses of a trace the
    /// nearer hide the farther where they overlap rather than all showing
    /// through one another.
    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            match key.bind_group_data.depth {
                GhostDepth::OverEverything => {
                    depth.depth_compare = Some(CompareFunction::Always);
                    depth.depth_write_enabled = Some(false);
                }
                GhostDepth::Tested => {
                    depth.depth_write_enabled = Some(true);
                }
            }
        }
        Ok(())
    }
}

/// Which shared ghost material a ghosted mesh shows: one per original
/// material, and per styled ghost for a ghost with a [`GhostStyle`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GhostKey {
    original: AssetId<StandardMaterial>,
    styled_by: Option<Entity>,
}

/// The ghost materials in use, by what they are the ghost of. Holding their
/// handles keeps a material alive between the meshes that share it; one no
/// mesh shows any more is dropped.
#[derive(Resource, Default)]
pub struct GhostMaterials {
    by_key: HashMap<GhostKey, Handle<GhostMaterial>>,
}

impl GhostMaterials {
    /// How many ghost materials are held.
    pub fn len(&self) -> usize {
        self.by_key.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }
}

pub struct GhostPlugin;

impl Plugin for GhostPlugin {
    /// Both consumers sharing an app may add it; it installs once.
    fn is_unique(&self) -> bool {
        false
    }

    fn build(&self, app: &mut App) {
        if !crate::installed::first_install::<Self>(app) {
            return;
        }
        embedded_asset!(app, "ghost.wesl");
        app.add_plugins(MaterialPlugin::<GhostMaterial>::default())
            .init_resource::<GhostLook>()
            .init_resource::<GhostMaterials>()
            .add_systems(Update, ghost_systems());
    }
}

/// Ghosting's systems, in order: meshes newly beneath a ghost are ghosted, a
/// material given to a ghosted mesh becomes its original, each mesh is moved
/// to the shared material its original and style now call for, shared
/// materials follow their originals, unshown ones are dropped, and meshes no
/// longer beneath a ghost are restored.
fn ghost_systems() -> bevy::ecs::schedule::ScheduleConfigs<bevy::ecs::system::ScheduleSystem> {
    (
        ghost_meshes,
        adopt_new_originals,
        rekey,
        follow_originals,
        restore_meshes,
        drop_unshown,
    )
        .chain()
}

/// The shared ghost material for `key`, made from `original` if there is
/// none yet.
fn shared(
    materials: &mut GhostMaterials,
    ghosts: &mut Assets<GhostMaterial>,
    key: GhostKey,
    original: &StandardMaterial,
    look: &GhostLook,
    style: Option<&GhostStyle>,
) -> Handle<GhostMaterial> {
    materials
        .by_key
        .entry(key)
        .or_insert_with(|| ghosts.add(GhostMaterial::styled(original, look, style)))
        .clone()
}

/// The key a mesh ghosted under `root` showing `original` takes.
fn key_for(
    original: &Handle<StandardMaterial>,
    root: Entity,
    styles: &Query<(), With<GhostStyle>>,
) -> GhostKey {
    GhostKey {
        original: original.id(),
        styled_by: styles.contains(root).then_some(root),
    }
}

/// Ghosts every mesh beneath a [`Ghost`] that is not yet ghosted.
#[allow(clippy::too_many_arguments)]
fn ghost_meshes(
    mut commands: Commands,
    roots: Query<(Entity, Option<&GhostStyle>), With<Ghost>>,
    has_style: Query<(), With<GhostStyle>>,
    children: Query<&Children>,
    meshes: Query<(&MeshMaterial3d<StandardMaterial>, Has<NotShadowCaster>), Without<GhostOf>>,
    standard: Res<Assets<StandardMaterial>>,
    look: Res<GhostLook>,
    mut materials: ResMut<GhostMaterials>,
    mut ghosts: ResMut<Assets<GhostMaterial>>,
) {
    for (root, style) in &roots {
        for entity in std::iter::once(root).chain(children.iter_descendants(root)) {
            let Ok((material, had_no_shadow)) = meshes.get(entity) else {
                continue;
            };
            let Some(original) = standard.get(&material.0) else {
                continue;
            };
            let key = key_for(&material.0, root, &has_style);
            let ghost = shared(&mut materials, &mut ghosts, key, original, &look, style);
            commands
                .entity(entity)
                .insert((
                    GhostOf {
                        original: material.0.clone(),
                        had_no_shadow,
                        root,
                        key,
                    },
                    MeshMaterial3d(ghost),
                    NotShadowCaster,
                ))
                .remove::<MeshMaterial3d<StandardMaterial>>();
        }
    }
}

/// Takes a material given to an already ghosted mesh as its new original,
/// so the mesh is not drawn twice — solid and as a ghost.
fn adopt_new_originals(
    mut commands: Commands,
    mut given: Query<(Entity, &mut GhostOf, &MeshMaterial3d<StandardMaterial>)>,
) {
    for (entity, mut of, material) in &mut given {
        of.original = material.0.clone();
        commands
            .entity(entity)
            .remove::<MeshMaterial3d<StandardMaterial>>();
    }
}

/// Moves each ghosted mesh to the shared material its original and its
/// ghost's style now call for: an original swapped by a writer, or a style
/// added to or taken from its ghost.
#[allow(clippy::too_many_arguments)]
fn rekey(
    mut ghosted: Query<(&mut GhostOf, &mut MeshMaterial3d<GhostMaterial>)>,
    has_style: Query<(), With<GhostStyle>>,
    styles: Query<&GhostStyle>,
    standard: Res<Assets<StandardMaterial>>,
    look: Res<GhostLook>,
    mut materials: ResMut<GhostMaterials>,
    mut ghosts: ResMut<Assets<GhostMaterial>>,
) {
    for (mut of, mut handle) in &mut ghosted {
        let wanted = key_for(&of.original, of.root, &has_style);
        if wanted == of.key {
            continue;
        }
        let Some(original) = standard.get(&of.original) else {
            continue;
        };
        let style = styles.get(of.root).ok();
        handle.0 = shared(&mut materials, &mut ghosts, wanted, original, &look, style);
        of.key = wanted;
    }
}

/// Copies each original's colour, opacity and image, its ghost's style and
/// the look into the ghost materials made from it, reading before writing so
/// an unchanged one is not re-uploaded. Done per shared material, not per
/// mesh, so a ghost of many meshes costs what its materials do.
fn follow_originals(
    materials: Res<GhostMaterials>,
    standard: Res<Assets<StandardMaterial>>,
    styles: Query<&GhostStyle>,
    look: Res<GhostLook>,
    mut ghosts: ResMut<Assets<GhostMaterial>>,
) {
    for (key, handle) in &materials.by_key {
        let Some(original) = standard.get(key.original) else {
            continue;
        };
        let style = key.styled_by.and_then(|root| styles.get(root).ok());
        let wanted = GhostMaterial::styled(original, &look, style);
        let stale = ghosts.get(handle).is_some_and(|ghost| {
            ghost.ghost != wanted.ghost
                || ghost.image != wanted.image
                || ghost.depth != wanted.depth
        });
        if stale {
            if let Some(mut ghost) = ghosts.get_mut(handle) {
                *ghost = wanted;
            }
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

/// Drops each shared ghost material no ghosted mesh shows, once a mesh has
/// stopped showing one: restored, despawned or moved to another. Checked only
/// then, so a frame in which nothing changed walks nothing.
fn drop_unshown(
    mut gone: RemovedComponents<GhostOf>,
    moved: Query<(), Changed<MeshMaterial3d<GhostMaterial>>>,
    ghosted: Query<&GhostOf>,
    mut materials: ResMut<GhostMaterials>,
) {
    let changed = gone.read().count() > 0 || !moved.is_empty();
    if !changed {
        return;
    }
    let shown: std::collections::HashSet<GhostKey> = ghosted.iter().map(|of| of.key).collect();
    materials.by_key.retain(|key, _| shown.contains(key));
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
        assert_eq!(plain.ghost.opacity, 1.0);
        assert_eq!(plain.depth, GhostDepth::OverEverything);
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

    /// A style overrides what it sets and leaves the rest to the look.
    #[test]
    fn a_style_overrides_what_it_sets() {
        let material = StandardMaterial {
            base_color: Color::srgb(1.0, 0.0, 0.0),
            ..default()
        };
        let look = GhostLook {
            wash: Color::BLACK,
            rim: 2.0,
        };
        let styled = GhostMaterial::styled(
            &material,
            &look,
            Some(&GhostStyle {
                colour: Some(Color::srgba(0.0, 0.0, 1.0, 0.5)),
                rim: Some(3.0),
                opacity: 0.25,
                depth: GhostDepth::Tested,
                ..default()
            }),
        );
        assert_eq!(
            styled.ghost.colour,
            LinearRgba::new(0.0, 0.0, 1.0, 0.5).to_vec4()
        );
        assert_eq!(styled.ghost.wash, LinearRgba::BLACK.to_vec4(), "the look's");
        assert_eq!(styled.ghost.rim, 3.0);
        assert_eq!(styled.ghost.opacity, 0.25);
        assert_eq!(styled.depth, GhostDepth::Tested);
    }

    fn app() -> App {
        let mut app = App::new();
        app.init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<GhostMaterial>>()
            .init_resource::<GhostLook>()
            .init_resource::<GhostMaterials>()
            .add_systems(Update, ghost_systems());
        app
    }

    fn material(app: &mut App, colour: Color) -> Handle<StandardMaterial> {
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                base_color: colour,
                ..default()
            })
    }

    fn ghost_of(app: &App, mesh: Entity) -> GhostMaterial {
        let handle = &app
            .world()
            .get::<MeshMaterial3d<GhostMaterial>>(mesh)
            .expect("ghosted")
            .0;
        app.world()
            .resource::<Assets<GhostMaterial>>()
            .get(handle)
            .unwrap()
            .clone()
    }

    /// Ghosting swaps a mesh's material for its ghost and keeps the original;
    /// removing the ghost puts the original back, shadow and all.
    #[test]
    fn a_ghost_is_undone_exactly() {
        let mut app = app();
        let original = material(&mut app, Color::WHITE);
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

    /// A style written on a ghost changes it the next frame, and only it: a
    /// host fades and recolours one ghost by writing its style.
    #[test]
    fn a_ghost_follows_its_style_as_it_changes() {
        let mut app = app();
        let shared_material = material(&mut app, Color::srgb(1.0, 0.0, 0.0));
        let styled = app.world_mut().spawn((Ghost, GhostStyle::default())).id();
        let plain = app.world_mut().spawn(Ghost).id();
        let a = app
            .world_mut()
            .spawn((MeshMaterial3d(shared_material.clone()), ChildOf(styled)))
            .id();
        let b = app
            .world_mut()
            .spawn((MeshMaterial3d(shared_material.clone()), ChildOf(plain)))
            .id();
        app.update();
        assert_eq!(ghost_of(&app, a).ghost.opacity, 1.0);

        app.world_mut()
            .get_mut::<GhostStyle>(styled)
            .unwrap()
            .opacity = 0.3;
        app.world_mut()
            .get_mut::<GhostStyle>(styled)
            .unwrap()
            .colour = Some(Color::srgb(0.0, 1.0, 0.0));
        app.update();
        let faded = ghost_of(&app, a);
        assert_eq!(faded.ghost.opacity, 0.3);
        assert_eq!(faded.ghost.colour, LinearRgba::GREEN.to_vec4());
        let other = ghost_of(&app, b);
        assert_eq!(other.ghost.opacity, 1.0, "the other ghost is untouched");
        assert_eq!(other.ghost.colour, LinearRgba::RED.to_vec4());

        // A style taken away returns the ghost to the look.
        app.world_mut().entity_mut(styled).remove::<GhostStyle>();
        app.update();
        assert_eq!(ghost_of(&app, a).ghost.opacity, 1.0);
        assert_eq!(ghost_of(&app, a).ghost.colour, LinearRgba::RED.to_vec4());
    }

    /// A material given to a mesh already ghosted becomes what its ghost
    /// follows; the mesh is not drawn solid beside its ghost.
    #[test]
    fn a_material_given_to_a_ghosted_mesh_is_ghosted_too() {
        let mut app = app();
        let first = material(&mut app, Color::srgb(1.0, 0.0, 0.0));
        let root = app.world_mut().spawn(Ghost).id();
        let mesh = app
            .world_mut()
            .spawn((MeshMaterial3d(first), ChildOf(root)))
            .id();
        app.update();

        let second = material(&mut app, Color::srgb(0.0, 0.0, 1.0));
        app.world_mut()
            .entity_mut(mesh)
            .insert(MeshMaterial3d(second.clone()));
        app.update();
        let world = app.world();
        assert!(
            world
                .get::<MeshMaterial3d<StandardMaterial>>(mesh)
                .is_none(),
            "not drawn solid as well"
        );
        assert_eq!(world.get::<GhostOf>(mesh).unwrap().original, second);
        assert_eq!(
            ghost_of(&app, mesh).ghost.colour,
            LinearRgba::BLUE.to_vec4()
        );

        app.world_mut().entity_mut(root).remove::<Ghost>();
        app.update();
        assert_eq!(
            app.world()
                .get::<MeshMaterial3d<StandardMaterial>>(mesh)
                .unwrap()
                .0,
            second,
            "restored to the material it was last given"
        );
    }

    /// A writer that swaps a ghosted mesh's material through
    /// [`SolidMaterial`] reaches the material its ghost follows.
    #[test]
    fn a_material_swapped_through_solid_material_reaches_the_ghost() {
        let mut app = app();
        let first = material(&mut app, Color::srgb(1.0, 0.0, 0.0));
        let second = material(&mut app, Color::srgb(0.0, 1.0, 0.0));
        let root = app.world_mut().spawn(Ghost).id();
        let mesh = app
            .world_mut()
            .spawn((MeshMaterial3d(first.clone()), ChildOf(root)))
            .id();
        app.update();

        let mut query = app.world_mut().query::<SolidMaterial>();
        let mut solid = query.get_mut(app.world_mut(), mesh).unwrap();
        assert_eq!(solid.handle(), Some(&first));
        solid.set(second.clone());
        app.update();
        assert_eq!(
            ghost_of(&app, mesh).ghost.colour,
            LinearRgba::GREEN.to_vec4()
        );
    }

    /// Meshes showing one material share one ghost material, across ghosts;
    /// a styled ghost has its own; one no mesh shows any more is dropped.
    #[test]
    fn ghost_materials_are_shared_and_dropped_when_unshown() {
        let mut app = app();
        let shared_material = material(&mut app, Color::WHITE);
        let roots: Vec<Entity> = (0..2).map(|_| app.world_mut().spawn(Ghost).id()).collect();
        let mut meshes = Vec::new();
        for &root in &roots {
            for _ in 0..2 {
                meshes.push(
                    app.world_mut()
                        .spawn((MeshMaterial3d(shared_material.clone()), ChildOf(root)))
                        .id(),
                );
            }
        }
        app.update();
        let handles: std::collections::HashSet<_> = meshes
            .iter()
            .map(|&mesh| {
                app.world()
                    .get::<MeshMaterial3d<GhostMaterial>>(mesh)
                    .unwrap()
                    .0
                    .id()
            })
            .collect();
        assert_eq!(handles.len(), 1, "four meshes, one ghost material");
        assert_eq!(app.world().resource::<GhostMaterials>().len(), 1);

        app.world_mut()
            .entity_mut(roots[0])
            .insert(GhostStyle::default());
        app.update();
        assert_eq!(
            app.world().resource::<GhostMaterials>().len(),
            2,
            "the styled ghost has its own"
        );

        app.world_mut().entity_mut(roots[0]).remove::<Ghost>();
        app.world_mut().entity_mut(roots[1]).despawn();
        app.update();
        app.update();
        assert!(
            app.world().resource::<GhostMaterials>().is_empty(),
            "none shown, none kept"
        );
    }

    /// A ghost given a style moves to a material of its own, and the one it
    /// showed before, now shown by nothing, is dropped.
    #[test]
    fn a_material_left_by_a_restyled_ghost_is_dropped() {
        let mut app = app();
        let original = material(&mut app, Color::WHITE);
        let root = app.world_mut().spawn(Ghost).id();
        app.world_mut()
            .spawn((MeshMaterial3d(original), ChildOf(root)));
        app.update();
        app.world_mut()
            .entity_mut(root)
            .insert(GhostStyle::default());
        app.update();
        app.update();
        assert_eq!(app.world().resource::<GhostMaterials>().len(), 1);
    }

    /// A ghost that tests depth asks for a pipeline that does.
    #[test]
    fn a_ghost_can_test_depth() {
        let mut app = app();
        let original = material(&mut app, Color::WHITE);
        let root = app
            .world_mut()
            .spawn((
                Ghost,
                GhostStyle {
                    depth: GhostDepth::Tested,
                    ..default()
                },
            ))
            .id();
        let mesh = app
            .world_mut()
            .spawn((MeshMaterial3d(original), ChildOf(root)))
            .id();
        app.update();
        let ghost = ghost_of(&app, mesh);
        assert_eq!(GhostMaterialKey::from(&ghost).depth, GhostDepth::Tested);
    }
}
