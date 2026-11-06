//! Fading something into view: every mesh beneath an entity, from clear to
//! its own look, over a moment, so it arrives rather than appearing.
//!
//! A host puts [`FadeIn`] on an entity once its meshes are there to show —
//! a robot just framed, say. Each mesh draws with a copy of its own material
//! for the length of the fade, its opacity rising from nothing to the
//! material's own, and is then given its own material back, untouched. The
//! entity loses `FadeIn` when the fade is done, so it fades once: nothing
//! fades again unless a host asks again.
//!
//! **The copies use alpha to coverage**, which keeps depth testing and depth
//! writes and needs the view's multisampling, rather than blending, under
//! which a body would show its own inside while half faded. A material that
//! is already blended keeps its own mode. Meshes under a ghost are left
//! alone: a ghost draws with its own material.

use bevy::prelude::*;

use crate::ghost::Ghost;

/// Fades every mesh beneath this entity into view ([module docs](self)).
#[derive(Component, Clone, Copy, Debug)]
pub struct FadeIn {
    /// How long the fade takes.
    pub seconds: f32,
}

impl Default for FadeIn {
    /// A quarter of a second: quick enough not to be waited on, slow enough
    /// to be seen as arriving.
    fn default() -> Self {
        Self { seconds: 0.25 }
    }
}

/// A fade under way: each mesh's own material, the copy it draws with
/// meanwhile and that copy's full opacity, and how far along the fade is.
#[derive(Component)]
struct Fading {
    meshes: Vec<(
        Entity,
        Handle<StandardMaterial>,
        Handle<StandardMaterial>,
        f32,
    )>,
    elapsed: f32,
}

pub struct FadePlugin;

impl Plugin for FadePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (start_fades, advance_fades).chain());
    }
}

/// Gives each mesh beneath a newly fading entity a clear copy of its own
/// material.
#[allow(clippy::type_complexity)]
fn start_fades(
    mut commands: Commands,
    starting: Query<Entity, (Added<FadeIn>, Without<Fading>)>,
    children: Query<&Children>,
    parents: Query<&ChildOf>,
    ghosts: Query<(), With<Ghost>>,
    mut meshes: Query<&mut MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for root in &starting {
        let ghosted = std::iter::once(root)
            .chain(parents.iter_ancestors(root))
            .any(|entity| ghosts.contains(entity));
        let mut fading = Vec::new();
        if !ghosted {
            for mesh in std::iter::once(root).chain(children.iter_descendants(root)) {
                let Ok(mut material) = meshes.get_mut(mesh) else {
                    continue;
                };
                let Some(own) = materials.get(&material.0) else {
                    continue;
                };
                let mut copy = own.clone();
                let opacity = copy.base_color.alpha();
                if !matches!(
                    copy.alpha_mode,
                    AlphaMode::Blend | AlphaMode::Premultiplied | AlphaMode::Add
                ) {
                    copy.alpha_mode = AlphaMode::AlphaToCoverage;
                }
                copy.base_color.set_alpha(0.0);
                let copy = materials.add(copy);
                let original = std::mem::replace(&mut material.0, copy.clone());
                fading.push((mesh, original, copy, opacity));
            }
        }
        commands.entity(root).insert(Fading {
            meshes: fading,
            elapsed: 0.0,
        });
    }
}

/// Raises each fading mesh's opacity, and gives it its own material back
/// once the fade is done.
fn advance_fades(
    mut commands: Commands,
    time: Res<Time>,
    mut fades: Query<(Entity, &FadeIn, &mut Fading)>,
    mut meshes: Query<&mut MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (root, fade, mut fading) in &mut fades {
        fading.elapsed += time.delta_secs();
        let progress = (fading.elapsed / fade.seconds.max(f32::EPSILON)).clamp(0.0, 1.0);
        if progress < 1.0 {
            for (_, _, copy, opacity) in &fading.meshes {
                if let Some(mut material) = materials.get_mut(copy) {
                    material.base_color.set_alpha(opacity * progress);
                }
            }
            continue;
        }
        for (mesh, original, copy, _) in fading.meshes.drain(..) {
            // Only where the copy is still what it draws with: a host that
            // gave the mesh another material meanwhile keeps its own choice.
            if let Ok(mut material) = meshes.get_mut(mesh) {
                if material.0 == copy {
                    material.0 = original;
                }
            }
        }
        commands.entity(root).remove::<(FadeIn, Fading)>();
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use bevy::time::TimeUpdateStrategy;

    use super::*;

    /// A fade starts clear, rises, and ends with the mesh's own material back
    /// and the entity no longer fading.
    #[test]
    fn a_fade_rises_and_gives_the_material_back() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<StandardMaterial>()
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
                100,
            )))
            .add_systems(Update, (start_fades, advance_fades).chain());
        let own = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        let root = app.world_mut().spawn(FadeIn { seconds: 0.25 }).id();
        let mesh = app
            .world_mut()
            .spawn((MeshMaterial3d(own.clone()), ChildOf(root)))
            .id();
        let drawn = |app: &App| {
            let handle = app
                .world()
                .get::<MeshMaterial3d<StandardMaterial>>(mesh)
                .unwrap()
                .0
                .clone();
            let material = app
                .world()
                .resource::<Assets<StandardMaterial>>()
                .get(&handle)
                .unwrap()
                .clone();
            (handle, material)
        };

        app.update();
        let (handle, material) = drawn(&app);
        assert_ne!(handle, own, "drawing with a copy while fading");
        assert!(material.base_color.alpha() < 0.5, "starting clear");
        assert_eq!(material.alpha_mode, AlphaMode::AlphaToCoverage);

        for _ in 0..4 {
            app.update();
        }
        let (handle, _) = drawn(&app);
        assert_eq!(handle, own, "its own material back");
        assert!(app.world().get::<FadeIn>(root).is_none(), "fades once");
    }
}
