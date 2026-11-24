//! One entity being drawn.
//!
//! An entity is Model + Behavior running on one engine instance, and entities
//! nest — a robot entity contains the entity on its screen. [`EntityInstance`]
//! is a world's handle on one of those: its GLB, where its animatables land,
//! the camera over it, and whether it is actually on screen yet.
//!
//! **What is not here is anything a product owns.** Where values come from and
//! where the instance stands are both real, and both are a consumer's: in an
//! ECS they do not have to live in this struct to live on this entity. A
//! renderer puts its own `Rig` and `Slot` components beside this one and
//! queries `(&EntityInstance, &Rig)`; this crate queries `&EntityInstance` and
//! never learns what a rig is. That is what keeps a rig's *form* — a direct
//! joint rig, an IK rig — out of the shared layer entirely.

use std::collections::{HashMap, HashSet};

use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::query::Changed;
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::SystemSet;
use bevy_ecs::system::{Commands, Query};
use bevy_ecs::world::{DeferredWorld, World};

use crate::binding::SceneBindings;
use crate::meta::AssetMeta;

/// One entity drawn by a view: an entity carrying its metadata, the GLB
/// serving it, and the joins to the spawned scene once indexed. Its scene is
/// a child; its camera is its own entity ([`ViewOf`]).
#[derive(Component)]
#[component(on_insert = index_instance, on_discard = unindex_instance)]
pub struct EntityInstance {
    /// What a front end names it by, and what a pick reports.
    pub id: String,
    /// Where its GLB is served from.
    pub asset_path: String,
    pub meta: AssetMeta,
    /// The joins to the spawned scene, once every element's node exists.
    pub bindings: SceneBindings,
    /// Its camera.
    pub camera: Entity,
    /// The order among the target's cameras its camera takes once ready.
    ///
    /// The order itself, not whatever a consumer derived it from. A view that
    /// lays instances out in slots computes this from the slot and keeps the
    /// slot; one that places them by transform computes it some other way.
    /// Either way what this layer needs to know is which draws over which.
    pub draw_order: isize,
    /// Whether it is drawn: every mesh visible to its camera has its pipeline
    /// compiled and the GPU has finished a frame drawing them.
    ///
    /// **Spawned is not drawable.** A browser's WebGL compiles pipelines
    /// asynchronously, so a mesh that exists, passes culling and is counted
    /// among the visible can still draw nothing. Until this is set the
    /// instance's camera is ordered below the one that clears the target, so
    /// the target never shows a half-compiled asset.
    pub ready: bool,
}

/// An instance being replaced by a reload under the same id.
///
/// It keeps drawing until its successor is ready, then goes, so a reload never
/// shows an empty rectangle. It is not reported ready and reports no picks.
#[derive(Component)]
#[component(on_insert = mark_superseded, on_discard = unmark_superseded)]
pub struct Superseded;

/// On a camera: the instance it draws.
///
/// The back-reference to [`EntityInstance::camera`]. A camera needs it because
/// the signals that say an instance is drawable arrive against the camera that
/// drew it, not against the instance.
#[derive(Component)]
pub struct ViewOf(pub Entity);

/// The camera that clears a render target, so a target with no instance ready
/// shows a background rather than whatever was in the texture.
#[derive(Component)]
pub struct ClearCamera;

/// The order among a target's cameras of the camera that clears it.
///
/// An instance's camera ordered below this draws nothing the target shows,
/// which is how an instance that is not ready is hidden without being
/// despawned or made invisible — it keeps warming where it cannot be seen.
pub const CLEAR_ORDER: isize = -1;

/// On an entity a host is loading an instance into, until it carries the
/// [`EntityInstance`] that instance becomes: what is asked of `instance`
/// before it can be addressed waits for it, rather than being refused.
///
/// **A host marks its loading roots**, and takes the mark off when it puts
/// the instance on. Whether an instance is still to come is the host's to
/// know — only it has the load under way — and without the mark a control
/// added in the frames a load takes is refused as being on nothing. A host
/// that spawns each instance whole, with no load in between, never adds it.
///
/// It may stand beside an instance of the same id already there, as when a
/// host loads a replacement in the background: what addresses the id reaches
/// the one there, and the mark says only that another is on its way.
#[derive(Component, Clone, Debug)]
#[component(on_insert = index_loading, on_discard = unindex_loading)]
pub struct Loading {
    /// The host's id for the instance being loaded, which becomes its
    /// [`EntityInstance::id`].
    pub instance: String,
}

/// Which Bevy entity carries the instance under each host id, and which ids
/// are still loading.
///
/// **Kept by the components' own hooks**, so it is right the moment an
/// [`EntityInstance`], [`Superseded`] or [`Loading`] is inserted, replaced,
/// removed or despawned, whoever does it: a host never maintains it, and
/// nothing has to scan every instance to find one by its id. The id stays an
/// opaque string — a Studio instance id, a page's own id and an Arora path
/// all key it unchanged.
///
/// An id can name two entities while a reload is under way: the instance
/// showing and its replacement. The one not [`Superseded`] is the instance
/// the id addresses.
#[derive(Resource, Default, Debug)]
pub struct InstanceIndex {
    by_id: HashMap<String, Vec<Entity>>,
    superseded: HashSet<Entity>,
    loading: HashMap<String, Vec<Entity>>,
}

impl InstanceIndex {
    /// The instance an id addresses: the one not being replaced, or, while
    /// only a replaced one is there, that one.
    pub fn get(&self, id: &str) -> Option<Entity> {
        let held = self.by_id.get(id)?;
        held.iter()
            .copied()
            .find(|entity| !self.superseded.contains(entity))
            .or_else(|| held.first().copied())
    }

    /// The instance an id addresses, unless every one under it is being
    /// replaced: what draws, picks and outlines take, since a replaced
    /// instance does neither.
    pub fn current(&self, id: &str) -> Option<Entity> {
        self.by_id
            .get(id)?
            .iter()
            .copied()
            .find(|entity| !self.superseded.contains(entity))
    }

    /// Every entity carrying an instance under the id.
    pub fn all(&self, id: &str) -> impl Iterator<Item = Entity> + '_ {
        self.by_id.get(id).into_iter().flatten().copied()
    }

    /// Whether a host is loading an instance under the id ([`Loading`]).
    pub fn is_loading(&self, id: &str) -> bool {
        self.loading.contains_key(id)
    }

    /// Whether any instance is loading.
    pub fn any_loading(&self) -> bool {
        !self.loading.is_empty()
    }

    fn add(map: &mut HashMap<String, Vec<Entity>>, id: &str, entity: Entity) {
        let held = map.entry(id.to_string()).or_default();
        if !held.contains(&entity) {
            held.push(entity);
        }
    }

    fn take(map: &mut HashMap<String, Vec<Entity>>, id: &str, entity: Entity) {
        if let Some(held) = map.get_mut(id) {
            held.retain(|held| *held != entity);
            if held.is_empty() {
                map.remove(id);
            }
        }
    }
}

/// Runs `change` on the index, making the index first if the world has none:
/// a host that never added the plugin still gets a correct index, because
/// what it records cannot depend on which plugin got there first.
fn on_index(world: &mut DeferredWorld, change: impl FnOnce(&mut InstanceIndex) + Send + 'static) {
    if let Some(mut index) = world.get_resource_mut::<InstanceIndex>() {
        change(&mut index);
    } else {
        world.commands().queue(move |world: &mut World| {
            change(&mut world.get_resource_or_init::<InstanceIndex>());
        });
    }
}

fn index_instance(mut world: DeferredWorld, context: HookContext) {
    let Some(id) = world
        .get::<EntityInstance>(context.entity)
        .map(|instance| instance.id.clone())
    else {
        return;
    };
    on_index(&mut world, move |index| {
        InstanceIndex::add(&mut index.by_id, &id, context.entity)
    });
}

fn unindex_instance(mut world: DeferredWorld, context: HookContext) {
    let Some(id) = world
        .get::<EntityInstance>(context.entity)
        .map(|instance| instance.id.clone())
    else {
        return;
    };
    on_index(&mut world, move |index| {
        InstanceIndex::take(&mut index.by_id, &id, context.entity)
    });
}

fn mark_superseded(mut world: DeferredWorld, context: HookContext) {
    on_index(&mut world, move |index| {
        index.superseded.insert(context.entity);
    });
}

fn unmark_superseded(mut world: DeferredWorld, context: HookContext) {
    on_index(&mut world, move |index| {
        index.superseded.remove(&context.entity);
    });
}

fn index_loading(mut world: DeferredWorld, context: HookContext) {
    let Some(id) = world
        .get::<Loading>(context.entity)
        .map(|loading| loading.instance.clone())
    else {
        return;
    };
    on_index(&mut world, move |index| {
        InstanceIndex::add(&mut index.loading, &id, context.entity)
    });
}

fn unindex_loading(mut world: DeferredWorld, context: HookContext) {
    let Some(id) = world
        .get::<Loading>(context.entity)
        .map(|loading| loading.instance.clone())
    else {
        return;
    };
    on_index(&mut world, move |index| {
        InstanceIndex::take(&mut index.loading, &id, context.entity)
    });
}

/// Where each stage of a frame's work on instances runs, in `Update`, in
/// this order. Every host orders its own systems into these rather than
/// against another host's, so a third consumer can slot in without knowing
/// either:
///
/// - **Intake**: what a host was asked — its queues — turned into components.
/// - **Index**: a spawned scene joined to its instance's elements.
/// - **Apply**: values applied to the scene; control commands run between
///   Index and here, so a control added this frame lands on an indexed
///   instance and its write-back is applied in the same frame.
/// - **Derive**: what is drawn from intent and values — materials, ghosts,
///   outlines.
///
/// Camera framing follows in `PostUpdate`
/// ([`FramingSystems`](crate::active_area::FramingSystems)).
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstanceSystems {
    Intake,
    Index,
    Apply,
    Derive,
}

/// On a glTF node an element became: the element's RobotData id.
///
/// Elements are in the ECS so element-level queries, change detection and
/// relationships — a face mounted on a robot's screen element — can be
/// expressed; a map inside a component is invisible to a query. Not named
/// `Element`, which is the declaration ([`meta::Element`](crate::meta::Element));
/// this is the node it was joined to.
#[derive(Component, Debug, Clone, PartialEq, Eq)]
pub struct ElementNode {
    pub id: String,
}

/// On an element's node: the instance it is an element of.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
#[relationship(relationship_target = InstanceElements)]
pub struct ElementOf(#[entities] pub Entity);

/// On an instance: the nodes of its elements, kept by [`ElementOf`].
#[derive(Component, Debug, Default)]
#[relationship_target(relationship = ElementOf)]
pub struct InstanceElements(Vec<Entity>);

impl InstanceElements {
    pub fn nodes(&self) -> &[Entity] {
        &self.0
    }
}

/// Tags each indexed instance's element nodes with [`ElementNode`] and
/// [`ElementOf`], from its [`SceneBindings`], so the two always agree: the
/// bindings stay the index a value is applied through, and the tags are the
/// same joins as ECS data. Runs after [`InstanceSystems::Index`], for any
/// host's indexing.
///
/// A node one element already tags keeps that tag; with ids taken in order, a
/// node two elements name carries the first, the same one each time.
pub fn tag_elements(
    mut commands: Commands,
    instances: Query<(Entity, &EntityInstance), Changed<EntityInstance>>,
    tagged: Query<&ElementOf>,
) {
    for (root, instance) in &instances {
        if !instance.bindings.indexed {
            continue;
        }
        let mut ids: Vec<&String> = instance.bindings.elements.keys().collect();
        ids.sort();
        let mut done = HashSet::new();
        for id in ids {
            let node = instance.bindings.elements[id].node;
            if !done.insert(node) || tagged.get(node).is_ok_and(|of| of.0 == root) {
                continue;
            }
            commands
                .entity(node)
                .try_insert((ElementNode { id: id.clone() }, ElementOf(root)));
        }
    }
}

/// The instance index, the frame's [`InstanceSystems`] in order, and element
/// tagging, for every host. Several plugins and hosts add it; it installs once.
#[cfg(feature = "view")]
pub struct InstancePlugin;

#[cfg(feature = "view")]
impl bevy::app::Plugin for InstancePlugin {
    fn is_unique(&self) -> bool {
        false
    }

    fn build(&self, app: &mut bevy::app::App) {
        use bevy::prelude::*;
        if !crate::installed::first_install::<Self>(app) {
            return;
        }
        app.init_resource::<InstanceIndex>()
            .configure_sets(
                Update,
                (
                    InstanceSystems::Intake,
                    InstanceSystems::Index,
                    InstanceSystems::Apply,
                    InstanceSystems::Derive,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                tag_elements
                    .after(InstanceSystems::Index)
                    .before(InstanceSystems::Apply),
            );
    }
}

#[cfg(test)]
mod tests {
    use bevy_ecs::world::World;

    use super::*;

    fn instance(id: &str) -> EntityInstance {
        EntityInstance {
            id: id.into(),
            asset_path: String::new(),
            meta: AssetMeta::default(),
            bindings: SceneBindings::default(),
            camera: Entity::PLACEHOLDER,
            draw_order: 0,
            ready: false,
        }
    }

    /// The index follows instances as they come, are replaced, renamed and
    /// go, with no resource made beforehand.
    #[test]
    fn the_index_follows_instances_through_their_lives() {
        let mut world = World::new();
        let a = world.spawn(instance("a")).id();
        let index = |world: &World| world.resource::<InstanceIndex>().get("a");
        assert_eq!(index(&world), Some(a), "indexed on spawn");

        // A replacement loading in the background: the id addresses the new
        // one once the old is marked replaced.
        let b = world.spawn(instance("a")).id();
        world.entity_mut(a).insert(Superseded);
        assert_eq!(index(&world), Some(b));
        assert_eq!(world.resource::<InstanceIndex>().current("a"), Some(b));
        world.despawn(b);
        assert_eq!(index(&world), Some(a), "only the replaced one is left");
        assert_eq!(world.resource::<InstanceIndex>().current("a"), None);

        // Renamed in place.
        world.entity_mut(a).insert(instance("c"));
        assert_eq!(index(&world), None);
        assert_eq!(world.resource::<InstanceIndex>().get("c"), Some(a));
        world.entity_mut(a).remove::<Superseded>();
        assert!(world.resource::<InstanceIndex>().current("c").is_some());

        world.despawn(a);
        assert_eq!(world.resource::<InstanceIndex>().get("c"), None);
    }

    #[test]
    fn the_index_knows_what_is_loading() {
        let mut world = World::new();
        let root = world
            .spawn(Loading {
                instance: "a".into(),
            })
            .id();
        assert!(world.resource::<InstanceIndex>().is_loading("a"));
        assert!(world.resource::<InstanceIndex>().any_loading());
        world.entity_mut(root).remove::<Loading>();
        assert!(!world.resource::<InstanceIndex>().is_loading("a"));
        assert!(!world.resource::<InstanceIndex>().any_loading());
    }

    /// An indexed instance's element nodes are tagged with their ids and
    /// related to it; one not yet indexed is left alone.
    #[test]
    fn an_indexed_instance_s_elements_are_tagged() {
        use bevy_ecs::schedule::Schedule;

        use crate::binding::ElementTargets;

        let mut world = World::new();
        let head = world.spawn_empty().id();
        let arm = world.spawn_empty().id();
        let mut shown = instance("a");
        for (id, node) in [("head", head), ("arm", arm)] {
            shown.bindings.elements.insert(
                id.into(),
                ElementTargets {
                    node,
                    mesh: None,
                    morph: None,
                },
            );
        }
        let root = world.spawn(shown).id();
        let mut schedule = Schedule::default();
        schedule.add_systems(tag_elements);
        schedule.run(&mut world);
        assert!(world.get::<ElementNode>(head).is_none(), "not indexed yet");

        world
            .get_mut::<EntityInstance>(root)
            .unwrap()
            .bindings
            .indexed = true;
        schedule.run(&mut world);
        assert_eq!(
            world.get::<ElementNode>(head),
            Some(&ElementNode { id: "head".into() })
        );
        assert_eq!(world.get::<ElementOf>(arm), Some(&ElementOf(root)));
        let mut nodes = world
            .get::<InstanceElements>(root)
            .unwrap()
            .nodes()
            .to_vec();
        nodes.sort();
        let mut expected = vec![head, arm];
        expected.sort();
        assert_eq!(nodes, expected);
    }
}
