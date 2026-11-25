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
use bevy_ecs::lifecycle::{HookContext, RemovedComponents};
use bevy_ecs::query::Changed;
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::SystemSet;
use bevy_ecs::system::{Commands, Query, ResMut};
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
///
/// **Each entity is filed under the id it was indexed with**, recorded beside
/// it, and taken out under that id, so an id edited in place — through a
/// mutable [`EntityInstance`] or [`Loading`], which fires no hook — can never
/// make a removal erase another entity's entry. [`reindex_renamed`] moves an
/// entity edited that way to its new id, before the frame's
/// [`InstanceSystems::Intake`]; until it runs, the entity is found under the
/// id it was indexed with.
#[derive(Resource, Default, Debug)]
pub struct InstanceIndex {
    by_id: HashMap<String, Vec<Entity>>,
    superseded: HashSet<Entity>,
    loading: HashMap<String, Vec<Entity>>,
    /// The id each instance entity is filed under.
    filed: HashMap<Entity, String>,
    /// The id each loading entity is filed under.
    filed_loading: HashMap<Entity, String>,
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

    fn file_instance(&mut self, entity: Entity, id: String) {
        if let Some(old) = self.filed.remove(&entity) {
            Self::take(&mut self.by_id, &old, entity);
        }
        Self::add(&mut self.by_id, &id, entity);
        self.filed.insert(entity, id);
    }

    fn unfile_instance(&mut self, entity: Entity) {
        if let Some(old) = self.filed.remove(&entity) {
            Self::take(&mut self.by_id, &old, entity);
        }
    }

    fn file_loading(&mut self, entity: Entity, id: String) {
        if let Some(old) = self.filed_loading.remove(&entity) {
            Self::take(&mut self.loading, &old, entity);
        }
        Self::add(&mut self.loading, &id, entity);
        self.filed_loading.insert(entity, id);
    }

    fn unfile_loading(&mut self, entity: Entity) {
        if let Some(old) = self.filed_loading.remove(&entity) {
            Self::take(&mut self.loading, &old, entity);
        }
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
        index.file_instance(context.entity, id)
    });
}

fn unindex_instance(mut world: DeferredWorld, context: HookContext) {
    on_index(&mut world, move |index| {
        index.unfile_instance(context.entity)
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
        index.file_loading(context.entity, id)
    });
}

fn unindex_loading(mut world: DeferredWorld, context: HookContext) {
    on_index(&mut world, move |index| {
        index.unfile_loading(context.entity)
    });
}

/// Moves each instance or loading entity whose id was edited in place to the
/// id it now carries: an edit through a mutable component fires no hook, so
/// the index learns of it from change detection here. [`InstancePlugin`] runs
/// it before [`InstanceSystems::Intake`], for edits made between frames, and
/// again after [`InstanceSystems::Index`], for edits made by a host's own
/// intake or indexing, so the control commands that follow look ids up as
/// they stand.
pub fn reindex_renamed(
    mut index: ResMut<InstanceIndex>,
    instances: Query<(Entity, &EntityInstance), Changed<EntityInstance>>,
    loading: Query<(Entity, &Loading), Changed<Loading>>,
) {
    for (entity, instance) in &instances {
        if index.filed.get(&entity) != Some(&instance.id) {
            index.file_instance(entity, instance.id.clone());
        }
    }
    for (entity, loading) in &loading {
        if index.filed_loading.get(&entity) != Some(&loading.instance) {
            index.file_loading(entity, loading.instance.clone());
        }
    }
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

/// Reconciles element tags — [`ElementNode`] and [`ElementOf`] — with the
/// instances' [`SceneBindings`], so the two always agree: the bindings stay
/// the index a value is applied through, and the tags are the same joins as
/// ECS data. Runs after [`InstanceSystems::Index`], for any host's indexing.
///
/// Only nodes something could have moved are looked at: those a changed
/// instance names or held, those of an instance whose [`EntityInstance`] was
/// taken off, and those whose `ElementOf` went because their instance was
/// despawned. Each is settled once, from every instance that names it now, so
/// the order instances are visited in cannot matter:
///
/// - An instance names a node only while its bindings are indexed.
/// - A node keeps its owner while the owner still names it, so a change to
///   another instance that also names it does not take it over.
/// - A node whose owner no longer names it goes to the instance naming it
///   with the lowest id, or, named by none, loses both tags.
/// - Of several ids one instance names a node by, the node carries the
///   lowest, the same one each time.
///
/// Two instances naming one node is not a shape anything builds on purpose —
/// a nested instance's nodes are meant to join only to it — but indexing by
/// name over descendants can produce it, and the rules above keep the tags
/// consistent when it does.
pub fn tag_elements(
    mut commands: Commands,
    changed: Query<(Entity, &EntityInstance), Changed<EntityInstance>>,
    mut unshown: RemovedComponents<EntityInstance>,
    mut unrelated: RemovedComponents<ElementOf>,
    instances: Query<(Entity, &EntityInstance)>,
    members: Query<&InstanceElements>,
    tags: Query<(Option<&ElementNode>, Option<&ElementOf>)>,
) {
    let mut touched: HashSet<Entity> = HashSet::new();
    for (root, instance) in &changed {
        if instance.bindings.indexed {
            touched.extend(instance.bindings.elements.values().map(|t| t.node));
        }
        if let Ok(held) = members.get(root) {
            touched.extend(held.nodes());
        }
    }
    for root in unshown.read() {
        if let Ok(held) = members.get(root) {
            touched.extend(held.nodes());
        }
    }
    touched.extend(unrelated.read());
    if touched.is_empty() {
        return;
    }

    // Every instance naming a touched node — its entity and id — with the
    // lowest element id it names the node by.
    let mut naming: HashMap<Entity, Vec<(Entity, &String, &String)>> = HashMap::new();
    for (root, instance) in &instances {
        if !instance.bindings.indexed {
            continue;
        }
        for (id, targets) in &instance.bindings.elements {
            if !touched.contains(&targets.node) {
                continue;
            }
            let claims = naming.entry(targets.node).or_default();
            match claims.iter_mut().find(|(claimant, ..)| *claimant == root) {
                Some((.., lowest)) if id < *lowest => *lowest = id,
                Some(_) => {}
                None => claims.push((root, &instance.id, id)),
            }
        }
    }

    for node in touched {
        let Ok((tag, of)) = tags.get(node) else {
            continue;
        };
        let claims = naming.get(&node).map(Vec::as_slice).unwrap_or_default();
        let kept = of.and_then(|of| claims.iter().find(|(claimant, ..)| *claimant == of.0));
        // Entities do not order by when they were made, so the fallback is
        // chosen by instance id, which a host can predict.
        let owner = kept.or_else(|| {
            claims
                .iter()
                .min_by_key(|(claimant, instance, _)| (*instance, *claimant))
        });
        match owner {
            Some(&(root, _, id)) => {
                if tag.is_some_and(|tag| tag.id == *id) && of.is_some_and(|of| of.0 == root) {
                    continue;
                }
                commands
                    .entity(node)
                    .try_insert((ElementNode { id: id.clone() }, ElementOf(root)));
            }
            None if tag.is_some() || of.is_some() => {
                commands
                    .entity(node)
                    .try_remove::<(ElementNode, ElementOf)>();
            }
            None => {}
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
                (
                    reindex_renamed.before(InstanceSystems::Intake),
                    reindex_renamed
                        .after(InstanceSystems::Index)
                        .before(tag_elements),
                    tag_elements
                        .after(InstanceSystems::Index)
                        .before(InstanceSystems::Apply),
                ),
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

    /// An id edited in place is removed under the id it was indexed with, so
    /// a removal never erases another entity's entry; and the index moves it
    /// to its new id once it notices the edit.
    #[test]
    fn an_id_edited_in_place_is_moved_and_removed_by_its_own_entry() {
        use bevy_ecs::system::RunSystemOnce;

        let mut world = World::new();
        let a = world.spawn(instance("a")).id();
        let b = world.spawn(instance("b")).id();
        world.get_mut::<EntityInstance>(a).unwrap().id = "b".into();
        assert_eq!(
            world.resource::<InstanceIndex>().get("a"),
            Some(a),
            "under its indexed id until noticed"
        );
        world.run_system_once(reindex_renamed).unwrap();
        let index = world.resource::<InstanceIndex>();
        assert_eq!(index.get("a"), None);
        let mut both: Vec<_> = index.all("b").collect();
        both.sort();
        let mut expected = vec![a, b];
        expected.sort();
        assert_eq!(both, expected);

        // Edited again and removed before it is noticed: its own entry goes,
        // and the other entity under "b" stays.
        world.get_mut::<EntityInstance>(a).unwrap().id = "c".into();
        world.entity_mut(a).remove::<EntityInstance>();
        let index = world.resource::<InstanceIndex>();
        assert_eq!(index.all("b").collect::<Vec<_>>(), vec![b]);
        assert_eq!(index.get("c"), None);
    }

    /// An app with the plugin notices an id edited in place within a frame.
    #[cfg(feature = "view")]
    #[test]
    fn the_plugin_moves_an_id_edited_in_place() {
        use bevy::prelude::*;

        let mut app = App::new();
        app.add_plugins((MinimalPlugins, InstancePlugin));
        let a = app.world_mut().spawn(instance("a")).id();
        app.update();
        app.world_mut().get_mut::<EntityInstance>(a).unwrap().id = "b".into();
        app.update();
        assert_eq!(app.world().resource::<InstanceIndex>().get("b"), Some(a));
    }

    /// An id a host edits in its own intake is looked up as it stands by the
    /// control commands later in the same frame.
    #[cfg(feature = "view")]
    #[test]
    fn an_id_edited_during_intake_is_found_by_the_commands() {
        use bevy::prelude::*;

        #[derive(Resource, Default)]
        struct Found(Option<Entity>);

        let mut app = App::new();
        app.add_plugins((MinimalPlugins, InstancePlugin))
            .init_resource::<Found>()
            .add_systems(
                Update,
                (
                    (|mut instances: Query<&mut EntityInstance>| {
                        for mut instance in &mut instances {
                            if instance.id == "a" {
                                instance.id = "b".into();
                            }
                        }
                    })
                    .in_set(InstanceSystems::Intake),
                    (|index: Res<InstanceIndex>, mut found: ResMut<Found>| {
                        found.0 = index.get("b");
                    })
                    .after(tag_elements)
                    .before(InstanceSystems::Apply),
                ),
            );
        let a = app.world_mut().spawn(instance("a")).id();
        // One frame: a later frame's pass before intake would find it anyway.
        app.update();
        assert_eq!(app.world().resource::<Found>().0, Some(a));
    }

    /// The same holds for a loading id edited in place.
    #[test]
    fn a_loading_id_edited_in_place_is_moved_and_removed_by_its_own_entry() {
        use bevy_ecs::system::RunSystemOnce;

        let mut world = World::new();
        let loading = |id: &str| Loading {
            instance: id.into(),
        };
        let a = world.spawn(loading("a")).id();
        let b = world.spawn(loading("b")).id();
        world.get_mut::<Loading>(a).unwrap().instance = "b".into();
        world.run_system_once(reindex_renamed).unwrap();
        assert!(!world.resource::<InstanceIndex>().is_loading("a"));
        world.get_mut::<Loading>(a).unwrap().instance = "c".into();
        world.entity_mut(a).remove::<Loading>();
        let index = world.resource::<InstanceIndex>();
        assert!(index.is_loading("b"), "b's own entry stays");
        assert!(!index.is_loading("c"));
        world.entity_mut(b).remove::<Loading>();
        assert!(!world.resource::<InstanceIndex>().any_loading());
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

        // The bindings change: "head" is renamed "face", "arm" is dropped and
        // a new element "eye" takes a node of its own. The tags follow.
        let eye = world.spawn_empty().id();
        {
            let mut shown = world.get_mut::<EntityInstance>(root).unwrap();
            let elements = &mut shown.bindings.elements;
            let head_targets = elements.remove("head").unwrap();
            elements.insert("face".into(), head_targets);
            elements.remove("arm");
            elements.insert(
                "eye".into(),
                ElementTargets {
                    node: eye,
                    mesh: None,
                    morph: None,
                },
            );
        }
        schedule.run(&mut world);
        assert_eq!(
            world.get::<ElementNode>(head),
            Some(&ElementNode { id: "face".into() }),
            "renamed"
        );
        assert!(world.get::<ElementNode>(arm).is_none(), "dropped");
        assert!(world.get::<ElementOf>(arm).is_none());
        assert_eq!(world.get::<ElementOf>(eye), Some(&ElementOf(root)));
        let mut nodes = world
            .get::<InstanceElements>(root)
            .unwrap()
            .nodes()
            .to_vec();
        nodes.sort();
        let mut expected = vec![head, eye];
        expected.sort();
        assert_eq!(nodes, expected);

        // "eye" moves to another node: the old one is untagged.
        let other = world.spawn_empty().id();
        world
            .get_mut::<EntityInstance>(root)
            .unwrap()
            .bindings
            .elements
            .get_mut("eye")
            .unwrap()
            .node = other;
        schedule.run(&mut world);
        assert!(world.get::<ElementOf>(eye).is_none(), "moved off");
        assert_eq!(
            world.get::<ElementNode>(other),
            Some(&ElementNode { id: "eye".into() })
        );

        // Bindings reset to unindexed name nothing: every node is untagged.
        world.get_mut::<EntityInstance>(root).unwrap().bindings = SceneBindings::default();
        schedule.run(&mut world);
        assert!(world.get::<ElementNode>(head).is_none(), "reset");
        assert!(world.get::<ElementOf>(other).is_none());
        assert!(world.get::<InstanceElements>(root).is_none());
    }

    fn element(node: Entity) -> crate::binding::ElementTargets {
        crate::binding::ElementTargets {
            node,
            mesh: None,
            morph: None,
        }
    }

    fn indexed(id: &str, elements: &[(&str, Entity)]) -> EntityInstance {
        let mut shown = instance(id);
        shown.bindings.indexed = true;
        for (element_id, node) in elements {
            shown
                .bindings
                .elements
                .insert((*element_id).into(), element(*node));
        }
        shown
    }

    /// An instance whose [`EntityInstance`] is taken off while it lives on
    /// leaves no tags behind on its nodes.
    #[test]
    fn taking_an_instance_off_untags_its_elements() {
        use bevy_ecs::schedule::Schedule;

        let mut world = World::new();
        let head = world.spawn_empty().id();
        let root = world.spawn(indexed("a", &[("head", head)])).id();
        let mut schedule = Schedule::default();
        schedule.add_systems(tag_elements);
        schedule.run(&mut world);
        assert_eq!(world.get::<ElementOf>(head), Some(&ElementOf(root)));

        world.entity_mut(root).remove::<EntityInstance>();
        schedule.run(&mut world);
        assert!(world.get::<ElementNode>(head).is_none());
        assert!(world.get::<ElementOf>(head).is_none());
        assert!(world.get::<InstanceElements>(root).is_none());
    }

    /// Of two instances naming one node, the one that has it keeps it through
    /// the other's changes, and it passes to the other once its owner stops
    /// naming it or is gone — whichever instance is visited first.
    #[test]
    fn a_node_two_instances_name_keeps_its_owner_and_falls_back() {
        use bevy_ecs::schedule::Schedule;

        for first_owns in [false, true] {
            let mut world = World::new();
            let node = world.spawn_empty().id();
            let a = world.spawn(indexed("a", &[("head", node)])).id();
            let b = world.spawn(indexed("b", &[("face", node)])).id();
            let mut schedule = Schedule::default();
            schedule.add_systems(tag_elements);
            schedule.run(&mut world);
            assert_eq!(
                world.get::<ElementOf>(node),
                Some(&ElementOf(a)),
                "lowest instance id"
            );

            let (owner, other, other_id) = if first_owns {
                (a, b, "face")
            } else {
                // Hand it to b: a stops naming it, then names it again.
                world
                    .get_mut::<EntityInstance>(a)
                    .unwrap()
                    .bindings
                    .elements
                    .clear();
                schedule.run(&mut world);
                assert_eq!(
                    world.get::<ElementOf>(node),
                    Some(&ElementOf(b)),
                    "fell back"
                );
                world
                    .get_mut::<EntityInstance>(a)
                    .unwrap()
                    .bindings
                    .elements
                    .insert("head".into(), element(node));
                schedule.run(&mut world);
                (b, a, "head")
            };
            assert_eq!(
                world.get::<ElementOf>(node),
                Some(&ElementOf(owner)),
                "kept"
            );

            // A change to the other that leaves its claim as it was.
            world.get_mut::<EntityInstance>(other).unwrap().ready = true;
            schedule.run(&mut world);
            assert_eq!(
                world.get::<ElementOf>(node),
                Some(&ElementOf(owner)),
                "first owns: {first_owns}"
            );

            // The owner goes: the other, still naming it, takes it.
            world.despawn(owner);
            schedule.run(&mut world);
            assert_eq!(world.get::<ElementOf>(node), Some(&ElementOf(other)));
            assert_eq!(
                world.get::<ElementNode>(node),
                Some(&ElementNode {
                    id: other_id.into()
                })
            );
        }
    }

    /// A node that outlives its instance keeps no element id.
    #[test]
    fn a_node_outliving_its_instance_loses_its_element_id() {
        use bevy_ecs::schedule::Schedule;

        let mut world = World::new();
        let head = world.spawn_empty().id();
        let root = world.spawn(indexed("a", &[("head", head)])).id();
        let mut schedule = Schedule::default();
        schedule.add_systems(tag_elements);
        schedule.run(&mut world);
        assert!(world.get::<ElementNode>(head).is_some());

        world.despawn(root);
        assert!(world.get::<ElementOf>(head).is_none(), "the relationship's");
        schedule.run(&mut world);
        assert!(world.get::<ElementNode>(head).is_none());
    }

    /// A node that moves from one instance's bindings to another's in one
    /// frame ends up tagged for the one it moved to, whichever instance is
    /// visited first.
    #[test]
    fn a_node_moved_between_instances_keeps_its_new_tags() {
        use bevy_ecs::schedule::Schedule;

        for giver_first in [false, true] {
            let mut world = World::new();
            let node = world.spawn_empty().id();
            let spare = world.spawn_empty().id();
            // Instances are visited in the order they were spawned, both
            // being in one archetype.
            let (giver, taker) = if giver_first {
                let giver = world.spawn(indexed("a", &[("head", node)])).id();
                (giver, world.spawn(indexed("b", &[])).id())
            } else {
                let taker = world.spawn(indexed("b", &[])).id();
                (world.spawn(indexed("a", &[("head", node)])).id(), taker)
            };
            let mut schedule = Schedule::default();
            schedule.add_systems(tag_elements);
            schedule.run(&mut world);
            assert_eq!(world.get::<ElementOf>(node), Some(&ElementOf(giver)));

            world
                .get_mut::<EntityInstance>(giver)
                .unwrap()
                .bindings
                .elements
                .insert("head".into(), element(spare));
            world
                .get_mut::<EntityInstance>(taker)
                .unwrap()
                .bindings
                .elements
                .insert("face".into(), element(node));
            schedule.run(&mut world);
            assert_eq!(
                world.get::<ElementOf>(node),
                Some(&ElementOf(taker)),
                "giver visited first: {giver_first}"
            );
            assert_eq!(
                world.get::<ElementNode>(node),
                Some(&ElementNode { id: "face".into() })
            );
            assert_eq!(world.get::<ElementOf>(spare), Some(&ElementOf(giver)));
        }
    }
}
