//! Where an animatable lands once the asset's scene has spawned.
//!
//! [`Binding`](crate::meta::Binding) is what the GLB *declares*: this
//! animatable moves this feature of the elements named here. A
//! [`SceneBinding`] is that same statement **resolved** — the named node is
//! now a spawned entity. Same relation, two stages, and the names say which.
//!
//! The word is qualified because at the architecture level a binding pairs an
//! entity with an engine instance. That is a different relation at a different
//! level, and the two come up in the same conversations.
//!
//! **What is not here is pose state.** What an animatable last wrote, the
//! triples per-axis writes assemble, which animatables contend for a target —
//! those describe applying a pose, not where it lands, and they belong with
//! the code that applies it.

use std::collections::HashMap;

use bevy_ecs::entity::Entity;

use crate::meta::FeatureKind;

/// The entities one element's features land on.
#[derive(Debug, Clone, Copy)]
pub struct ElementTargets {
    /// Its glTF node: the transform features.
    pub node: Entity,
    /// Its own mesh — the node's, or its first primitive's: the static
    /// material features, the pick, the glow. `None` for a group.
    pub mesh: Option<Entity>,
    /// The descendant carrying its morph weights, if it has morph targets.
    pub morph: Option<Entity>,
}

/// One place an animatable lands in a spawned scene.
#[derive(Debug, Clone)]
pub struct SceneBinding {
    /// The entity it moves: the element's node for a transform feature, its
    /// mesh for a material feature, the bearer of its morph weights for a
    /// morph.
    pub entity: Entity,
    pub feature: FeatureKind,
    /// The component of the feature it drives alone
    /// ([`Binding::axis`](crate::meta::Binding::axis)).
    pub axis: Option<usize>,
    /// The morph target's index among the mesh's weights, for a morph.
    pub morph_index: Option<usize>,
}

/// The joins from an instance's animatables to its spawned scene: built once
/// the GLB scene has spawned, empty until then.
#[derive(Default)]
pub struct SceneBindings {
    /// animatable id → every place it lands. **Many, not one**: several
    /// elements can share an animatable, and each takes the same value. A
    /// map of one would silently drop all but the last, which is a defect
    /// only a push-side index can have — a renderer that instead lets each
    /// element subscribe to the animatables it names cannot.
    pub by_id: HashMap<String, Vec<SceneBinding>>,
    /// element id → the entities its features land on: what a pick tests,
    /// what a selection outlines, what a static feature is set on.
    pub elements: HashMap<String, ElementTargets>,
    /// Whether the joins are built. Until they are, the instance has a scene
    /// but nothing addressing it, so anything that walks elements must check
    /// this rather than find an empty map and conclude the asset is empty.
    pub indexed: bool,
}
