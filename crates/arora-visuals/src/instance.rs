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

use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;

use crate::binding::SceneBindings;
use crate::meta::AssetMeta;

/// One entity drawn by a view: an entity carrying its metadata, the GLB
/// serving it, and the joins to the spawned scene once indexed. Its scene is
/// a child; its camera is its own entity ([`ViewOf`]).
#[derive(Component)]
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
