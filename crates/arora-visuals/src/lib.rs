//! The shared layer above Bevy for Semio assets.
//!
//! **This is not a renderer.** Bevy is the renderer. What lives here is
//! everything a Semio renderer needs that Bevy does not supply and that no
//! single product should own: the asset semantics, the data modelling, and the
//! interaction primitives.
//!
//! # What it must not contain
//!
//! Its two consumers do not share a rendering model. Vizij's view is a 2D
//! compositing model — Z-up, faces in the XY plane layered along Z, an
//! orthographic camera fit to the asset's authored bounds, ambient-only light
//! composed into unlit materials. A robot scene is perspective, lit and
//! shadowed. **The moment this crate contains an up-axis, a camera convention
//! or a lighting model, it has excluded one of the two.**
//!
//! The same line applies to vocabulary. The concepts here are element,
//! controllable, entity instance and view. A type or a field named for a face
//! has already decided which consumer it serves.
//!
//! **An entity is Model + Behavior running on one engine instance, and
//! entities nest** — a robot entity contains the entity on its screen. That is
//! the Arora architecture's definition and the one this crate follows, because
//! containment is what a screen showing another asset *is*, rather than
//! something a renderer invents.
//!
//! **Take care naming anything near it.** "Entity" means something different
//! at three other places in the stack: Arora's key grammar makes the *joint*
//! the entity (`robot1/joint1.position` → namespace `robot1`, entity
//! `joint1`), which is what this crate calls an [`Element`](meta::Element);
//! Studio splits it into `entityId` for the model and `entityInstanceId` for
//! the occurrence; and Bevy's `Entity` is the ECS id, which appears as a field
//! inside the very types that would carry the name. "Entity instance" names
//! the live occurrence, which no other reading claims.
//!
//! "Binding" is spoken for in the same way: at the architecture level it pairs
//! an entity with an engine instance. What this crate joins is an animatable
//! to the thing it moves, which is a different relation at a different level.
//!
//! # The pointer
//!
//! Picks, the camera's gestures and the gizmos' grabs all take a press only
//! where it lands on the window itself, so a press on a panel or a control
//! drawn over the scene is never the scene's. **They read that from Bevy's
//! picking**: a host needs its pointer input with window picking enabled,
//! and nothing else in the scene pickable under the pointer. This crate's
//! own handles and aids are never pickable; a mesh picking backend that picks
//! a host's meshes takes the press off the window, and the scene stops
//! answering it.
//!
//! # What it costs to be here
//!
//! A module that names a Bevy type binds every consumer of this crate to one
//! Bevy minor version, for the reason the workspace manifest gives. That is
//! the price of sharing a `Component`, and it is worth paying for code that
//! has to agree across products.
//!
//! It is not worth paying for code that needs no Bevy type. [`tone`] is the
//! standing example: colour science, no imports, no lockstep. **A module that
//! can stay free of Bevy should stay free of Bevy.**

#[cfg(feature = "view")]
pub mod active_area;
#[cfg(feature = "view")]
pub mod axes;
#[cfg(feature = "view")]
pub mod axis_line;
pub mod binding;
#[cfg(feature = "camera_input")]
pub mod camera_input;
#[cfg(feature = "view")]
pub mod drawn;
#[cfg(feature = "view")]
pub mod fade;
#[cfg(feature = "view")]
pub mod ghost;
#[cfg(feature = "view")]
pub mod glow;
pub mod hold;
pub mod instance;
#[cfg(feature = "view")]
pub mod joint_gizmo;
pub mod meta;
#[cfg(feature = "view")]
pub mod outline;
#[cfg(feature = "view")]
pub mod overlay;
#[cfg(feature = "view")]
pub mod pick;
#[cfg(feature = "view")]
pub mod select;
#[cfg(feature = "snapshot")]
pub mod snapshot;
pub mod target;
pub mod tone;
#[cfg(feature = "view")]
pub mod transform_gizmo;
pub mod value;
#[cfg(feature = "view_gizmo")]
pub mod view_gizmo;
#[cfg(feature = "view")]
pub mod view_offset;

pub use binding::{ElementTargets, SceneBinding, SceneBindings};
pub use hold::{HoldTarget, Holds};
pub use instance::{ClearCamera, EntityInstance, Superseded, ViewOf, CLEAR_ORDER};
pub use meta::{glb_json_chunk, AssetMeta, Binding, Element, FeatureKind};
#[cfg(feature = "view")]
pub use pick::{Picked, Picks};
pub use target::OffscreenTarget;
pub use tone::ToneMapping;
pub use value::Decoded;
