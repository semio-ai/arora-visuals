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
//! controllable, instance and view. A type or a field named for a face has
//! already decided which consumer it serves.
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

pub mod hold;
pub mod meta;
#[cfg(feature = "snapshot")]
pub mod snapshot;
pub mod target;
pub mod tone;
pub mod value;

pub use hold::{HoldTarget, Holds};
pub use meta::{glb_json_chunk, AssetMeta, Binding, Element, FeatureKind};
pub use target::OffscreenTarget;
pub use tone::ToneMapping;
pub use value::Decoded;
