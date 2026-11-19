//! Optional passes a `.glb` goes through when an asset is imported, before
//! any renderer loads it.
//!
//! Each pass rewrites the file's mesh data in place and reports what it
//! did; the JSON chunk is left byte-for-byte, so a pass never changes the
//! document's structure, names or extensions.
//!
//! - [`optimize_morph_triangles`] chooses, for each part modelled as one
//!   large polygon, the triangles that stay unfolded across its morph
//!   targets ([`morph_triangles`] explains why exported triangles fold).
//!
//! ```no_run
//! use arora_import::Mode;
//! let bytes = std::fs::read("asset.glb")?;
//! let optimized = arora_import::optimize_morph_triangles(&bytes, Mode::Safe)?;
//! if let Some(rewritten) = optimized.bytes {
//!     std::fs::write("asset.glb", rewritten)?;
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod glb;
mod gltf;
pub mod morph_triangles;

use glb::{Glb, CHUNK_BIN};
use thiserror::Error;

pub use glb::GlbError;
pub use morph_triangles::{Mode, Report};

/// Why a file could not go through a pass at all. Problems confined to one
/// mesh are reported in that pass's report instead, and the mesh is left
/// as it was.
#[derive(Debug, Error)]
pub enum ImportError {
    #[error(transparent)]
    Glb(#[from] GlbError),
    #[error("JSON chunk: {0}")]
    Json(#[from] serde_json::Error),
}

/// The result of a pass: the rewritten file when the pass changed
/// anything, and what it did.
#[derive(Debug, Clone)]
pub struct Optimized {
    pub bytes: Option<Vec<u8>>,
    pub report: Report,
}

/// Replaces the triangles of every part modelled as one large polygon with
/// the triangulation of the same outline that folds least across the
/// mesh's morph targets. Vertices, the JSON chunk and every other mesh are
/// unchanged; see [`morph_triangles`] for which parts qualify.
pub fn optimize_morph_triangles(glb: &[u8], mode: Mode) -> Result<Optimized, ImportError> {
    let mut container = Glb::parse(glb)?;
    let json: serde_json::Value = serde_json::from_slice(&container.json)?;
    let Some(bin) = container.tail.iter_mut().find(|c| c.kind == CHUNK_BIN) else {
        return Ok(Optimized {
            bytes: None,
            report: Report::default(),
        });
    };
    let report = morph_triangles::optimize(&json, &mut bin.data, mode);
    let bytes = report.changed().then(|| container.to_bytes());
    Ok(Optimized { bytes, report })
}
