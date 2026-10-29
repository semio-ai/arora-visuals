//! A feature value in the form a renderer applies it.
//!
//! A device writes values in its own vocabulary; a renderer applies two
//! shapes and only two. [`Decoded`] is the narrowing between them, and it is
//! shared because both the narrowing and the shapes are the same whatever is
//! being drawn — a vector, an euler and a colour are one triple here, because
//! nothing downstream of the decode treats them differently.

/// A feature value as a renderer takes it: a triple (a vector, an euler, a
/// linear RGB colour) or a scalar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decoded {
    Triple([f32; 3]),
    Scalar(f32),
}
