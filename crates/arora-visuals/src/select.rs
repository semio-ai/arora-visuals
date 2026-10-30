//! What is selected, and which element meshes a selection covers.
//!
//! Every way of showing a selection reads both from here, so the outline and
//! the feature-edge marking can never disagree about what a selected group
//! includes.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::binding::ElementTargets;
use crate::instance::EntityInstance;

/// The selected elements of each instance, by instance id.
///
/// Kept apart from the instances so a selection survives an instance's
/// reload: what a person selected is their state, not the asset's.
#[derive(Resource, Default)]
pub struct Selections(pub HashMap<String, Vec<String>>);

/// The element meshes one selected element covers: its own mesh, or, for a
/// group with no mesh of its own, the mesh of every element beneath it.
pub fn covered<'a>(
    instance: &'a EntityInstance,
    id: &str,
    parents: &Query<&ChildOf>,
) -> Vec<&'a ElementTargets> {
    let Some(targets) = instance.bindings.elements.get(id) else {
        return Vec::new();
    };
    if targets.mesh.is_some() {
        return vec![targets];
    }
    instance
        .bindings
        .elements
        .values()
        .filter(|element| element.mesh.is_some() && descends(element.node, targets.node, parents))
        .collect()
}

/// Whether `entity` is `ancestor` or lies under it.
fn descends(mut entity: Entity, ancestor: Entity, parents: &Query<&ChildOf>) -> bool {
    loop {
        if entity == ancestor {
            return true;
        }
        match parents.get(entity) {
            Ok(parent) => entity = parent.parent(),
            Err(_) => return false,
        }
    }
}
