//! What is selected, and which element meshes a selection covers.
//!
//! Every way of showing a selection reads both from here, so the outline and
//! the feature-edge marking can never disagree about what a selected group
//! includes.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::binding::ElementTargets;
use crate::instance::EntityInstance;

/// One selected element, and the colour it asks to be shown in, if any.
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    pub element: String,
    pub colour: Option<Color>,
}

impl Selection {
    /// An element selected with no colour of its own.
    pub fn of(element: impl Into<String>) -> Self {
        Self {
            element: element.into(),
            colour: None,
        }
    }
}

/// The selected elements of each instance, by instance id.
///
/// Kept apart from the instances so a selection survives an instance's
/// reload: what a person selected is their state, not the asset's.
#[derive(Resource, Default)]
pub struct Selections(pub HashMap<String, Vec<Selection>>);

impl Selections {
    /// The colour the whole selection is shown in, when it has one: the colour
    /// of the one selected element that brings a colour. With none, or with
    /// several, there is no single colour to choose, and a view falls back to
    /// its own — Studio's rule, which keeps two highlight colours from fighting
    /// over one outline.
    pub fn colour(&self) -> Option<Color> {
        let mut colours = self
            .0
            .values()
            .flatten()
            .filter_map(|selection| selection.colour);
        match (colours.next(), colours.next()) {
            (Some(colour), None) => Some(colour),
            _ => None,
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn selections(colours: &[Option<Color>]) -> Selections {
        let entries = colours
            .iter()
            .enumerate()
            .map(|(index, colour)| Selection {
                element: format!("element-{index}"),
                colour: *colour,
            })
            .collect();
        Selections(HashMap::from([("robot".to_string(), entries)]))
    }

    #[test]
    fn one_coloured_element_colours_the_selection() {
        let red = Color::srgb(1.0, 0.0, 0.0);
        assert_eq!(selections(&[Some(red)]).colour(), Some(red));
        assert_eq!(
            selections(&[None, Some(red), None]).colour(),
            Some(red),
            "uncoloured elements do not compete",
        );
    }

    /// Two colours have no single answer, even when they are the same colour:
    /// the rule counts elements asking, not distinct colours, as Studio does.
    #[test]
    fn several_or_no_colours_leave_the_view_to_choose() {
        let red = Color::srgb(1.0, 0.0, 0.0);
        let blue = Color::srgb(0.0, 0.0, 1.0);
        assert_eq!(selections(&[]).colour(), None);
        assert_eq!(selections(&[None, None]).colour(), None);
        assert_eq!(selections(&[Some(red), Some(blue)]).colour(), None);
        assert_eq!(selections(&[Some(red), Some(red)]).colour(), None);
    }
}
