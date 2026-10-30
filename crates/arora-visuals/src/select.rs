//! What is selected, and which element meshes a selection covers.
//!
//! Every way of showing a selection reads both from here, so the outline and
//! the feature-edge marking can never disagree about what a selected group
//! includes.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::binding::ElementTargets;
use crate::instance::EntityInstance;

/// How a selection asks to be shown. Anything it leaves unset is the view's
/// own, so a selection can ask for a colour alone and keep the view's width.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Highlight {
    pub colour: Option<Color>,
    /// The line's width, in logical pixels.
    pub width: Option<f32>,
    /// How strongly the selection itself is filled, from not at all to solid.
    pub fill: Option<f32>,
}

/// One selected element, and how it asks to be shown, if it asks.
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    pub element: String,
    pub highlight: Option<Highlight>,
}

impl Selection {
    /// An element selected with no highlight of its own.
    pub fn of(element: impl Into<String>) -> Self {
        Self {
            element: element.into(),
            highlight: None,
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
    /// The highlight the whole selection is shown with, when it has one: that
    /// of the one selected element that brings a highlight. With none, or with
    /// several, there is no single highlight to choose, and a view falls back
    /// to its own — Studio's rule, which keeps two highlights from fighting
    /// over one outline.
    pub fn highlight(&self) -> Option<Highlight> {
        let mut asking = self
            .0
            .values()
            .flatten()
            .filter_map(|selection| selection.highlight);
        match (asking.next(), asking.next()) {
            (Some(highlight), None) => Some(highlight),
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

    fn selections(highlights: &[Option<Highlight>]) -> Selections {
        let entries = highlights
            .iter()
            .enumerate()
            .map(|(index, highlight)| Selection {
                element: format!("element-{index}"),
                highlight: *highlight,
            })
            .collect();
        Selections(HashMap::from([("robot".to_string(), entries)]))
    }

    fn coloured(colour: Color) -> Option<Highlight> {
        Some(Highlight {
            colour: Some(colour),
            ..default()
        })
    }

    #[test]
    fn one_highlighted_element_highlights_the_selection() {
        let red = coloured(Color::srgb(1.0, 0.0, 0.0));
        assert_eq!(selections(&[red]).highlight(), red);
        assert_eq!(
            selections(&[None, red, None]).highlight(),
            red,
            "elements that bring no highlight do not compete",
        );
    }

    /// Two highlights have no single answer, even when they are the same: the
    /// rule counts elements asking, not distinct highlights, as Studio does.
    #[test]
    fn several_or_no_highlights_leave_the_view_to_choose() {
        let red = coloured(Color::srgb(1.0, 0.0, 0.0));
        let blue = coloured(Color::srgb(0.0, 0.0, 1.0));
        assert_eq!(selections(&[]).highlight(), None);
        assert_eq!(selections(&[None, None]).highlight(), None);
        assert_eq!(selections(&[red, blue]).highlight(), None);
        assert_eq!(selections(&[red, red]).highlight(), None);
    }
}
