//! Rig outputs held against the device — the authoring inspector's lock.
//!
//! A held animatable keeps what the view shows while its device writes on;
//! a held component keeps its value while the animatable's other components
//! follow the device. Components are `x`, `y`, `z` of a vector or an euler,
//! or `r`, `g`, `b` of a colour (`r` is `x`, `g` is `y`, `b` is `z`), and
//! hold nothing on a scalar output.
//!
//! **A hold is the view declining a write, not the device pausing.** The
//! device and its store run on untouched, so a release shows the device's
//! current value at once rather than resuming from where the hold began.
//! That is the whole semantic, and it is shared because any product that
//! edits a live asset needs exactly it — two implementations of a lock would
//! disagree about this one point first, and the disagreement would only show
//! up as values jumping on release.
//!
//! Holds are kept per instance and survive that instance's reloads: what a
//! held component keeps is what the scene shows, and a reloaded scene shows
//! its GLB's values until the device writes.

use std::collections::HashMap;
use std::str::FromStr;

use bevy_ecs::resource::Resource;

use crate::value::Decoded;

/// One held output: an animatable, or one component of it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HoldTarget {
    /// The animatable's id: any string, as its RobotData names it.
    pub animatable: String,
    /// The component held: 0 (`x`/`r`), 1 (`y`/`g`) or 2 (`z`/`b`); `None`
    /// holds the whole output.
    pub component: Option<usize>,
}

impl FromStr for HoldTarget {
    type Err = String;

    /// `<animatable id>` or `<animatable id>:<component>`, the id bare or as
    /// the rig path it is written under (`rig/<instance>/<id>`): the path's
    /// last `/`-segment, less a final `:x`, `:y`, `:z`, `:r`, `:g` or `:b`,
    /// which names the component. An id is any other non-empty string.
    fn from_str(path: &str) -> Result<Self, Self::Err> {
        let last = path.rsplit('/').next().unwrap_or(path).trim();
        let (id, component) = match last.rsplit_once(':') {
            Some((id, "x" | "r")) => (id, Some(0)),
            Some((id, "y" | "g")) => (id, Some(1)),
            Some((id, "z" | "b")) => (id, Some(2)),
            _ => (last, None),
        };
        if id.is_empty() {
            return Err(format!("{path:?} names no animatable id"));
        }
        Ok(Self {
            animatable: id.to_string(),
            component,
        })
    }
}

/// What is held of one animatable: the whole output, and each component on
/// its own — held and released independently, as the inspector locks them.
///
/// Deliberately opaque. A caller gets one from [`Holds::of`] and gives it to
/// [`resolve`]; there is nothing else to do with it, because deciding what a
/// hold means is the point of this module rather than the caller's business.
/// If a consumer needs to *show* a lock, that is a reason to add an accessor
/// for the thing it shows, not a reason for the fields to have been public
/// all along.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Held {
    whole: bool,
    components: [bool; 3],
}

impl Held {
    fn is_empty(&self) -> bool {
        !self.whole && !self.components.contains(&true)
    }
}

/// The holds of every instance, by instance id.
#[derive(Resource, Default)]
pub struct Holds(HashMap<String, HashMap<String, Held>>);

impl Holds {
    pub fn hold(&mut self, instance: &str, targets: Vec<HoldTarget>) {
        let held_here = self.0.entry(instance.to_string()).or_default();
        for target in targets {
            let held = held_here.entry(target.animatable).or_default();
            match target.component {
                Some(i) => held.components[i] = true,
                None => held.whole = true,
            }
        }
    }

    /// Release `targets`, or every hold of the instance given `None`.
    pub fn release(&mut self, instance: &str, targets: Option<Vec<HoldTarget>>) {
        let Some(targets) = targets else {
            self.0.remove(instance);
            return;
        };
        let Some(held_here) = self.0.get_mut(instance) else {
            return;
        };
        for target in targets {
            if let Some(held) = held_here.get_mut(&target.animatable) {
                match target.component {
                    Some(i) => held.components[i] = false,
                    None => held.whole = false,
                }
                if held.is_empty() {
                    held_here.remove(&target.animatable);
                }
            }
        }
        if held_here.is_empty() {
            self.0.remove(instance);
        }
    }

    pub fn clear(&mut self, instance: &str) {
        self.0.remove(instance);
    }

    /// The instance's holds, `None` when it holds nothing.
    pub fn of(&self, instance: &str) -> Option<&HashMap<String, Held>> {
        self.0.get(instance)
    }
}

/// What the scene takes of the device's `value` for animatable `id` under
/// the instance's holds: the value itself when nothing of it is held, `None`
/// when it is held whole, and, with components held, the value with those
/// components taken from what the scene `shown`s — or `None` when the scene
/// shows nothing to take them from.
pub fn resolve(
    holds: &HashMap<String, Held>,
    id: &str,
    value: Decoded,
    shown: impl FnOnce() -> Option<Decoded>,
) -> Option<Decoded> {
    let Some(held) = holds.get(id) else {
        return Some(value);
    };
    if held.whole {
        return None;
    }
    match value {
        Decoded::Scalar(_) => Some(value),
        Decoded::Triple(mut v) => {
            let Some(Decoded::Triple(kept)) = shown() else {
                return None;
            };
            for (i, held) in held.components.iter().enumerate() {
                if *held {
                    v[i] = kept[i];
                }
            }
            Some(Decoded::Triple(v))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "6b8f0c1e-3f6a-4c55-9d1e-2b7a4c0e9f11";

    fn target(path: &str) -> HoldTarget {
        path.parse().expect("a hold target")
    }

    #[test]
    fn a_target_names_an_animatable_and_maybe_a_component() {
        let id = ID.to_string();
        assert_eq!(
            target(ID),
            HoldTarget {
                animatable: id.clone(),
                component: None
            }
        );
        assert_eq!(target(&format!("{ID}:x")).component, Some(0));
        assert_eq!(target(&format!("{ID}:g")).component, Some(1));
        assert_eq!(target(&format!("{ID}:b")).component, Some(2));
        let rig_path = target(&format!("rig/an-instance/{ID}:z"));
        assert_eq!((rig_path.animatable, rig_path.component), (id, Some(2)));
        // Any string is an id: a plain one, and one whose last `:` names no
        // component.
        assert_eq!(target("left_eye").animatable, "left_eye");
        assert_eq!(
            (
                target("rig/f/left_eye:x").animatable,
                target("left_eye:x").component
            ),
            ("left_eye".to_string(), Some(0))
        );
        assert_eq!(target(&format!("{ID}:w")).animatable, format!("{ID}:w"));
        assert!("rig/f/".parse::<HoldTarget>().is_err());
        assert!(":x".parse::<HoldTarget>().is_err());
    }

    #[test]
    fn a_whole_hold_declines_every_write_and_releases() {
        let id = ID;
        let mut holds = Holds::default();
        holds.hold("an-instance", vec![target(ID)]);
        let held = holds.of("an-instance").unwrap();
        assert_eq!(
            resolve(held, id, Decoded::Scalar(0.5), || None),
            None,
            "a held scalar"
        );
        assert_eq!(
            resolve(held, id, Decoded::Triple([1.0, 2.0, 3.0]), || None),
            None
        );
        holds.release("an-instance", Some(vec![target(ID)]));
        assert!(holds.of("an-instance").is_none(), "nothing left held");
    }

    #[test]
    fn a_held_component_keeps_what_the_scene_shows() {
        let id = ID;
        let mut holds = Holds::default();
        holds.hold("an-instance", vec![target(&format!("{ID}:y"))]);
        let held = holds.of("an-instance").unwrap();
        let shown = || Some(Decoded::Triple([0.0, 9.0, 0.0]));
        assert_eq!(
            resolve(held, id, Decoded::Triple([1.0, 2.0, 3.0]), shown),
            Some(Decoded::Triple([1.0, 9.0, 3.0]))
        );
        // Nothing shown to keep it from: the write is declined whole.
        assert_eq!(
            resolve(held, id, Decoded::Triple([1.0, 2.0, 3.0]), || None),
            None
        );
        // A component holds nothing on a scalar output.
        assert_eq!(
            resolve(held, id, Decoded::Scalar(0.25), shown),
            Some(Decoded::Scalar(0.25))
        );
        // Another animatable is free.
        assert_eq!(
            resolve(held, "another", Decoded::Scalar(0.25), shown),
            Some(Decoded::Scalar(0.25))
        );
    }

    /// The whole output and its components are held and released on their
    /// own, as the inspector's locks are.
    #[test]
    fn whole_and_component_holds_are_independent() {
        let id = ID;
        let mut holds = Holds::default();
        holds.hold("an-instance", vec![target(ID), target(&format!("{ID}:x"))]);
        holds.release("an-instance", Some(vec![target(ID)]));
        let held = holds.of("an-instance").expect("the component is still held");
        assert_eq!(
            resolve(held, id, Decoded::Triple([1.0, 2.0, 3.0]), || Some(
                Decoded::Triple([7.0, 7.0, 7.0])
            )),
            Some(Decoded::Triple([7.0, 2.0, 3.0]))
        );
        holds.release("an-instance", None);
        assert!(holds.of("an-instance").is_none());
    }
}
