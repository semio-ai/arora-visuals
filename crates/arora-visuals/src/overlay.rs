//! What a host needs to draw its own labels and tooltips over the view: where
//! things are on screen, and which gizmo the pointer has.
//!
//! **The renderer reports; the host draws.** In a browser the host is a web
//! application placing DOM tooltips over the canvas; natively it may be Bevy
//! UI drawn by the same app. Both need the same facts, so they are gathered
//! once a frame into an [`OverlayFrame`] — kept as the [`Overlay`] resource for
//! a consumer inside the app, and handed to an [`OverlaySink`] for one outside
//! it, such as a wasm binding forwarding it to JavaScript.
//!
//! **Every name in it is the host's.** Elements go by their instance's id and
//! the ids the asset's `RobotData` declares, with their glTF node's name
//! alongside for a host that knows them by that; gizmos go by the `key` the
//! host gave them. Entity ids mean nothing outside this process, so none
//! appear.
//!
//! **An element anchor serialises flat**: `instance`, `element`, `node`, then
//! the box's `x`, `y`, `min_x`, `min_y`, `max_x` and `max_y` beside them, so a
//! host reading per-element boxes by those field names reads these
//! unchanged.
//!
//! **Positions are boxes, in logical pixels of the camera's target**, which
//! are a canvas's CSS pixels when the canvas is sized in CSS pixels. A box
//! rather than a point because an overlay anchors to an edge: a label wants
//! the top of what it names, or to sit clear of a ring rather than over it.
//! Anything partly behind its camera is left out, since half a projection is
//! nothing to anchor to, and so is a hidden gizmo, which has nothing on screen
//! at all.
//!
//! The pointer half says which gizmo is hovered or held, and for a joint its
//! value, so a host can show the tooltip for the animatable that gizmo drives;
//! where the cursor is, the host already knows.

use std::collections::HashMap;

use bevy::camera::primitives::Aabb;
use bevy::prelude::*;
use serde::Serialize;

use crate::instance::{EntityInstance, Superseded};
use crate::joint_gizmo::{self, JointGizmo, JointMotion, Rest};
use crate::transform_gizmo::{self, Facing, GizmoHandle, GizmoPointer, TransformGizmo};

/// A box on screen, in logical pixels of the camera's target, with its
/// centre.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct ScreenBox {
    pub x: f32,
    pub y: f32,
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

impl ScreenBox {
    /// The box around these points, or nothing for none.
    fn around(points: impl IntoIterator<Item = Vec2>) -> Option<Self> {
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        let mut any = false;
        for point in points {
            min = min.min(point);
            max = max.max(point);
            any = true;
        }
        any.then(|| Self {
            x: (min.x + max.x) / 2.0,
            y: (min.y + max.y) / 2.0,
            min_x: min.x,
            min_y: min.y,
            max_x: max.x,
            max_y: max.y,
        })
    }
}

/// One element's box on screen.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ElementAnchor {
    /// The instance's id.
    pub instance: String,
    /// The element's id, as the asset declares it.
    pub element: String,
    /// The name of the element's glTF node.
    pub node: String,
    #[serde(flatten)]
    pub screen: ScreenBox,
}

/// Which kind of gizmo an anchor or the pointer is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum GizmoKind {
    Transform,
    Joint,
}

/// One gizmo's box on screen: around everything it draws, so a label beside
/// it sits clear of its handles.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GizmoAnchor {
    /// The key the host gave the gizmo.
    pub key: String,
    pub kind: GizmoKind,
    pub screen: ScreenBox,
}

/// The gizmo the pointer has.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GizmoInHand {
    pub key: String,
    pub kind: GizmoKind,
    pub handle: GizmoHandle,
    /// Held in a drag, rather than only under the pointer.
    pub held: bool,
    /// A joint's value, where the gizmo is a joint's.
    pub value: Option<f32>,
}

/// Everything an overlay needs for one frame.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OverlayFrame {
    pub elements: Vec<ElementAnchor>,
    pub gizmos: Vec<GizmoAnchor>,
    pub pointer: Option<GizmoInHand>,
}

/// The latest frame's overlay, for a consumer inside the app.
#[derive(Resource, Default, Debug)]
pub struct Overlay(pub OverlayFrame);

/// Where each frame's overlay goes, for a consumer outside the app.
#[derive(Resource)]
pub struct OverlaySink(pub Box<dyn Fn(&OverlayFrame) + Send + Sync>);

impl OverlaySink {
    pub fn new(sink: impl Fn(&OverlayFrame) + Send + Sync + 'static) -> Self {
        Self(Box::new(sink))
    }
}

pub struct OverlayPlugin;

impl Plugin for OverlayPlugin {
    fn build(&self, app: &mut App) {
        // After the gizmos have placed themselves, so their boxes are where
        // this frame draws them.
        app.init_resource::<Overlay>().add_systems(
            PostUpdate,
            publish
                .after(TransformSystems::Propagate)
                .after(transform_gizmo::TransformGizmoSystems),
        );
    }
}

/// Points around a ring, for its box.
const RING_SAMPLES: usize = 16;

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn publish(
    instances: Query<&EntityInstance, Without<Superseded>>,
    transform_gizmos: Query<(
        Entity,
        &TransformGizmo,
        Option<&Facing>,
        &InheritedVisibility,
    )>,
    joint_gizmos: Query<(Entity, &JointGizmo, Option<&Rest>, &InheritedVisibility)>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    globals: Query<&GlobalTransform>,
    bounds: Query<(&GlobalTransform, &Aabb)>,
    children: Query<&Children>,
    parents: Query<&ChildOf>,
    pointer: Option<Res<GizmoPointer>>,
    sink: Option<Res<OverlaySink>>,
    mut overlay: ResMut<Overlay>,
) {
    let mut frame = OverlayFrame::default();

    for instance in &instances {
        if !instance.bindings.indexed {
            continue;
        }
        let Ok(camera) = cameras.get(instance.camera) else {
            continue;
        };
        let nodes: HashMap<&str, &str> = instance
            .meta
            .elements
            .iter()
            .map(|element| (element.id.as_str(), element.node_name.as_str()))
            .collect();
        for (element, targets) in &instance.bindings.elements {
            // An element with a mesh of its own is that mesh; a group is
            // everything drawn beneath its node.
            let meshes: Vec<Entity> = match targets.mesh {
                Some(mesh) => vec![mesh],
                None => children.iter_descendants(targets.node).collect(),
            };
            if let Some(screen) = box_of_meshes(&meshes, &bounds, camera) {
                frame.elements.push(ElementAnchor {
                    instance: instance.id.clone(),
                    element: element.clone(),
                    node: nodes
                        .get(element.as_str())
                        .copied()
                        .unwrap_or_default()
                        .to_string(),
                    screen,
                });
            }
        }
    }

    let parent_of = |target: Entity| {
        parents
            .get(target)
            .ok()
            .and_then(|child_of| globals.get(child_of.parent()).ok())
            .copied()
            .unwrap_or(GlobalTransform::IDENTITY)
    };
    // A hidden gizmo has nothing on screen to anchor to.
    for (_, gizmo, facing, _) in transform_gizmos.iter().filter(|(.., shown)| shown.get()) {
        let (Ok(camera), Ok(target), Some(facing)) =
            (cameras.get(gizmo.camera), globals.get(gizmo.target), facing)
        else {
            continue;
        };
        let Some(reach) = transform_gizmo::reach_points(gizmo, facing, target, camera) else {
            continue;
        };
        if let Some(screen) = box_of_points(&reach, camera) {
            frame.gizmos.push(GizmoAnchor {
                key: gizmo.key.clone(),
                kind: GizmoKind::Transform,
                screen,
            });
        }
    }
    for (_, gizmo, rest, _) in joint_gizmos.iter().filter(|(.., shown)| shown.get()) {
        let Ok(camera) = cameras.get(gizmo.camera) else {
            continue;
        };
        let Some(reach) = joint_gizmo::reach(gizmo, rest, &parent_of(gizmo.target), camera) else {
            continue;
        };
        let points: Vec<Vec3> = match gizmo.motion {
            JointMotion::Turn => (0..RING_SAMPLES)
                .map(|i| {
                    let angle = i as f32 * std::f32::consts::TAU / RING_SAMPLES as f32;
                    reach.frame.centre
                        + (reach.frame.u * angle.cos() + reach.frame.v * angle.sin())
                            * transform_gizmo::RING_RADIUS
                            * reach.unit
                })
                .collect(),
            JointMotion::Slide => {
                let (from, to) = joint_gizmo::bar_ends(gizmo, &reach.frame);
                vec![from, to]
            }
        };
        if let Some(screen) = box_of_points(&points, camera) {
            frame.gizmos.push(GizmoAnchor {
                key: gizmo.key.clone(),
                kind: GizmoKind::Joint,
                screen,
            });
        }
    }

    if let Some(pointer) = pointer {
        let in_hand = pointer
            .dragging
            .map(|grip| (grip, true))
            .or(pointer.hovered.map(|grip| (grip, false)));
        frame.pointer = in_hand.and_then(|((entity, handle), held)| {
            if let Ok((_, gizmo, ..)) = transform_gizmos.get(entity) {
                return Some(GizmoInHand {
                    key: gizmo.key.clone(),
                    kind: GizmoKind::Transform,
                    handle,
                    held,
                    value: None,
                });
            }
            let (_, gizmo, ..) = joint_gizmos.get(entity).ok()?;
            Some(GizmoInHand {
                key: gizmo.key.clone(),
                kind: GizmoKind::Joint,
                handle,
                held,
                value: Some(gizmo.value),
            })
        });
    }

    if let Some(sink) = sink {
        (sink.0)(&frame);
    }
    if overlay.0 != frame {
        overlay.0 = frame;
    }
}

/// The box around world points, or nothing if any is behind the camera or
/// off its target.
fn box_of_points(points: &[Vec3], (camera, at): (&Camera, &GlobalTransform)) -> Option<ScreenBox> {
    let screen: Option<Vec<Vec2>> = points
        .iter()
        .map(|point| camera.world_to_viewport(at, *point).ok())
        .collect();
    ScreenBox::around(screen?)
}

/// The box around the bounding boxes of these meshes, or nothing if any
/// corner of any is behind the camera.
fn box_of_meshes(
    meshes: &[Entity],
    bounds: &Query<(&GlobalTransform, &Aabb)>,
    camera: (&Camera, &GlobalTransform),
) -> Option<ScreenBox> {
    let mut corners = Vec::new();
    for mesh in meshes {
        let Ok((at, aabb)) = bounds.get(*mesh) else {
            continue;
        };
        let centre = Vec3::from(aabb.center);
        let half = Vec3::from(aabb.half_extents);
        for corner in 0..8u8 {
            let sign = Vec3::new(
                if corner & 1 == 0 { -1.0 } else { 1.0 },
                if corner & 2 == 0 { -1.0 } else { 1.0 },
                if corner & 4 == 0 { -1.0 } else { 1.0 },
            );
            corners.push(at.transform_point(centre + sign * half));
        }
    }
    box_of_points(&corners, camera)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A box is the extent of its points, centred between its edges.
    #[test]
    fn a_box_spans_its_points() {
        let screen = ScreenBox::around([Vec2::new(10.0, 40.0), Vec2::new(30.0, 20.0)]).unwrap();
        assert_eq!(
            screen,
            ScreenBox {
                x: 20.0,
                y: 30.0,
                min_x: 10.0,
                min_y: 20.0,
                max_x: 30.0,
                max_y: 40.0,
            }
        );
        assert!(ScreenBox::around([]).is_none());
    }

    /// What reaches a host serialises with the host's names and no entity
    /// ids: the shape a wasm binding hands to JavaScript.
    #[test]
    fn a_frame_serialises_with_host_names_only() {
        let frame = OverlayFrame {
            elements: vec![ElementAnchor {
                instance: "robot-0".into(),
                element: "head".into(),
                node: "Head".into(),
                screen: ScreenBox {
                    x: 2.0,
                    y: 3.0,
                    min_x: 1.0,
                    min_y: 2.0,
                    max_x: 3.0,
                    max_y: 4.0,
                },
            }],
            gizmos: vec![],
            pointer: Some(GizmoInHand {
                key: "tilt".into(),
                kind: GizmoKind::Joint,
                handle: GizmoHandle::Joint,
                held: true,
                value: Some(0.5),
            }),
        };
        let json = serde_json::to_value(&frame).unwrap();
        assert_eq!(json["elements"][0]["element"], "head");
        assert_eq!(json["elements"][0]["node"], "Head");
        // The box sits beside the names, not nested under its own key.
        for (field, value) in [
            ("x", 2.0),
            ("y", 3.0),
            ("min_x", 1.0),
            ("min_y", 2.0),
            ("max_x", 3.0),
            ("max_y", 4.0),
        ] {
            assert_eq!(json["elements"][0][field], value, "{field}");
        }
        assert!(json["elements"][0].get("screen").is_none());
        assert_eq!(json["pointer"]["key"], "tilt");
        assert_eq!(json["pointer"]["kind"], "Joint");
        assert_eq!(json["pointer"]["value"], 0.5);
    }
}
