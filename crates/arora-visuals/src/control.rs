//! Controls a driver adds, shows, hides, moves and removes, by keys of its own.
//!
//! A view does not decide what a control means. A driver (a device, an IK
//! solver, an editor) adds a control under a key, hears each drag on it as a
//! [`ControlEvent`], and writes back whatever the drag should change. **A drag
//! moves nothing by itself**: a joint control reports the value the drag asks
//! for, and a transform control reports the pose, and the instance moves when
//! the driver writes that value or pose. One path changes the view, and it is
//! the one every other writer uses, so the data a driver keeps and what the
//! view shows cannot disagree.
//!
//! - A **transform control** moves an instance as a whole, one of its
//!   elements, or a goal: a pose with nothing drawn but the control itself,
//!   which is what an IK target is.
//! - A **joint control** is keyed by its joint's value animatable, the key the
//!   driver writes the joint under.
//!
//! **A drag reports a lifecycle a driver can build an edit on.** Each has one
//! `start`, a `move` for each frame its value changes, and exactly one
//! closing event: `end` when the pointer lets go, `cancel` when the drag is
//! abandoned instead — by the host ([`GizmoDrags::cancel_drag`]), or by the
//! control being removed, replaced, hidden or made not `interactive` while
//! the button is down. Hidden counts however the gizmo comes not to show:
//! by its driver or a ghosting, or by anything above it in the hierarchy,
//! such as a host hiding the instance the gizmo sits under; the drag is
//! cancelled in the frame after the one it is hidden in. Nothing follows
//! the closing event. Every event of a drag carries the same `gesture`, a
//! number each drag takes in turn, and the [`Modifiers`] the host says are
//! held. A `cancel` carries no value: a driver drops what the drag
//! previewed, and keeps nothing.
//!
//! Every host takes the same commands, through [`ControlCommands`], under the
//! same rules, and reports the same events. What differs between hosts is
//! told to this module rather than decided by it, in four places.
//!
//! **The controls are the app's, not a host's.** [`Controls`],
//! [`ControlCommands`], [`ControlEvents`] and [`ControlPresses`] are one each
//! per app, so hosts sharing an app — a robot scene and the face on its
//! screen — share one key space: an add under a key another host used
//! replaces that host's control. The host that embeds the others is the one
//! that drains the events and presses, and it names keys so they do not
//! collide, as it routes the commands that carry them.
//!
//! # The driver's frame: [`DriverFrame`]
//!
//! **Poses are in the driver's frame: Z up, metres, quaternions as
//! `[x, y, z, w]`.** A world's own axes are its host's: a Z-up world is
//! already the driver's frame, and a Y-up one is a quarter turn from it. The
//! host says which by a [`DriverFrame`], the turn from its world to the
//! driver's frame.
//!
//! **The frame belongs to a view, not to the app.** Two hosts can share one
//! app — a Y-up robot scene and a Z-up face on its screen — so one app-wide
//! frame would turn one of them wrongly. A host puts [`DriverFrame`] on the
//! camera its controls are used through (below), and a control speaks in the
//! frame of its camera. The [`DefaultDriverFrame`] resource, identity unless
//! a host inserts one, is the frame of a camera that carries none. [`Pose`]
//! converts at this boundary, and a transform control's handles are labelled
//! from the same turn, so a driver never sees the world's own axes.
//!
//! # What is still to come: [`Loading`](crate::instance::Loading)
//!
//! A control on an instance that is still loading waits for it, and one on an
//! instance nothing is loading is refused. Only the host knows which is
//! which, so it says: it puts [`Loading`](crate::instance::Loading) on the entity it is loading an
//! instance into until that entity carries its [`EntityInstance`].
//!
//! An instance that is there but not yet indexed
//! ([`SceneBindings::indexed`](crate::binding::SceneBindings::indexed)) is
//! still to come as well: until it is, an element it lacks cannot be told
//! from one not yet joined, so a control on it waits rather than being
//! refused. A host that spawns an instance and indexes it later needs no
//! [`Loading`](crate::instance::Loading) for that gap.
//!
//! # Which camera a control is used through: [`ControlCamera`]
//!
//! A control on an instance is used through that instance's own
//! [`camera`](EntityInstance::camera), so a face drawn by a camera of its own
//! has its controls sized and picked through that camera. A goal is on no
//! instance, so it names the view it is used through: `view` on
//! [`Target::Goal`], the camera carrying a [`ControlView`] by that name.
//! Naming none, it is used through the one camera the host marks
//! [`ControlCamera`]. A goal waits while no camera answers — the view it
//! names is not there yet, or none is marked — and is refused when it names
//! none and several are marked, since taking one would be a guess at which
//! view the driver meant. The camera also decides which [`DriverFrame`] the
//! control's poses are in.
//!
//! # How big a control is: [`ControlSize`]
//!
//! A control's size is one of its properties: an add or a `configure` may
//! give it (`size`), in pixels on screen or metres in the world. A control
//! given none takes its view's: the [`ControlSize`] on the camera it is used
//! through, which a host sets where its world is not robot-sized — a face
//! authored tens of units across, say — so that every driver's controls are
//! sized for it without each knowing the world's units. With neither, a
//! transform control is [`GIZMO_SIZE`](crate::transform_gizmo::GIZMO_SIZE)
//! and a joint control what its host resolved it at.
//!
//! # Joint controls: [`JointRequest`]
//!
//! What a joint is — which body it moves, about which axis, between which
//! limits, standing at what value now — is read from a host's rig, which this
//! crate never sees. So a joint add is resolved by the host, in the middle of
//! the command passes, while everything around it stays here:
//!
//! 1. The add spawns an entity carrying a [`JointRequest`], placed under its
//!    key like any other control. Keys, order, removal, showing, configuring
//!    and events all act on that entity as on any other control's.
//! 2. In [`ControlSystems::Joints`], in the [`ControlPass`] schedule — added
//!    with `app.add_systems(ControlPass, resolver.in_set(ControlSystems::Joints))`
//!    — the host's resolver puts a [`JointGizmo`] on each request entity it
//!    can resolve, built from its rig and the request's
//!    [`camera`](JointRequest::camera). It runs after every pass, so a joint
//!    added in any pass is resolved before the next.
//! 3. Then this module finishes it: names it by its key, makes it report
//!    rather than move, and gives it the colour the add asked for. A request
//!    nothing resolved is dropped, and said in the log — which is all a host
//!    with no rig, and so no resolver, has to do with joint adds.
//!
//! A request rather than a callback, because the resolver then is an ordinary
//! system with whatever queries the host's rig needs, and the spawn timing
//! every other control has — a change sent with an add holds from the frame
//! the control first shows — is kept by ordering alone.
//!
//! # The rules
//!
//! - **Commands under one key act in the order they were sent.** A command
//!   that waits on what is already asked for — an add for its instance or
//!   camera, a change for the gizmo its control's add is spawning — holds
//!   every later command under its key behind it, so a later add is the
//!   control left, and a change acts on the control the add before it
//!   makes, not the one that replaces it. The one reordering is a change
//!   sent before any add makes its control: it waits for the next add under
//!   its key that lands, acts on the control that add makes, and acts
//!   before anything sent after that add. How a key is held is spelled out
//!   on `Hold`, in the source.
//! - **A remove is not held**: it cancels a waiting add and what waits behind
//!   it, and takes away the control the add would have replaced.
//! - **A refused add changes nothing, and takes its changes with it.** An add
//!   on an instance nothing is loading, an element its instance lacks, a goal
//!   at something that is not a pose ([`Pose::is_valid`]) or a goal whose
//!   view is ambiguous is refused, and said in the log; the control it would
//!   have replaced stays. A joint its host cannot resolve fails the same way
//!   once its pass has ended: a joint add replaces the control under its key
//!   only when its host resolves it, so until then that control is kept, and
//!   on a failure it stays. Either way the changes that were the add's —
//!   sent before it and waiting for it, or after it up to the next add or
//!   remove under its key — are dropped, rather than landing on the next
//!   add's control.
//! - **A pose that is not one moves nothing**: a pose set with a value that
//!   is not a number or a zero rotation is refused, as is one for a target
//!   with no transform to set.
//! - **A control that does not show takes no input.** Hidden by its driver
//!   or by a ghosting, its gizmo takes no press from the frame after the one
//!   it is hidden in, and a drag in hand is cancelled, as one is when the
//!   control is made not `interactive`. A press in the frame it is hidden in can
//!   still be reported: a press with another button than the primary is
//!   taken in `PreUpdate`, before this frame's commands, and so was made on
//!   a control that showed when it was made. What the driver last asked of
//!   `interactive` is kept, and holds again once the control shows.
//! - **A key names one control in the view**; an add under a key in use
//!   replaces its control.
//! - A change for a key — a show, a pose, a configure, a highlight — waits for
//!   its control's gizmo, which an add spawns at the end of its pass, or for
//!   an add still to come. A change for a key nothing will add is dropped, so
//!   it does not ask for frames for good.
//! - **A change sent with its control's add takes hold the frame the control
//!   first shows**, for any batch that sends up to seven adds under one key,
//!   each with its changes (six replacements). Commands are carried out in
//!   passes, each [`ControlPass`], and the gizmos a pass spawns are there
//!   for the next; a frame runs another pass while the last took something
//!   off the queue and left a change for a control now placed, up to eight
//!   passes a frame.
//! - **Past that limit**, a batch chaining more replacements under one key
//!   than a frame's passes carry out lands over more than one frame: the
//!   control placed by the frame's last pass draws one frame as added — at
//!   full opacity and live, say — before the changes sent with it land in
//!   the next, and a warning is logged. Order under the key is kept
//!   throughout. A driver sending one add per control, as drivers do, never
//!   comes near it.
//! - **A control whose gizmo, instance, target or camera is gone is
//!   forgotten**, at the start of each pass: what is left of it goes, its
//!   goal included, and its key is free again, so what is sent under the key
//!   next acts as for a key with nothing placed. This does not wait on a host
//!   to take a control's gizmo with its instance, or its goal with its gizmo.
//! - **Only a control's own gizmo reports as it.** A drag or a press is
//!   reported when [`Controls`] has its gizmo under its key; a gizmo a host
//!   spawned itself reports nothing here, whatever key it carries.
//! - **A frame is asked for only when a command changed the scene**, or
//!   waits on a gizmo spawned in its pass. An add waiting for an instance or
//!   a camera asks for none: the host's own change brings the frame it
//!   lands in.

use std::collections::{HashMap, HashSet};

use bevy::ecs::entity::Entities;
use bevy::picking::pointer::PointerButton;
use bevy::prelude::*;
use bevy::window::RequestRedraw;
use serde::{Deserialize, Serialize};

use crate::ghost::Ghost;
use crate::instance::{EntityInstance, InstanceIndex, InstanceSystems};
use crate::joint_gizmo::{JointDrag, JointGizmo};
use crate::transform_gizmo::{
    current_global, DragPhase, GizmoDrag, GizmoDrags, GizmoHandle, GizmoPress, GizmoSize,
    GizmoSpace, TransformGizmo, TransformGizmoSystems, TransformModes,
};

/// A pose as a driver sees it: Z up, in metres, with the rotation as a
/// quaternion `[x, y, z, w]`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    pub position: [f32; 3],
    #[serde(default = "no_turn")]
    pub rotation: [f32; 4],
}

fn no_turn() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

/// The turn that carries a world's axes onto the driver's: Z up, X and Y
/// level. Identity, the default, is a world that is Z up already.
///
/// The host's to say, because the world's axes are its own; this crate holds
/// no up-axis of its own. It is said per view, on the camera the controls
/// are used through, because two hosts sharing an app can draw worlds that
/// stand different ways up. A camera without one speaks in
/// [`DefaultDriverFrame`] ([module docs](self#the-drivers-frame-driverframe)).
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct DriverFrame(pub Quat);

impl Default for DriverFrame {
    fn default() -> Self {
        Self(Quat::IDENTITY)
    }
}

/// The [`DriverFrame`] of a camera that carries none: identity, a Z-up world,
/// unless a host inserts another. A host that is alone in its app can say its
/// frame once here rather than on each camera.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct DefaultDriverFrame(pub DriverFrame);

impl DriverFrame {
    /// The frame of a world that is Bevy's own Y up: a quarter turn about X
    /// takes Y to Z and Z to -Y.
    pub fn y_up() -> Self {
        Self(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2))
    }

    /// The directions in the world the driver calls X, Y and Z, which is how a
    /// transform control's handles are labelled.
    pub fn axes(&self) -> [Dir3; 3] {
        let back = self.0.inverse();
        [Vec3::X, Vec3::Y, Vec3::Z].map(|axis| Dir3::new(back * axis).unwrap_or(Dir3::Z))
    }

    /// The frame a control used through `camera` speaks in: the camera's own,
    /// or the app's default when it carries none.
    fn of(camera: Entity, frames: &Query<&DriverFrame>, fallback: &DefaultDriverFrame) -> Self {
        frames.get(camera).copied().unwrap_or(fallback.0)
    }
}

impl Pose {
    /// Whether something can stand at this pose: every value a number, and a
    /// rotation that is a turn. The rotation is judged by what normalizing it
    /// gives, since that is what [`to_world`](Self::to_world) uses: the zero
    /// quaternion normalizes to nothing finite, and one with components so
    /// large its length overflows normalizes to zero. Either, or a position
    /// not finite, gives a transform the renderer cannot draw, so a command
    /// carrying such a pose is refused rather than carried out.
    pub fn is_valid(&self) -> bool {
        let turn = Quat::from_array(self.rotation).normalize();
        self.position.iter().all(|value| value.is_finite())
            && turn.is_finite()
            && (turn.length() - 1.0).abs() < 1e-3
    }

    /// Where `transform`, in the world, stands as the driver sees it.
    pub fn from_world(transform: &Transform, frame: &DriverFrame) -> Self {
        let c = frame.0;
        Self {
            position: (c * transform.translation).to_array(),
            rotation: (c * transform.rotation * c.inverse()).to_array(),
        }
    }

    /// This pose in the world.
    pub fn to_world(&self, frame: &DriverFrame) -> Transform {
        let c = frame.0.inverse();
        Transform {
            translation: c * Vec3::from(self.position),
            rotation: c * Quat::from_array(self.rotation).normalize() * c.inverse(),
            scale: Vec3::ONE,
        }
    }
}

/// What a transform control moves.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Target {
    /// An instance as a whole: the entity carrying its [`EntityInstance`], by
    /// the host's id for it.
    Instance { instance: String },
    /// One element of an instance, by the id `RobotData` gives it.
    Element { instance: String, id: String },
    /// A goal standing at `pose`, in the world, with nothing drawn but the
    /// control.
    Goal {
        pose: Pose,
        /// The view the goal is used through: the camera carrying a
        /// [`ControlView`] by this name. Absent, the one camera marked
        /// [`ControlCamera`]
        /// ([module docs](self#which-camera-a-control-is-used-through-controlcamera)).
        #[serde(default)]
        view: Option<String>,
    },
}

impl Target {
    /// The instance the target is on, if it is on one.
    pub fn instance(&self) -> Option<&str> {
        match self {
            Target::Instance { instance } | Target::Element { instance, .. } => Some(instance),
            Target::Goal { .. } => None,
        }
    }
}

/// What a transform control's handles can do.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Mode {
    Translate,
    Rotate,
    Scale,
}

/// A control to add.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Control {
    /// A ring or bar on an instance's joint, resolved by the host
    /// ([module docs](self#joint-controls-jointrequest)).
    Joint {
        /// The host's id for the instance.
        instance: String,
        /// The joint, by the animatable that drives its value.
        joint: String,
        /// A CSS hex colour; the joint's own when absent.
        #[serde(default)]
        color: Option<String>,
        /// Its size; its view's when absent ([`ControlSize`]).
        #[serde(default)]
        size: Option<GizmoSize>,
    },
    Transform {
        target: Target,
        /// Every mode when absent.
        #[serde(default)]
        modes: Option<Vec<Mode>>,
        /// Handles along the target's own axes rather than the world's.
        #[serde(default)]
        local: bool,
        /// The X, Y and Z handles' CSS hex colours; red, green and blue when
        /// absent.
        #[serde(default)]
        colors: Option<[String; 3]>,
        /// Its size; its view's when absent ([`ControlSize`]).
        #[serde(default)]
        size: Option<GizmoSize>,
        /// Turn each arrow to the side of its axis facing the camera; on the
        /// positive side when absent
        /// ([`TransformGizmo::face_viewer`]).
        #[serde(default, rename = "faceViewer")]
        face_viewer: bool,
    },
}

impl Control {
    /// The instance the control is on, if it is on one.
    pub fn instance(&self) -> Option<&str> {
        match self {
            Control::Joint { instance, .. } => Some(instance),
            Control::Transform { target, .. } => target.instance(),
        }
    }
}

/// A driver's instruction about one control.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "op", rename_all = "camelCase")]
pub enum ControlCommand {
    Add {
        key: String,
        control: Control,
    },
    Remove {
        key: String,
    },
    Show {
        key: String,
        visible: bool,
    },
    /// Moves a transform control's target to `pose`, in the world, and, given
    /// `scale`, stretches it along its own axes: what a driver writes back
    /// after a drag on the move, turn or scale handles.
    SetPose {
        key: String,
        pose: Pose,
        #[serde(default)]
        scale: Option<[f32; 3]>,
    },
    /// Changes how a control looks or works, in place: a joint control's
    /// `color`, a transform control's `colors`, whether it is `local` and
    /// whether it `faceViewer`s, and
    /// for either its `size`, its `opacity` and whether it is `interactive`. What is
    /// absent stays as it is, and what the control's kind does not take is
    /// ignored, so one command can recolour controls of both kinds.
    ///
    /// `opacity`, from 0 to 1, is faded to evenly over [`FADE_SECS`] (longer
    /// while frames come slower than thirty a second), so a host sends the
    /// value it wants once. It is how the control looks only: a faded control
    /// still takes the pointer. A control that is not `interactive` takes no
    /// hover, press or drag, the pointer reaches the instance and the view
    /// behind it, and a drag in hand ends; a host's highlight still shows on
    /// it.
    Configure {
        key: String,
        #[serde(default)]
        color: Option<String>,
        #[serde(default)]
        colors: Option<[String; 3]>,
        #[serde(default)]
        local: Option<bool>,
        #[serde(default)]
        opacity: Option<f32>,
        #[serde(default)]
        interactive: Option<bool>,
        /// Its size from now on, in place of its view's ([`ControlSize`]).
        #[serde(default)]
        size: Option<GizmoSize>,
        #[serde(default, rename = "faceViewer")]
        face_viewer: Option<bool>,
    },
    /// Draws a control as the one under the pointer is drawn, or stops: a
    /// host pointing at it from elsewhere. `handle` names one handle of a
    /// transform control, in the form the overlay gives it; the whole control
    /// when absent.
    Highlight {
        key: String,
        #[serde(default)]
        handle: Option<GizmoHandle>,
        on: bool,
    },
}

impl ControlCommand {
    /// The key the command is about.
    pub fn key(&self) -> &str {
        match self {
            Self::Add { key, .. }
            | Self::Remove { key }
            | Self::Show { key, .. }
            | Self::SetPose { key, .. }
            | Self::Configure { key, .. }
            | Self::Highlight { key, .. } => key,
        }
    }
}

/// Instructions waiting for the next frame. One that cannot be carried out
/// yet, because its instance has not arrived, waits for the frame that can.
#[derive(Resource, Default)]
pub struct ControlCommands(pub Vec<ControlCommand>);

impl ControlCommands {
    /// Drops the adds waiting for `instance`, for an instance a host has taken
    /// away before it arrived, with every change that is theirs: they would
    /// otherwise wait for a control that never comes, or land on another's.
    ///
    /// A change is an add's when it follows the add under its key, up to the
    /// next add or remove, and when it was sent before the add with no
    /// control for it to act on: none placed under the key in `controls` or
    /// added before it, since the key's last remove — such a change waits
    /// for the add. A change sent while a control is placed under the key is
    /// that control's, and stays. A later add under the key is another
    /// control's and stays, as does a remove, which is never held: what
    /// follows a remove is not the forgotten add's.
    pub fn forget_instance(&mut self, instance: &str, controls: &Controls) {
        self.forget_adds(controls, |control| control.instance() == Some(instance));
    }

    /// Drops the goal adds waiting for the view named `view`, for a view a
    /// host has taken away before they were placed, with every change that is
    /// theirs, by the rule [`forget_instance`](Self::forget_instance) gives.
    /// A goal already placed through it goes with its camera.
    pub fn forget_view(&mut self, view: &str, controls: &Controls) {
        self.forget_adds(controls, |control| {
            matches!(
                control,
                Control::Transform {
                    target: Target::Goal { view: Some(named), .. },
                    ..
                } if named == view
            )
        });
    }

    /// Drops the adds `forgotten` picks, with every change that is theirs.
    fn forget_adds(&mut self, controls: &Controls, forgotten: impl Fn(&Control) -> bool) {
        /// One key's commands, as far as the walk has read them.
        struct Run {
            /// Changes since the key's last remove with no control before
            /// them: the next add's, whichever it is.
            loose: Vec<usize>,
            /// Whether a control — placed, or added before — is there for what
            /// comes next to act on, which is then that control's.
            on_control: bool,
            /// Whether what comes next is a forgotten add's.
            dropping: bool,
        }
        let mut keep = vec![true; self.0.len()];
        let mut runs: HashMap<&str, Run> = HashMap::new();
        for (at, command) in self.0.iter().enumerate() {
            let key = command.key();
            let run = runs.entry(key).or_insert_with(|| Run {
                loose: Vec::new(),
                on_control: controls.0.contains_key(key),
                dropping: false,
            });
            match command {
                ControlCommand::Add { control, .. } if forgotten(control) => {
                    keep[at] = false;
                    for loose in run.loose.drain(..) {
                        keep[loose] = false;
                    }
                    run.dropping = true;
                    run.on_control = false;
                }
                ControlCommand::Add { .. } => {
                    run.loose.clear();
                    run.dropping = false;
                    run.on_control = true;
                }
                ControlCommand::Remove { .. } => {
                    run.loose.clear();
                    run.dropping = false;
                    run.on_control = false;
                }
                _ if run.dropping => keep[at] = false,
                _ if !run.on_control => run.loose.push(at),
                _ => {}
            }
        }
        let mut keep = keep.into_iter();
        self.0.retain(|_| keep.next().unwrap_or(true));
    }

    /// Drops the changes under `key` that come before its next add or
    /// remove: when an add placed under `key` fails after its pass, the
    /// changes still waiting under the key are the ones held behind it, and
    /// were its.
    fn drop_changes_ahead_of_next_add(&mut self, key: &str) {
        let mut ahead = true;
        self.0.retain(|command| {
            if command.key() != key || !ahead {
                return true;
            }
            if matches!(
                command,
                ControlCommand::Add { .. } | ControlCommand::Remove { .. }
            ) {
                ahead = false;
                return true;
            }
            false
        });
    }
}

/// On a control the driver has hidden. A view also hides controls on its own
/// account (on a ghosted instance, say), and shows a control again only when
/// neither wants it hidden.
#[derive(Component)]
pub struct HiddenByDriver;

/// On the camera a goal's control is used through when the goal names no
/// view. A host that offers goals marks one; a control on an instance is used
/// through that instance's own.
#[derive(Component, Default)]
pub struct ControlCamera;

/// On a camera: the name a goal gives to be used through it
/// ([`Target::Goal::view`](Target::Goal)). What lets a goal choose among the
/// views of hosts sharing an app, where one [`ControlCamera`] cannot serve.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct ControlView(pub String);

/// A joint control waiting for its host to resolve it
/// ([module docs](self#joint-controls-jointrequest)).
///
/// The host's resolver, in [`ControlSystems::Joints`] of [`ControlPass`],
/// puts a [`JointGizmo`]
/// on the entity carrying this, built from its rig for `joint` on `instance`
/// and used through `camera`, standing where the joint stands now, in the
/// joint's own colour. It leaves the key, whether the gizmo moves the body,
/// and a colour the driver asked for to this module. A request it cannot
/// resolve it leaves as it is, and this module drops it. The request may be
/// despawned before the resolver's commands land, by a remove in the same
/// frame, so a resolver inserts with `try_insert`.
#[derive(Component, Clone, Debug)]
pub struct JointRequest {
    /// The entity carrying the joint's instance.
    pub instance: Entity,
    /// The joint, by the animatable that drives its value.
    pub joint: String,
    /// The camera the control is used through: the instance's.
    pub camera: Entity,
    key: String,
    colour: Option<Color>,
}

/// The size of the controls used through this camera that are given none of
/// their own ([module docs](self#how-big-a-control-is-controlsize)). A kind
/// left `None` keeps its own default.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct ControlSize {
    pub transform: Option<GizmoSize>,
    pub joint: Option<GizmoSize>,
}

/// Where a drag is ([module docs](self)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Phase {
    Start,
    Move,
    /// Finished: the pointer let go.
    End,
    /// Abandoned: nothing of the drag is to be kept.
    Cancel,
}

impl From<DragPhase> for Phase {
    fn from(phase: DragPhase) -> Self {
        match phase {
            DragPhase::Start => Self::Start,
            DragPhase::Move => Self::Move,
            DragPhase::End => Self::End,
            DragPhase::Cancel => Self::Cancel,
        }
    }
}

/// The modifier keys held, as the host has them. The host says, from its
/// own pointer or keyboard events, because the scene's view of the keyboard
/// is stale whenever its canvas lacks focus. Stamped on every [`ControlEvent`]
/// and given no meaning here: what a key changes about a drag is the
/// driver's to decide.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Modifiers {
    pub alt: bool,
    pub shift: bool,
    pub meta: bool,
    pub ctrl: bool,
}

/// A drag on a control: what it asks its target to become. Nothing has moved;
/// the driver writes the change if it wants it.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ControlEvent {
    pub key: String,
    pub phase: Phase,
    /// The drag this belongs to: the same on every event of one drag, and
    /// different from every other drag's in this app.
    pub gesture: u32,
    /// The modifier keys held when it was sent.
    pub modifiers: Modifiers,
    /// A joint control's value; none on a `cancel`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<f32>,
    /// A transform control's target's pose, in the world; none on a
    /// `cancel`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pose: Option<Pose>,
    /// A transform control's target's scale, along its own axes; none on a
    /// `cancel`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<[f32; 3]>,
}

/// Drags since a host last drained them.
#[derive(Resource, Default)]
pub struct ControlEvents(pub Vec<ControlEvent>);

/// The drag in hand on a control, from its `start` until its closing event:
/// its key, its gesture and the gizmo it is on. Kept so a drag whose control
/// is taken away — when there is no gizmo left to report it — still closes.
#[derive(Resource, Default)]
struct InHand(Option<(String, u32, Entity)>);

impl InHand {
    /// Notes an event reported for a control: a start puts its drag in hand,
    /// and a closing event of the drag in hand lets it go.
    fn note(&mut self, event: &ControlEvent, gizmo: Entity) {
        match event.phase {
            Phase::Start => self.0 = Some((event.key.clone(), event.gesture, gizmo)),
            Phase::Move => {}
            Phase::End | Phase::Cancel => {
                if self
                    .0
                    .as_ref()
                    .is_some_and(|(_, gesture, _)| *gesture == event.gesture)
                {
                    self.0 = None;
                }
            }
        }
    }
}

/// The button of a press on a control that is the host's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum PressButton {
    Secondary,
    Middle,
}

/// A press with a button other than the primary on a control, which is the
/// host's to act on (to open a menu for it, say). It moves nothing, and turns
/// and pans no view.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ControlPress {
    pub key: String,
    /// The handle pressed, in the form the overlay gives it.
    pub handle: GizmoHandle,
    pub button: PressButton,
    /// Where the press was, in the canvas's CSS pixels.
    pub position: [f32; 2],
}

/// Presses since a host last drained them.
#[derive(Resource, Default)]
pub struct ControlPresses(pub Vec<ControlPress>);

/// The controls in the view, by key.
#[derive(Resource, Default)]
pub struct Controls(HashMap<String, Placed>);

impl Controls {
    /// The entity carrying the gizmo of the control under `key`: a joint
    /// control's is its request's until the host resolves it.
    pub fn gizmo(&self, key: &str) -> Option<Entity> {
        self.0.get(key).map(|placed| placed.gizmo)
    }

    /// The keys of the controls in the view.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }
}

struct Placed {
    gizmo: Entity,
    /// What a transform control moves; `None` for a joint control.
    target: Option<Entity>,
    /// A goal the control brought with it, which goes when it goes.
    goal: Option<Entity>,
    /// The camera it is used through, whose frame its poses are in.
    camera: Entity,
    /// The entity carrying the instance it is on; `None` for a goal. The
    /// control goes when it does, whatever is left of its target.
    instance: Option<Entity>,
    /// Whether its driver wants it to take the pointer. The gizmo takes it
    /// only while this holds and the control shows.
    interactive: bool,
    /// Whether it is a joint add its host has yet to resolve: not yet a
    /// control, so an add replacing it cancels it rather than keeping it.
    resolving: bool,
    /// The opacity the host asked for, which the gizmo's fades toward.
    opacity: f32,
    /// The size its driver gave it, over its view's.
    size: Option<GizmoSize>,
    /// The size its gizmo was made at — [`GIZMO_SIZE`](crate::transform_gizmo::GIZMO_SIZE) for a transform
    /// control, what its host resolved a joint control at — which it returns
    /// to when neither its driver nor its view gives one, so a view's size
    /// taken away does not stay behind on it. `None` until it is known.
    made_at: Option<GizmoSize>,
    /// The fade toward `opacity` under way, if one is.
    fade: Option<Fade>,
    /// Whether its gizmo shows: drawn, and hidden neither by the host nor by
    /// its instance's being ghosted. While it does not, an opacity asked for
    /// is taken at once rather than faded to, so a control added dimmed, or
    /// dimmed while hidden, never shows at full strength on its way there.
    shown: bool,
}

impl Placed {
    fn new(
        gizmo: Entity,
        target: Option<Entity>,
        goal: Option<Entity>,
        camera: Entity,
        instance: Option<Entity>,
    ) -> Self {
        Self {
            gizmo,
            target,
            goal,
            camera,
            instance,
            interactive: true,
            resolving: false,
            opacity: 1.0,
            size: None,
            made_at: None,
            fade: None,
            shown: false,
        }
    }
}

/// A fade under way. It runs for [`FADE_SECS`] however far it has to go, so
/// it keeps where it started rather than moving at a set rate, which would
/// take a small change in a frame or two.
struct Fade {
    /// The gizmo's opacity when the fade began, read on its first frame.
    from: Option<f32>,
    /// How long it has run, in steps of at most a thirtieth of a second.
    elapsed: f32,
}

/// How long a control takes to fade to the opacity a host asks for.
pub const FADE_SECS: f32 = 0.2;

/// The controls' systems, for another system to order itself against.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum ControlSystems {
    /// In `Update`: carrying out commands, every pass of [`ControlPass`] the
    /// frame takes.
    Commands,
    /// In [`ControlPass`]: where a host resolves joint requests, after the
    /// pass has spawned them and before they are finished for the next.
    Joints,
    /// In `Update`: the fade toward the opacity each control was asked for.
    Fade,
}

/// One pass over the commands waiting: carry them out, let the host resolve
/// the joint requests the pass spawned ([`ControlSystems::Joints`]), and
/// finish them. Run from `Update`, in [`ControlSystems::Commands`], as many
/// times a frame as the commands need ([module docs](self#the-rules)).
///
/// A schedule of its own because a pass is repeated: a host's joint
/// resolver goes here, so it runs between every two passes, rather than in
/// `Update`, where it would run once.
#[derive(bevy::ecs::schedule::ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ControlPass;

/// The most passes a frame takes. Each pass a batch needs beyond the first
/// is one more replacement chained under a key in it, so a host reaching
/// this is sending something stranger than a driver would; what is left
/// waits for the next frame, and the log says so.
const MOST_PASSES: usize = 8;

/// A pass, either side of the joints.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
enum Pass {
    CarryOut,
    Finish,
}

/// What one pass leaves the next.
#[derive(Resource, Default)]
struct PassState {
    /// Whether another pass would carry out more of what is waiting.
    again: bool,
}

/// The controls joint adds would replace, by key, kept until their hosts
/// resolve them. A joint add replaces the control under its key only once
/// it is resolved: one that fails leaves that control as it was, as any
/// refused add does.
#[derive(Resource, Default)]
struct Replaced(HashMap<String, Placed>);

/// Commands, events and presses for every host's controls
/// ([module docs](self)). A host adds the gizmos' own plugins beside it.
pub struct ControlsPlugin;

impl Plugin for ControlsPlugin {
    /// Both consumers sharing an app may add it; it installs once.
    fn is_unique(&self) -> bool {
        false
    }

    fn build(&self, app: &mut App) {
        if !crate::installed::first_install::<Self>(app) {
            return;
        }
        app.add_plugins(crate::instance::InstancePlugin);
        app.init_resource::<ControlCommands>()
            .init_resource::<ControlEvents>()
            .init_resource::<InHand>()
            .init_resource::<Modifiers>()
            .init_resource::<GizmoDrags>()
            .init_resource::<ControlPresses>()
            .init_resource::<Controls>()
            .init_resource::<DefaultDriverFrame>()
            .init_resource::<PassState>()
            .init_resource::<Replaced>()
            .add_message::<RequestRedraw>();
        // Small and run several times a frame, so on one thread: the cost of
        // handing a few systems to a pool would outweigh the work. It applies
        // the commands its systems leave as it ends, so a pass's spawns are
        // there for the next.
        let mut pass = Schedule::new(ControlPass);
        pass.set_executor(bevy::ecs::schedule::SingleThreadedExecutor::new());
        app.add_schedule(pass)
            .configure_sets(
                ControlPass,
                (Pass::CarryOut, ControlSystems::Joints, Pass::Finish).chain(),
            )
            .add_systems(
                ControlPass,
                (
                    (forget_orphans, carry_out).chain().in_set(Pass::CarryOut),
                    finish_joints.in_set(Pass::Finish),
                ),
            )
            // Before the gizmos' own drag handling, so a left press in the
            // frame a control is made inert does not take it. A press with
            // another button is reported in `PreUpdate`, before any pass, so
            // one in that frame can still reach the host: a press and a
            // change arriving between the same two frames have no order to
            // keep.
            .configure_sets(
                Update,
                (ControlSystems::Commands, ControlSystems::Fade)
                    .chain()
                    .before(TransformGizmoSystems),
            )
            // On indexed instances, and before values are applied, so a
            // driver's write-back of a drag lands in the frame it is sent.
            .configure_sets(
                Update,
                ControlSystems::Commands
                    .after(InstanceSystems::Index)
                    .after(crate::instance::tag_elements)
                    .before(InstanceSystems::Apply),
            )
            .add_systems(
                Update,
                (
                    run_passes.in_set(ControlSystems::Commands),
                    // After the commands, so a control added or resized this
                    // frame is drawn at its size in it.
                    size_controls
                        .after(ControlSystems::Commands)
                        .before(hide_ghosted),
                    // After the commands, so a control added this frame is
                    // hidden in it, and before the fade reads whether a
                    // control shows.
                    hide_ghosted
                        .after(ControlSystems::Commands)
                        .before(ControlSystems::Fade),
                    // After whatever hides a control this frame, and before
                    // the gizmos' own pointer handling reads `interactive`.
                    (fade, take_input_only_while_shown)
                        .chain()
                        .in_set(ControlSystems::Fade),
                    // After the commands and the fade, which take controls
                    // away and forget them, and before the gizmos' own
                    // pointer handling, which lets go of a drag whose gizmo
                    // is gone without a word and could start another in the
                    // same frame.
                    cancel_taken_away
                        .after(ControlSystems::Fade)
                        .before(TransformGizmoSystems),
                ),
            )
            .add_observer(report_transform_drag)
            .add_observer(report_joint_drag)
            .add_observer(report_press);
    }
}

/// Sizes each control's gizmo: the size its driver gave it, else its view's
/// ([`ControlSize`]), else the size it was made at. A joint control is sized
/// only once its host has resolved it, in the passes before this, so the size
/// it was made at is the host's rather than its view's.
fn size_controls(
    controls: Res<Controls>,
    views: Query<&ControlSize>,
    mut transform_gizmos: Query<&mut TransformGizmo>,
    mut joint_gizmos: Query<&mut JointGizmo>,
) {
    for placed in controls.0.values() {
        let view = views.get(placed.camera).ok();
        if let Ok(gizmo) = transform_gizmos.get_mut(placed.gizmo) {
            if let Some(size) = placed
                .size
                .or(view.and_then(|view| view.transform))
                .or(placed.made_at)
            {
                gizmo
                    .map_unchanged(|gizmo| &mut gizmo.size)
                    .set_if_neq(size);
            }
        } else if let Ok(gizmo) = joint_gizmos.get_mut(placed.gizmo) {
            if let Some(size) = placed
                .size
                .or(view.and_then(|view| view.joint))
                .or(placed.made_at)
            {
                gizmo
                    .map_unchanged(|gizmo| &mut gizmo.size)
                    .set_if_neq(size);
            }
        }
    }
}

/// Forgets each control whose gizmo, instance, target or camera is gone:
/// despawns what is left of it — its gizmo, and its goal if it brought one —
/// and frees its key, so what is sent under the key next acts as for a key
/// with nothing placed.
///
/// A host may take an instance away without taking the gizmos on it — one
/// that despawns an instance and its camera together, say — or take a gizmo
/// away without the goal it was on. A control kept for a target that is gone
/// would hold its key for good: a pose for it waits on a transform that never
/// comes, asking for frames, and holds every later command under the key
/// behind it; and a goal is nobody's but its control's, so left behind it
/// stays for good. So this does not depend on a host's cleanup.
fn forget_orphans(
    mut controls: ResMut<Controls>,
    entities: &Entities,
    placeable: Query<(), Placeable>,
    joint_gizmos: Query<&JointGizmo>,
    mut commands: Commands,
) {
    controls.0.retain(|key, placed| {
        // A joint control's target is the body its gizmo turns or slides.
        let body = joint_gizmos
            .get(placed.gizmo)
            .ok()
            .map(|gizmo| gizmo.target);
        let there = placeable.contains(placed.gizmo)
            && std::iter::once(placed.camera)
                .chain(placed.instance)
                .chain(placed.target)
                .chain(body)
                .all(|entity| entities.contains_spawned(entity));
        if !there {
            log::debug!("{key:?}'s gizmo, instance, target or camera is gone; its control goes");
            commands.entity(placed.gizmo).try_despawn();
            if let Some(goal) = placed.goal {
                commands.entity(goal).try_despawn();
            }
        }
        there
    });
}

/// Runs [`ControlPass`] until a pass leaves nothing that another would carry
/// out, at most [`MOST_PASSES`] times. The commands a pass leaves land when
/// it ends, so the gizmos it spawned are there for the next.
fn run_passes(world: &mut World) {
    let mut passes = 0;
    while passes < MOST_PASSES {
        world.run_schedule(ControlPass);
        passes += 1;
        if !world.resource::<PassState>().again {
            break;
        }
    }
    if passes == MOST_PASSES && world.resource::<PassState>().again {
        log::warn!(
            "control commands still waiting on gizmos after {MOST_PASSES} passes; \
             the rest wait for the next frame"
        );
    }
}

/// Makes each control's gizmo take the pointer only while its driver wants
/// it to and it shows. A gizmo's hit test reads last frame's visibility, so a
/// control hidden — by its driver or by a ghosting — would go on taking
/// presses, and a drag in hand would go on reporting until released. Not
/// interactive, it takes neither from the next frame on, and a drag in hand
/// ends; the driver's own setting is kept on the control, and comes back
/// when it shows again. A press with another button than the primary is
/// taken in `PreUpdate`, before this runs, so one in the frame a control is
/// hidden in is still reported: it was made while the control showed.
fn take_input_only_while_shown(
    controls: Res<Controls>,
    visibility: Query<&Visibility>,
    mut transform_gizmos: Query<&mut TransformGizmo>,
    mut joint_gizmos: Query<&mut JointGizmo>,
) {
    for placed in controls.0.values() {
        let hidden = visibility
            .get(placed.gizmo)
            .is_ok_and(|visibility| *visibility == Visibility::Hidden);
        let wanted = placed.interactive && !hidden;
        if let Ok(mut gizmo) = transform_gizmos.get_mut(placed.gizmo) {
            if gizmo.interactive != wanted {
                gizmo.interactive = wanted;
            }
        } else if let Ok(mut gizmo) = joint_gizmos.get_mut(placed.gizmo) {
            if gizmo.interactive != wanted {
                gizmo.interactive = wanted;
            }
        }
    }
}

/// A CSS hex colour, or `None`, said in the log, for one that is not.
fn colour(hex: &str) -> Option<Color> {
    match Srgba::hex(hex) {
        Ok(colour) => Some(colour.into()),
        Err(_) => {
            log::warn!("{hex:?} is not a hex colour; ignored");
            None
        }
    }
}

/// Three CSS hex colours, or `None` if any is not one.
fn colours(hex: &[String; 3]) -> Option<[Color; 3]> {
    Some([colour(&hex[0])?, colour(&hex[1])?, colour(&hex[2])?])
}

/// Fades each control's gizmo evenly from where its fade began to the opacity
/// its host asked for, over [`FADE_SECS`], asking for frames while any fade
/// is still under way.
///
/// A step is at most a thirtieth of a second's: a view that draws only on
/// change can take its first frame after a quiet spell seconds after the one
/// before, which would otherwise take the whole fade in one jump.
fn fade(
    time: Res<Time>,
    mut controls: ResMut<Controls>,
    mut transform_gizmos: Query<&mut TransformGizmo>,
    mut joint_gizmos: Query<&mut JointGizmo>,
    visibility: Query<&Visibility>,
    mut redraw: MessageWriter<RequestRedraw>,
) {
    let step = time.delta_secs().min(1.0 / 30.0);
    let mut fading = false;
    for placed in controls.0.values_mut() {
        // Drawn after this, in the frame's own `PostUpdate`, unless hidden:
        // whatever hides a control, the host or a ghosting, does it here.
        placed.shown = visibility
            .get(placed.gizmo)
            .is_ok_and(|visibility| *visibility != Visibility::Hidden);
        let Some(fade) = placed.fade.as_mut() else {
            continue;
        };
        let opacity = if let Ok(gizmo) = transform_gizmos.get_mut(placed.gizmo) {
            gizmo.map_unchanged(|gizmo| &mut gizmo.opacity)
        } else if let Ok(gizmo) = joint_gizmos.get_mut(placed.gizmo) {
            gizmo.map_unchanged(|gizmo| &mut gizmo.opacity)
        } else {
            // Its gizmo is gone, and the fade with it.
            placed.fade = None;
            continue;
        };
        let from = *fade.from.get_or_insert(*opacity);
        fade.elapsed += step;
        let done = (fade.elapsed / FADE_SECS).min(1.0);
        let mut opacity = opacity;
        *opacity = from + (placed.opacity - from) * done;
        if done >= 1.0 {
            // Exactly what was asked for, whatever the rounding on the way.
            *opacity = placed.opacity;
            placed.fade = None;
        } else {
            fading = true;
        }
    }
    if fading {
        redraw.write(RequestRedraw);
    }
}

/// A control's gizmo of either kind, with whether it shows and whether its
/// driver hid it.
type Showing = (
    Option<&'static JointGizmo>,
    Option<&'static TransformGizmo>,
    &'static mut Visibility,
    Has<HiddenByDriver>,
);

/// Hides the controls on a ghosted instance: a ghost shows where an instance
/// is, not something to take hold of. A control shows only when neither the
/// ghosting nor the driver hides it. Only the controls placed here are
/// touched; a gizmo a host spawned for itself keeps whatever it was given.
fn hide_ghosted(
    controls: Res<Controls>,
    mut gizmos: Query<Showing>,
    ghosts: Query<(), With<Ghost>>,
    parents: Query<&ChildOf>,
) {
    for placed in controls.0.values() {
        let Ok((joint, transform, mut visibility, hidden_by_driver)) = gizmos.get_mut(placed.gizmo)
        else {
            continue;
        };
        let Some(target) = joint
            .map(|gizmo| gizmo.target)
            .or(transform.map(|gizmo| gizmo.target))
        else {
            continue;
        };
        let ghosted = std::iter::once(target)
            .chain(parents.iter_ancestors(target))
            .any(|entity| ghosts.contains(entity));
        let wanted = if ghosted || hidden_by_driver {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        if *visibility != wanted {
            *visibility = wanted;
        }
    }
}

/// Finishes each joint request its host resolved, and drops each it did not
/// ([module docs](self#joint-controls-jointrequest)).
///
/// A resolved joint replaces the control its add would replace, kept until
/// now. A failed one leaves that control as it was, and takes the changes
/// waiting on it out of the queue there and then: they were its, and left
/// there they would wait for, and land on, the next add under the key —
/// in a later frame, if this was a frame's last pass.
fn finish_joints(
    mut commands: Commands,
    mut requests: Query<(Entity, &JointRequest, Option<&mut JointGizmo>)>,
    instances: Query<&EntityInstance>,
    mut controls: ResMut<Controls>,
    mut replaced: ResMut<Replaced>,
    mut pending: ResMut<ControlCommands>,
) {
    for (entity, request, gizmo) in &mut requests {
        let key = &request.key;
        // A later add under the key has taken it since; whatever it replaced
        // is that add's business.
        let current = controls.owns(key, entity);
        match gizmo {
            Some(mut gizmo) => {
                gizmo.key = key.clone();
                gizmo.apply = false;
                if let Some(colour) = request.colour {
                    gizmo.colour = colour;
                }
                if !current {
                    // Cancelled by the add that took its key.
                    commands.entity(entity).despawn();
                    continue;
                }
                commands.entity(entity).remove::<JointRequest>();
                if let Some(placed) = controls.0.get_mut(key) {
                    placed.resolving = false;
                    placed.made_at = Some(gizmo.size);
                }
                if let Some(old) = replaced.0.remove(key) {
                    despawn(&mut commands, old);
                }
            }
            None => {
                let instance = instances
                    .get(request.instance)
                    .map_or("an instance gone", |instance| instance.id.as_str());
                log::warn!(
                    "no joint of {instance} is driven by {:?}; no control added for {key:?}",
                    request.joint,
                );
                commands.entity(entity).despawn();
                if current {
                    controls.0.remove(key);
                    if let Some(old) = replaced.0.remove(key) {
                        controls.0.insert(key.clone(), old);
                    }
                    pending.drop_changes_ahead_of_next_add(key);
                }
            }
        }
    }
}

/// What a control's gizmo entity carries once spawned, or, for a joint
/// control, while its host resolves it.
type Placeable = Or<(With<TransformGizmo>, With<JointGizmo>, With<JointRequest>)>;

#[allow(clippy::too_many_arguments)]
fn carry_out(
    mut pending: ResMut<ControlCommands>,
    mut controls: ResMut<Controls>,
    app_frame: Res<DefaultDriverFrame>,
    frames: Query<&DriverFrame>,
    mut commands: Commands,
    cameras: Query<GoalCamera, GoalCameras>,
    instances: Query<&EntityInstance>,
    index: Res<InstanceIndex>,
    parents: Query<&ChildOf>,
    globals: Query<&GlobalTransform>,
    mut transforms: Query<&mut Transform>,
    mut transform_gizmos: Query<&mut TransformGizmo>,
    mut joint_gizmos: Query<&mut JointGizmo>,
    mut redraw: MessageWriter<RequestRedraw>,
    mut state: ResMut<PassState>,
    mut replaced: ResMut<Replaced>,
) {
    state.again = false;
    // Keys whose add was refused in this pass, and whose changes are dropped
    // up to the next add or remove under the key: those changes were the
    // refused add's. A pass carries out every command it does not put off,
    // so the changes this owns are all met in this pass.
    let mut failed: HashSet<String> = HashSet::new();
    if pending.0.is_empty() {
        return;
    }
    // The instance under a host's id, if one is there, indexed or not. One
    // being replaced gives way to its successor, which is the one a control
    // should stay on.
    let instance = |name: &str| {
        let entity = index.get(name)?;
        instances
            .get(entity)
            .ok()
            .map(|instance| (entity, instance))
    };
    // Whether an instance not ready to be addressed is on its way: loading,
    // or there but not yet indexed, so its elements cannot be told from ones
    // it lacks. What a control on it waits for.
    let coming = |name: &str| index.is_loading(name);
    let asked = std::mem::take(&mut pending.0);
    let asked_count = asked.len();
    // Keys an add is coming for, in this batch or waiting from an earlier one.
    let adding: HashSet<String> = asked
        .iter()
        .filter_map(|command| match command {
            ControlCommand::Add { key, .. } => Some(key.clone()),
            _ => None,
        })
        .collect();
    // Whether a change for `key` should wait: for its control's gizmo, which
    // an add spawns at the end of its pass, or for an add still to come, in
    // this batch or behind an instance still loading, as a change sent before
    // its add is. Anything else is for a key nothing will add, such as one
    // whose control went with its instance, and waiting would hold it, and
    // the frames it asks for, for good.
    let any_loading =
        index.any_loading() || instances.iter().any(|instance| !instance.bindings.indexed);
    let worth_waiting = |key: &str, controls: &Controls| {
        controls.0.contains_key(key) || adding.contains(key) || any_loading
    };
    let mut waiting = Vec::new();
    // How each key with a command waiting holds the commands after it
    // ([`Hold`]): what keeps a key's commands landing in the order sent.
    let mut holds: HashMap<String, Hold> = HashMap::new();
    // Whether to draw another frame: for a command that changed the scene, or
    // one waiting on a gizmo spawned at the end of this pass. Not for one
    // waiting on an instance or a camera, which no frame of this module's
    // brings: a view that draws on demand can idle until the host's own
    // change comes.
    let mut again = false;
    // Goals spawned in this pass, which have no transform until it ends.
    let mut spawned_goals: HashSet<Entity> = HashSet::new();
    for command in asked {
        // A remove is not held: it cancels the waiting add and what waits
        // behind it, and takes away the control the add would have replaced.
        let behind = match (holds.get(command.key()), &command) {
            (_, ControlCommand::Remove { .. }) | (None, _) => false,
            (Some(Hold::All), _) => true,
            (Some(Hold::ForAdd), ControlCommand::Add { .. }) => false,
            (Some(Hold::ForAdd), _) => true,
        };
        if behind {
            waiting.push(command);
            continue;
        }
        if failed.contains(command.key()) {
            if matches!(
                command,
                ControlCommand::Add { .. } | ControlCommand::Remove { .. }
            ) {
                failed.remove(command.key());
            } else {
                log::debug!(
                    "{:?}'s add failed; a change sent with it goes with it",
                    command.key()
                );
                continue;
            }
        }
        match command {
            ControlCommand::Add { key, control } => {
                // A control on an instance waits for it while it is loading
                // or not yet indexed; one on an instance nothing is loading
                // is refused.
                let on = match control.instance() {
                    None => None,
                    Some(name) => match instance(name) {
                        Some((_, found)) if !found.bindings.indexed => {
                            defer(
                                &mut waiting,
                                &mut holds,
                                Hold::All,
                                ControlCommand::Add { key, control },
                            );
                            continue;
                        }
                        Some(found) => Some(found),
                        None if coming(name) => {
                            defer(
                                &mut waiting,
                                &mut holds,
                                Hold::All,
                                ControlCommand::Add { key, control },
                            );
                            continue;
                        }
                        None => {
                            let why = format!("no instance {name:?}");
                            refuse_add(&why, key, &mut waiting, &mut holds, &mut failed);
                            continue;
                        }
                    },
                };
                // An instance's own camera, or for a goal the view it names.
                // A goal at something that is not a pose is refused before its
                // view is looked for: waiting for a view that may never come,
                // it would hold every later add under its key for good.
                if let Control::Transform {
                    target: Target::Goal { pose, .. },
                    ..
                } = &control
                {
                    if !pose.is_valid() {
                        let why = format!("{pose:?} is not a pose a goal can stand at");
                        refuse_add(&why, key, &mut waiting, &mut holds, &mut failed);
                        continue;
                    }
                }
                let camera = match (on, &control) {
                    (Some((_, instance)), _) => instance.camera,
                    (
                        None,
                        Control::Transform {
                            target: Target::Goal { view, .. },
                            ..
                        },
                    ) => match goal_camera(&cameras, view.as_deref()) {
                        Ok(Some(camera)) => camera,
                        Ok(None) => {
                            log::debug!("no camera for {key:?}'s goal yet; it waits for one");
                            defer(
                                &mut waiting,
                                &mut holds,
                                Hold::All,
                                ControlCommand::Add { key, control },
                            );
                            continue;
                        }
                        Err(why) => {
                            refuse_add(&why, key, &mut waiting, &mut holds, &mut failed);
                            continue;
                        }
                    },
                    // Every other control is on an instance.
                    (None, _) => continue,
                };
                let frame = DriverFrame::of(camera, &frames, &app_frame);
                // What a transform control moves, settled before anything is
                // changed: a refused add leaves the control it would have
                // replaced as it was. `None` for a joint, which its host
                // resolves, and for a goal, which is spawned below.
                let target = match &control {
                    Control::Joint { .. } => Ok(None),
                    Control::Transform { target, .. } => match (target, on) {
                        (Target::Instance { .. }, Some((root, _))) => Ok(Some(root)),
                        (Target::Element { id, .. }, Some((_, instance))) => instance
                            .bindings
                            .elements
                            .get(id)
                            .map(|element| Some(element.node))
                            .ok_or_else(|| format!("no element {id:?} on {}", instance.id)),
                        (Target::Goal { .. }, _) => Ok(None),
                        (Target::Instance { .. } | Target::Element { .. }, None) => continue,
                    },
                };
                let target = match target {
                    Ok(target) => target,
                    Err(why) => {
                        refuse_add(&why, key, &mut waiting, &mut holds, &mut failed);
                        continue;
                    }
                };
                // A joint add replaces the control under its key once its
                // host resolves it, so that control is kept until then. Any
                // other add replaces it now, and with it a control a joint
                // add still being resolved would have replaced. A joint add
                // still being resolved is no control yet: replaced, it is
                // cancelled, and the control it would have replaced is kept
                // for the add that replaces it.
                let joint = matches!(control, Control::Joint { .. });
                if let Some(old) = controls.0.remove(&key) {
                    if joint && !old.resolving && !replaced.0.contains_key(&key) {
                        replaced.0.insert(key.clone(), old);
                    } else {
                        despawn(&mut commands, old);
                    }
                }
                if !joint {
                    if let Some(kept) = replaced.0.remove(&key) {
                        despawn(&mut commands, kept);
                    }
                }
                again = true;
                let root = on.map(|(root, _)| root);
                let placed = match control {
                    Control::Joint {
                        joint, color, size, ..
                    } => {
                        let Some(root) = root else {
                            continue;
                        };
                        let request = JointRequest {
                            instance: root,
                            joint,
                            camera,
                            key: key.clone(),
                            colour: color.as_deref().and_then(colour),
                        };
                        Placed {
                            resolving: true,
                            size,
                            ..Placed::new(
                                commands.spawn(request).id(),
                                None,
                                None,
                                camera,
                                Some(root),
                            )
                        }
                    }
                    Control::Transform {
                        target: asked,
                        modes,
                        local,
                        colors,
                        size,
                        face_viewer,
                    } => {
                        let (target, goal) = match (target, asked) {
                            (Some(target), _) => (target, None),
                            (None, Target::Goal { pose, .. }) => {
                                let goal = commands
                                    .spawn((pose.to_world(&frame), Visibility::default()))
                                    .id();
                                spawned_goals.insert(goal);
                                (goal, Some(goal))
                            }
                            (None, _) => continue,
                        };
                        let modes = match modes {
                            None => TransformModes::ALL,
                            Some(modes) => TransformModes {
                                translate: modes.iter().any(|m| matches!(m, Mode::Translate)),
                                rotate: modes.iter().any(|m| matches!(m, Mode::Rotate)),
                                scale: modes.iter().any(|m| matches!(m, Mode::Scale)),
                            },
                        };
                        let base = TransformGizmo::new(target, camera);
                        let gizmo = commands
                            .spawn(TransformGizmo {
                                key: key.clone(),
                                colours: colors.as_ref().and_then(colours).unwrap_or(base.colours),
                                modes,
                                space: if local {
                                    GizmoSpace::Local
                                } else {
                                    GizmoSpace::World
                                },
                                // Labelled as the driver names the axes.
                                axes: frame.axes(),
                                apply: false,
                                face_viewer,
                                ..base
                            })
                            .id();
                        Placed {
                            size,
                            made_at: Some(base.size),
                            ..Placed::new(gizmo, Some(target), goal, camera, root)
                        }
                    }
                };
                // Changes waiting for an add are this control's now, and act
                // on it before anything sent after it under the key.
                if let Some(hold) = holds.get_mut(&key) {
                    *hold = Hold::All;
                }
                controls.0.insert(key, placed);
            }
            ControlCommand::Remove { key } => {
                if let Some(old) = controls.0.remove(&key) {
                    despawn(&mut commands, old);
                    again = true;
                }
                if let Some(kept) = replaced.0.remove(&key) {
                    despawn(&mut commands, kept);
                    again = true;
                }
                // Whatever the driver asked of this key that is still
                // waiting goes too, or an add made before its instance
                // arrived would bring back a control the driver has since
                // removed.
                waiting.retain(|waiting| waiting.key() != key);
                holds.remove(&key);
            }
            ControlCommand::Show { key, visible } => {
                if let Some(placed) = controls.0.get_mut(&key) {
                    again = true;
                    let mut gizmo = commands.entity(placed.gizmo);
                    if visible {
                        gizmo
                            .remove::<HiddenByDriver>()
                            .insert(Visibility::Inherited);
                    } else {
                        gizmo.insert((HiddenByDriver, Visibility::Hidden));
                        // Hidden from now, for a change later in the batch;
                        // the fade reads whether it shows afresh each frame.
                        placed.shown = false;
                    }
                } else if worth_waiting(&key, &controls) {
                    // Shown or hidden before it was added, it waits for the
                    // add, as any change does.
                    defer(
                        &mut waiting,
                        &mut holds,
                        Hold::ForAdd,
                        ControlCommand::Show { key, visible },
                    );
                } else {
                    log::debug!("no control {key:?} to show or hide");
                }
            }
            ControlCommand::SetPose { key, pose, scale } => {
                if !pose.is_valid()
                    || scale.is_some_and(|scale| !scale.iter().all(|v| v.is_finite()))
                {
                    log::warn!("{pose:?}, scale {scale:?}, is not a pose; {key:?} not moved");
                    continue;
                }
                let placed = controls.0.get(&key);
                let target = placed.and_then(|p| p.target);
                let frame = placed.map(|p| DriverFrame::of(p.camera, &frames, &app_frame));
                // A goal added in this pass is spawned at the end of it, so it
                // has no transform yet; the pose waits for the next pass, as
                // one set before the add would. Any other target is there
                // already, and one with no transform of its own never will
                // have one, so the pose is refused rather than waiting.
                let (Some(target), Some(frame)) =
                    (target.filter(|t| transforms.contains(*t)), frame)
                else {
                    if target.is_none() && controls.0.contains_key(&key) {
                        log::warn!("{key:?} is a joint control; write its value instead");
                    } else if target.is_some_and(|target| !spawned_goals.contains(&target)) {
                        log::warn!("{key:?}'s target has no transform to set; not moved");
                    } else if worth_waiting(&key, &controls) {
                        // Behind the gizmo its add is spawning, or behind
                        // an add still to come.
                        let placed = controls.0.contains_key(&key);
                        again |= placed;
                        defer(
                            &mut waiting,
                            &mut holds,
                            if placed { Hold::All } else { Hold::ForAdd },
                            ControlCommand::SetPose { key, pose, scale },
                        );
                    } else {
                        log::debug!("no control {key:?} to pose");
                    }
                    continue;
                };
                again = true;
                let world = pose.to_world(&frame);
                // The pose is in the world; the target's transform is in its
                // parent's frame, as it stands now.
                let local = match parents.get(target).ok().and_then(|p| {
                    current_global(
                        p.parent(),
                        |entity| transforms.get(entity).ok().copied(),
                        &parents,
                        &globals,
                    )
                }) {
                    Some(parent) => parent.inverse() * world.compute_affine(),
                    None => world.compute_affine(),
                };
                let Ok(mut transform) = transforms.get_mut(target) else {
                    continue;
                };
                let (_, rotation, translation) = local.to_scale_rotation_translation();
                transform.translation = translation;
                transform.rotation = rotation;
                if let Some(scale) = scale {
                    transform.scale = Vec3::from(scale);
                }
            }
            ControlCommand::Configure {
                key,
                color,
                colors,
                local,
                opacity,
                interactive,
                size,
                face_viewer,
            } => {
                // A control added in this pass is spawned at the end of it,
                // and a joint control resolved between the passes, so its
                // gizmo is not there yet; the change waits for the next pass,
                // as one made before the add would.
                let gizmo = controls.0.get(&key).map(|placed| placed.gizmo);
                let spawned = gizmo.is_some_and(|gizmo| {
                    transform_gizmos.contains(gizmo) || joint_gizmos.contains(gizmo)
                });
                let (Some(gizmo), true) = (gizmo, spawned) else {
                    if worth_waiting(&key, &controls) {
                        // Behind the gizmo its add is spawning, or behind
                        // an add still to come.
                        let placed = controls.0.contains_key(&key);
                        again |= placed;
                        defer(
                            &mut waiting,
                            &mut holds,
                            if placed { Hold::All } else { Hold::ForAdd },
                            ControlCommand::Configure {
                                key,
                                color,
                                colors,
                                local,
                                opacity,
                                interactive,
                                size,
                                face_viewer,
                            },
                        );
                    } else {
                        log::debug!("no control {key:?} to configure");
                    }
                    continue;
                };
                again = true;
                // An opacity for a control not yet drawn, taken as it is.
                let mut at_once = None;
                if let (Some(opacity), Some(placed)) = (opacity, controls.0.get_mut(&key)) {
                    let opacity = opacity.clamp(0.0, 1.0);
                    if !placed.shown {
                        placed.opacity = opacity;
                        placed.fade = None;
                        at_once = Some(opacity);
                    } else if opacity != placed.opacity {
                        // Asked again for where it is already going, a fade
                        // runs on; asked for somewhere new, it starts over
                        // from where the gizmo has got to.
                        placed.opacity = opacity;
                        placed.fade = Some(Fade {
                            from: None,
                            elapsed: 0.0,
                        });
                    }
                }
                // Kept on the control and put on its gizmo by
                // `size_controls`, over its view's.
                if let (Some(size), Some(placed)) = (size, controls.0.get_mut(&key)) {
                    placed.size = Some(size);
                }
                // Kept on the control, which a hidden gizmo's `interactive`
                // is not: it comes back when the control shows again.
                if let (Some(interactive), Some(placed)) = (interactive, controls.0.get_mut(&key)) {
                    placed.interactive = interactive;
                }
                // Neither kind's own: one command may carry these for both.
                let common = opacity.is_some() || interactive.is_some() || size.is_some();
                if let Ok(mut gizmo) = transform_gizmos.get_mut(gizmo) {
                    if let Some(opacity) = at_once {
                        gizmo.opacity = opacity;
                    }
                    if let Some(interactive) = interactive {
                        gizmo.interactive = interactive;
                    }
                    if let Some(colours) = colors.as_ref().and_then(colours) {
                        gizmo.colours = colours;
                    }
                    if let Some(local) = local {
                        gizmo.space = if local {
                            GizmoSpace::Local
                        } else {
                            GizmoSpace::World
                        };
                    }
                    if let Some(face_viewer) = face_viewer {
                        gizmo.face_viewer = face_viewer;
                    }
                    if color.is_some()
                        && colors.is_none()
                        && local.is_none()
                        && face_viewer.is_none()
                        && !common
                    {
                        log::warn!("{key:?} is a transform control; give it colors");
                    }
                } else if let Ok(mut gizmo) = joint_gizmos.get_mut(gizmo) {
                    if let Some(opacity) = at_once {
                        gizmo.opacity = opacity;
                    }
                    if let Some(interactive) = interactive {
                        gizmo.interactive = interactive;
                    }
                    if let Some(colour) = color.as_deref().and_then(colour) {
                        gizmo.colour = colour;
                    }
                    if color.is_none()
                        && (colors.is_some() || local.is_some() || face_viewer.is_some())
                        && !common
                    {
                        log::warn!("{key:?} is a joint control; give it a color");
                    }
                }
            }
            ControlCommand::Highlight { key, handle, on } => {
                // Waits for a control added in this pass, as a configure does.
                let gizmo = controls.0.get(&key).map(|placed| placed.gizmo);
                let spawned = gizmo.is_some_and(|gizmo| {
                    transform_gizmos.contains(gizmo) || joint_gizmos.contains(gizmo)
                });
                let (Some(gizmo), true) = (gizmo, spawned) else {
                    if worth_waiting(&key, &controls) {
                        // Behind the gizmo its add is spawning, or behind
                        // an add still to come.
                        let placed = controls.0.contains_key(&key);
                        again |= placed;
                        defer(
                            &mut waiting,
                            &mut holds,
                            if placed { Hold::All } else { Hold::ForAdd },
                            ControlCommand::Highlight { key, handle, on },
                        );
                    } else {
                        log::debug!("no control {key:?} to highlight");
                    }
                    continue;
                };
                again = true;
                if let Ok(mut gizmo) = transform_gizmos.get_mut(gizmo) {
                    match (handle, on) {
                        (Some(handle), true) => {
                            if !gizmo.highlight.contains(&handle) {
                                gizmo.highlight.push(handle);
                            }
                        }
                        (Some(handle), false) => gizmo.highlight.retain(|h| *h != handle),
                        (None, true) => gizmo.highlight = GizmoHandle::ALL.to_vec(),
                        (None, false) => gizmo.highlight.clear(),
                    }
                } else if let Ok(mut gizmo) = joint_gizmos.get_mut(gizmo) {
                    gizmo.show_in_hand = on;
                }
            }
        }
    }
    // Another pass, if this one took something off the queue and left a
    // change for a control now placed: its gizmo, spawned or resolved as
    // this pass ends, is there for the next. That covers a change sent with
    // its add, before or after it. What waits on anything else — an
    // instance, a camera, an add not yet sent — another pass cannot help.
    let on_spawn = waiting.iter().any(|command| {
        !matches!(command, ControlCommand::Add { .. }) && controls.0.contains_key(command.key())
    });
    state.again = on_spawn && waiting.len() < asked_count;
    pending.0 = waiting;
    if again {
        redraw.write(RequestRedraw);
    }
}

/// How a key's later commands wait behind one of its commands that waits.
///
/// Commands under a key take effect in the order sent, with one reordering
/// that is the point of waiting at all: a change sent before its control's
/// add acts on the control that add makes. So a key is held one of two ways,
/// and moves only toward holding more until a remove frees it:
///
/// - no hold → [`ForAdd`](Hold::ForAdd), when a change waits for an add not
///   yet made;
/// - no hold or `ForAdd` → [`All`](Hold::All), when an add waits (for its
///   instance or camera), when a change waits for the gizmo its control's
///   add is spawning, and when the add a `ForAdd` key waits for lands.
///
/// The last is what keeps a change sent before an add ahead of what was sent
/// after that add: once the add lands, the changes waiting are its
/// control's, and everything later, a replacing add included, waits behind
/// them for the pass that applies them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hold {
    /// Changes wait for an add: later changes wait behind them, and an add
    /// goes ahead, being what they wait for.
    ForAdd,
    /// Every later command waits.
    All,
}

/// Puts `command` off to a later pass, holding the commands after it under
/// its key as `hold` says, and never less than they already are.
fn defer(
    waiting: &mut Vec<ControlCommand>,
    holds: &mut HashMap<String, Hold>,
    hold: Hold,
    command: ControlCommand,
) {
    let held = holds.entry(command.key().to_string()).or_insert(hold);
    if hold == Hold::All {
        *held = Hold::All;
    }
    waiting.push(command);
}

/// Refuses an add, saying why. The changes waiting for it were its, sent
/// before it, and go with it; so do the changes sent after it, up to the
/// next add or remove under its key, which `failed` marks.
fn refuse_add(
    why: &str,
    key: String,
    waiting: &mut Vec<ControlCommand>,
    holds: &mut HashMap<String, Hold>,
    failed: &mut HashSet<String>,
) {
    log::warn!("{why}; no control added for {key:?}");
    // The add was not held, so all that waits under its key is changes
    // waiting for an add: this one.
    waiting.retain(|command| command.key() != key);
    holds.remove(&key);
    failed.insert(key);
}

/// A camera a goal can be used through, whether it is marked
/// [`ControlCamera`], and the view it is named, if it is.
type GoalCamera = (Entity, Has<ControlCamera>, Option<&'static ControlView>);

/// The cameras a goal can be used through: marked, or named.
type GoalCameras = Or<(With<ControlCamera>, With<ControlView>)>;

/// The camera a goal is used through: the one named the view it names, or,
/// naming none, the one marked [`ControlCamera`]. `Ok(None)` while there is
/// none yet, which the goal waits for. With several marked and none named,
/// any choice would be a guess at which view the driver meant, so the goal
/// is refused.
fn goal_camera(
    cameras: &Query<GoalCamera, GoalCameras>,
    view: Option<&str>,
) -> Result<Option<Entity>, String> {
    match view {
        Some(name) => Ok(cameras
            .iter()
            .find(|(_, _, named)| named.is_some_and(|named| named.0 == name))
            .map(|(camera, ..)| camera)),
        None => {
            let mut marked = cameras
                .iter()
                .filter(|(_, marked, _)| *marked)
                .map(|(camera, ..)| camera);
            match (marked.next(), marked.next()) {
                (None, _) => Ok(None),
                (Some(camera), None) => Ok(Some(camera)),
                (Some(_), Some(_)) => Err(
                    "several cameras are marked ControlCamera and the goal names no view".into(),
                ),
            }
        }
    }
}

fn despawn(commands: &mut Commands, placed: Placed) {
    commands.entity(placed.gizmo).despawn();
    if let Some(goal) = placed.goal {
        commands.entity(goal).despawn();
    }
}

impl Controls {
    /// Whether `gizmo` is the gizmo of the control under `key`. A gizmo's key
    /// alone is no proof: a host names the gizmos it spawns itself for the
    /// overlay, under keys a control may share, and a control's key may be
    /// empty. Only a control's own gizmo reports as that control.
    fn owns(&self, key: &str, gizmo: Entity) -> bool {
        self.0.get(key).is_some_and(|placed| placed.gizmo == gizmo)
    }
}

#[allow(clippy::too_many_arguments)]
fn report_transform_drag(
    drag: On<GizmoDrag>,
    controls: Res<Controls>,
    gizmos: Query<&TransformGizmo>,
    parents: Query<&ChildOf>,
    transforms: Query<&Transform>,
    globals: Query<&GlobalTransform>,
    app_frame: Res<DefaultDriverFrame>,
    frames: Query<&DriverFrame>,
    modifiers: Res<Modifiers>,
    mut in_hand: ResMut<InHand>,
    mut events: ResMut<ControlEvents>,
) {
    let Ok(gizmo) = gizmos.get(drag.entity) else {
        return;
    };
    if !controls.owns(&gizmo.key, drag.entity) {
        return;
    }
    if drag.phase == DragPhase::Cancel {
        let event = cancelled(&gizmo.key, drag.gesture, *modifiers);
        in_hand.note(&event, drag.entity);
        events.0.push(event);
        return;
    }
    // The drag gives the target's transform in its parent's frame, and the
    // parent is read as it stands now: drags are reported in `Update`, before
    // this frame's moves have propagated, as a pose is set.
    let world = match parents.get(drag.target).ok().and_then(|p| {
        current_global(
            p.parent(),
            |entity| transforms.get(entity).ok().copied(),
            &parents,
            &globals,
        )
    }) {
        Some(parent) => Transform::from_matrix((parent * drag.transform.compute_affine()).into()),
        None => drag.transform,
    };
    let event = ControlEvent {
        key: gizmo.key.clone(),
        phase: drag.phase.into(),
        gesture: drag.gesture,
        modifiers: *modifiers,
        value: None,
        pose: Some(Pose::from_world(
            &world,
            &DriverFrame::of(gizmo.camera, &frames, &app_frame),
        )),
        scale: Some(drag.transform.scale.to_array()),
    };
    in_hand.note(&event, drag.entity);
    events.0.push(event);
}

/// The closing event of a drag abandoned on the control under `key`.
fn cancelled(key: &str, gesture: u32, modifiers: Modifiers) -> ControlEvent {
    ControlEvent {
        key: key.to_owned(),
        phase: Phase::Cancel,
        gesture,
        modifiers,
        value: None,
        pose: None,
        scale: None,
    }
}

/// Cancels the drag in hand on a control that has been taken away — removed,
/// replaced under its key, or forgotten with its instance — since its gizmo
/// is gone and can report nothing.
fn cancel_taken_away(
    controls: Res<Controls>,
    modifiers: Res<Modifiers>,
    mut in_hand: ResMut<InHand>,
    mut events: ResMut<ControlEvents>,
) {
    let Some((key, gesture, gizmo)) = &in_hand.0 else {
        return;
    };
    if controls.owns(key, *gizmo) {
        return;
    }
    events.0.push(cancelled(key, *gesture, *modifiers));
    in_hand.0 = None;
}

fn report_press(
    press: On<GizmoPress>,
    controls: Res<Controls>,
    transform_gizmos: Query<&TransformGizmo>,
    joint_gizmos: Query<&JointGizmo>,
    mut presses: ResMut<ControlPresses>,
) {
    let key = match (
        transform_gizmos.get(press.entity),
        joint_gizmos.get(press.entity),
    ) {
        (Ok(gizmo), _) => &gizmo.key,
        (_, Ok(gizmo)) => &gizmo.key,
        _ => return,
    };
    let button = match press.button {
        PointerButton::Secondary => PressButton::Secondary,
        PointerButton::Middle => PressButton::Middle,
        PointerButton::Primary => return,
    };
    if !controls.owns(key, press.entity) {
        return;
    }
    presses.0.push(ControlPress {
        key: key.clone(),
        handle: press.handle,
        button,
        position: press.position.to_array(),
    });
}

fn report_joint_drag(
    drag: On<JointDrag>,
    controls: Res<Controls>,
    gizmos: Query<&JointGizmo>,
    modifiers: Res<Modifiers>,
    mut in_hand: ResMut<InHand>,
    mut events: ResMut<ControlEvents>,
) {
    let Ok(gizmo) = gizmos.get(drag.entity) else {
        return;
    };
    if !controls.owns(&gizmo.key, drag.entity) {
        return;
    }
    let event = if drag.phase == DragPhase::Cancel {
        cancelled(&gizmo.key, drag.gesture, *modifiers)
    } else {
        ControlEvent {
            key: gizmo.key.clone(),
            phase: drag.phase.into(),
            gesture: drag.gesture,
            modifiers: *modifiers,
            value: Some(drag.value),
            pose: None,
            scale: None,
        }
    };
    in_hand.note(&event, drag.entity);
    events.0.push(event);
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use bevy::time::TimeUpdateStrategy;

    use super::*;
    use crate::binding::{ElementTargets, SceneBindings};
    use crate::instance::{Loading, Superseded};
    use crate::joint_gizmo::JointMotion;

    const TILT: &str = "tilt-value";

    /// The instance the fixture's robot loads into.
    const ROBOT: &str = "robot-0";

    /// A host's rig, as far as a joint control needs one: the body each
    /// joint's animatable moves.
    #[derive(Component, Default)]
    struct Rig(HashMap<String, Entity>);

    /// The fixture host's joint resolver: a turn about X between -0.5 and
    /// 0.5, standing at 0, in white.
    fn resolve_joints(
        requests: Query<(Entity, &JointRequest), Without<JointGizmo>>,
        rigs: Query<&Rig>,
        mut commands: Commands,
    ) {
        for (entity, request) in &requests {
            let Some(&body) = rigs
                .get(request.instance)
                .ok()
                .and_then(|rig| rig.0.get(&request.joint))
            else {
                continue;
            };
            commands.entity(entity).try_insert(JointGizmo::new(
                body,
                request.camera,
                JointMotion::Turn,
                Dir3::X,
                -0.5,
                0.5,
                0.0,
                Color::WHITE,
            ));
        }
    }

    /// The fixture's world: Bevy's own Y up, as a robot scene's is.
    fn y_up() -> DriverFrame {
        DriverFrame::y_up()
    }

    /// An app with the controls and nothing else: no camera, no instance.
    fn bare() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(ControlsPlugin)
            .insert_resource(DefaultDriverFrame(y_up()));
        app
    }

    /// A view with a control camera, a joint resolver and a body the tilt
    /// joint moves, and a robot loading into `robot-0` but not yet there:
    /// `robot` puts it there.
    fn scene() -> (App, Entity) {
        let mut app = bare();
        app.add_systems(ControlPass, resolve_joints.in_set(ControlSystems::Joints));
        app.world_mut().spawn(ControlCamera);
        app.world_mut().spawn(Loading {
            instance: ROBOT.into(),
        });
        let body = app.world_mut().spawn(Transform::default()).id();
        (app, body)
    }

    fn camera(world: &mut World) -> Entity {
        world
            .query_filtered::<Entity, With<ControlCamera>>()
            .single(world)
            .unwrap()
    }

    fn instance(id: &str, camera: Entity) -> EntityInstance {
        EntityInstance {
            id: id.into(),
            asset_path: String::new(),
            meta: Default::default(),
            bindings: SceneBindings {
                indexed: true,
                ..Default::default()
            },
            camera,
            draw_order: 0,
            ready: true,
        }
    }

    fn rig(body: Entity) -> Rig {
        Rig(HashMap::from([(TILT.to_string(), body)]))
    }

    fn robot(app: &mut App, body: Entity) {
        spawn_robot(app.world_mut(), body);
    }

    /// The robot loading into `robot-0` arrives: no longer loading, and an
    /// instance.
    fn spawn_robot(world: &mut World, body: Entity) {
        let camera = camera(world);
        let loading = world
            .query_filtered::<Entity, With<Loading>>()
            .single(world)
            .unwrap();
        world
            .entity_mut(loading)
            .remove::<Loading>()
            .insert((rig(body), instance(ROBOT, camera)));
    }

    /// A second robot, `robot-1`, there already, with a tilt joint of its own
    /// moving `body`.
    fn second_robot(app: &mut App, body: Entity) {
        let world = app.world_mut();
        let camera = camera(world);
        world.spawn((rig(body), instance("robot-1", camera)));
    }

    fn joint_on(instance: &str) -> Control {
        Control::Joint {
            instance: instance.into(),
            joint: TILT.into(),
            color: None,
            size: None,
        }
    }

    fn add(key: &str, control: Control) -> ControlCommand {
        ControlCommand::Add {
            key: key.into(),
            control,
        }
    }

    fn send(app: &mut App, command: ControlCommand) {
        app.world_mut()
            .resource_mut::<ControlCommands>()
            .0
            .push(command);
        app.update();
    }

    fn queue(app: &mut App, commands: impl IntoIterator<Item = ControlCommand>) {
        app.world_mut()
            .resource_mut::<ControlCommands>()
            .0
            .extend(commands);
    }

    fn only<C: Component>(app: &mut App) -> Option<Entity> {
        let found: Vec<Entity> = app
            .world_mut()
            .query_filtered::<Entity, With<C>>()
            .iter(app.world())
            .collect();
        assert!(found.len() <= 1, "{} found", found.len());
        found.first().copied()
    }

    fn events(app: &mut App) -> Vec<ControlEvent> {
        std::mem::take(&mut app.world_mut().resource_mut::<ControlEvents>().0)
    }

    fn frames_asked(app: &mut App) -> usize {
        let mut messages = app.world_mut().resource_mut::<Messages<RequestRedraw>>();
        let asked = messages.len();
        messages.clear();
        asked
    }

    /// Each joint control lands on the joint of the robot it names, though
    /// both robots' joints are driven by one animatable id; one on a robot
    /// nothing is loading is refused rather than kept waiting.
    #[test]
    fn a_joint_control_lands_on_the_robot_it_names() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let other = app.world_mut().spawn(Transform::default()).id();
        second_robot(&mut app, other);
        for (key, instance) in [("first", ROBOT), ("second", "robot-1"), ("nobody", "ghost")] {
            send(&mut app, add(key, joint_on(instance)));
        }
        let mut gizmos = app.world_mut().query::<&JointGizmo>();
        let mut on: Vec<(String, Entity)> = gizmos
            .iter(app.world())
            .map(|gizmo| (gizmo.key.clone(), gizmo.target))
            .collect();
        on.sort();
        assert_eq!(on, vec![("first".into(), body), ("second".into(), other)]);
        assert!(
            app.world().resource::<ControlCommands>().0.is_empty(),
            "nothing waits for a robot nothing is loading",
        );
    }

    /// Of two adds under one key, the one asked for last is the control
    /// left, even when the first waits for its robot to load.
    #[test]
    fn a_later_add_under_a_key_outlasts_one_waiting_for_its_robot() {
        let (mut app, body) = scene();
        let other = app.world_mut().spawn(Transform::default()).id();
        second_robot(&mut app, other);
        queue(
            &mut app,
            [add("k", joint_on(ROBOT)), add("k", joint_on("robot-1"))],
        );
        app.update();
        robot(&mut app, body);
        app.update();
        app.update();
        let mut gizmos = app.world_mut().query::<&JointGizmo>();
        let on: Vec<Entity> = gizmos.iter(app.world()).map(|gizmo| gizmo.target).collect();
        assert_eq!(on, vec![other]);
    }

    /// A change sent behind an add waiting for its robot acts on the
    /// control that add makes, not the one under that key it replaces.
    #[test]
    fn a_change_behind_a_waiting_add_reaches_the_control_it_makes() {
        let (mut app, body) = scene();
        let other = app.world_mut().spawn(Transform::default()).id();
        second_robot(&mut app, other);
        send(&mut app, add("k", joint_on("robot-1")));
        queue(
            &mut app,
            [
                add("k", joint_on(ROBOT)),
                ControlCommand::Show {
                    key: "k".into(),
                    visible: false,
                },
            ],
        );
        app.update();
        let mut hidden = app
            .world_mut()
            .query_filtered::<&JointGizmo, With<HiddenByDriver>>();
        assert_eq!(
            hidden.iter(app.world()).count(),
            0,
            "the old control is left alone"
        );

        robot(&mut app, body);
        app.update();
        app.update();
        let mut gizmos = app
            .world_mut()
            .query::<(&JointGizmo, Has<HiddenByDriver>)>();
        let on: Vec<(Entity, bool)> = gizmos
            .iter(app.world())
            .map(|(gizmo, hidden)| (gizmo.target, hidden))
            .collect();
        assert_eq!(on, vec![(body, true)]);
    }

    /// A remove sent behind an add waiting for its robot takes the control
    /// away at once, the one the add would have replaced included, and the
    /// add with it.
    #[test]
    fn a_remove_behind_a_waiting_add_cancels_both() {
        let (mut app, body) = scene();
        let other = app.world_mut().spawn(Transform::default()).id();
        second_robot(&mut app, other);
        send(&mut app, add("k", joint_on("robot-1")));
        queue(
            &mut app,
            [
                add("k", joint_on(ROBOT)),
                ControlCommand::Remove { key: "k".into() },
            ],
        );
        app.update();
        assert!(only::<JointGizmo>(&mut app).is_none(), "removed at once");

        robot(&mut app, body);
        app.update();
        app.update();
        assert!(
            only::<JointGizmo>(&mut app).is_none(),
            "and not brought back"
        );
    }

    #[test]
    fn a_joint_control_reports_its_drag_and_moves_nothing() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        send(&mut app, add(TILT, joint_on(ROBOT)));
        let gizmo = only::<JointGizmo>(&mut app).expect("a joint control");
        let joint = app.world().get::<JointGizmo>(gizmo).unwrap();
        assert_eq!(
            (joint.target, joint.key.as_str(), joint.apply),
            (body, TILT, false)
        );
        assert!(app.world().get::<JointRequest>(gizmo).is_none(), "finished");

        app.world_mut().trigger(JointDrag {
            entity: gizmo,
            target: body,
            phase: DragPhase::Move,
            gesture: 0,
            value: 0.3,
        });
        let events = events(&mut app);
        assert_eq!(events.len(), 1);
        assert_eq!((events[0].key.as_str(), events[0].value), (TILT, Some(0.3)));
        assert_eq!(
            *app.world().get::<Transform>(body).unwrap(),
            Transform::default()
        );
    }

    /// A joint the host's rig does not have is dropped, and its key freed,
    /// whatever the add replaced; so is every joint add in a host with no
    /// resolver at all.
    #[test]
    fn a_joint_its_host_cannot_resolve_is_dropped() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        send(
            &mut app,
            add(
                "k",
                Control::Joint {
                    instance: ROBOT.into(),
                    joint: "nothing drives this".into(),
                    color: None,
                    size: None,
                },
            ),
        );
        assert!(only::<JointGizmo>(&mut app).is_none());
        assert!(only::<JointRequest>(&mut app).is_none(), "and its request");
        app.update();
        assert_eq!(app.world().resource::<Controls>().keys().count(), 0);

        let mut app = bare();
        let camera = app.world_mut().spawn(ControlCamera).id();
        let body = app.world_mut().spawn(Transform::default()).id();
        app.world_mut().spawn((rig(body), instance(ROBOT, camera)));
        send(&mut app, add(TILT, joint_on(ROBOT)));
        assert!(only::<JointGizmo>(&mut app).is_none());
        assert!(only::<JointRequest>(&mut app).is_none());
    }

    /// A joint control takes the colour its add asked for over the joint's
    /// own, which is the host's.
    #[test]
    fn a_joint_control_takes_the_colour_asked_for() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        send(
            &mut app,
            add(
                TILT,
                Control::Joint {
                    instance: ROBOT.into(),
                    joint: TILT.into(),
                    color: Some("#ff0000".into()),
                    size: None,
                },
            ),
        );
        let joint = only::<JointGizmo>(&mut app).unwrap();
        assert_eq!(app.world().get::<JointGizmo>(joint).unwrap().colour, red());
    }

    #[test]
    fn a_control_added_before_the_robot_arrives_waits_for_it() {
        let (mut app, body) = scene();
        send(&mut app, add(TILT, joint_on(ROBOT)));
        send(
            &mut app,
            ControlCommand::Show {
                key: TILT.into(),
                visible: false,
            },
        );
        assert!(only::<JointGizmo>(&mut app).is_none());
        robot(&mut app, body);
        app.update();
        let gizmo = only::<JointGizmo>(&mut app).expect("added once the robot arrived");
        assert_eq!(
            app.world().get::<Visibility>(gizmo),
            Some(&Visibility::Hidden)
        );
    }

    #[test]
    fn a_control_whose_gizmo_went_with_its_robot_is_forgotten() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        send(&mut app, add(TILT, joint_on(ROBOT)));
        let gizmo = only::<JointGizmo>(&mut app).unwrap();
        // What taking a robot away does to the controls on it.
        app.world_mut().despawn(gizmo);
        let hide = || ControlCommand::Show {
            key: TILT.into(),
            visible: false,
        };
        // Not shown on a dead entity: the key is free, and with nothing to
        // add it, the show goes.
        send(&mut app, hide());
        assert_eq!(app.world().resource::<Controls>().keys().count(), 0);
        assert!(app.world().resource::<ControlCommands>().0.is_empty());

        // A show sent before the add that comes for the key again waits for
        // it, and reaches the control it makes.
        queue(&mut app, [hide(), add(TILT, joint_on(ROBOT))]);
        app.update();
        let gizmo = only::<JointGizmo>(&mut app).expect("added again");
        assert!(app.world().get::<HiddenByDriver>(gizmo).is_some());
    }

    #[test]
    fn a_goal_reports_its_drag_in_the_driver_s_frame_and_moves_when_set() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let at = Pose {
            position: [1.0, 2.0, 3.0],
            rotation: no_turn(),
        };
        send(
            &mut app,
            add(
                "goal",
                Control::Transform {
                    target: Target::Goal {
                        pose: at,
                        view: None,
                    },
                    modes: None,
                    local: false,
                    colors: None,
                    size: None,
                    face_viewer: false,
                },
            ),
        );
        let gizmo = only::<TransformGizmo>(&mut app).expect("a transform control");
        let goal = app.world().get::<TransformGizmo>(gizmo).unwrap().target;
        assert!(!app.world().get::<TransformGizmo>(gizmo).unwrap().apply);
        assert_eq!(
            *app.world().get::<Transform>(goal).unwrap(),
            at.to_world(&y_up())
        );

        // A drag one metre up the driver's Z, which is the world's Y.
        let asked = Transform::from_translation(at.to_world(&y_up()).translation + Vec3::Y);
        app.world_mut().trigger(GizmoDrag {
            entity: gizmo,
            target: goal,
            handle: GizmoHandle::Arrow(2),
            phase: DragPhase::End,
            gesture: 0,
            transform: asked,
        });
        let events = events(&mut app);
        let pose = events[0].pose.expect("a pose");
        assert!(Vec3::from(pose.position).abs_diff_eq(Vec3::new(1.0, 2.0, 4.0), 1e-5));
        assert_eq!(
            *app.world().get::<Transform>(goal).unwrap(),
            at.to_world(&y_up())
        );

        send(
            &mut app,
            ControlCommand::SetPose {
                key: "goal".into(),
                pose,
                scale: Some([2.0, 1.0, 1.0]),
            },
        );
        let moved = app.world().get::<Transform>(goal).unwrap();
        assert!(moved.translation.abs_diff_eq(asked.translation, 1e-5));
        assert_eq!(moved.scale, Vec3::new(2.0, 1.0, 1.0));

        send(&mut app, ControlCommand::Remove { key: "goal".into() });
        assert!(app.world().get_entity(gizmo).is_err());
        assert!(app.world().get_entity(goal).is_err());
    }

    /// A parent moved this frame is read where it now stands, not where it
    /// was last propagated.
    #[test]
    fn a_parent_moved_this_frame_is_read_where_it_now_stands() {
        use bevy::ecs::system::RunSystemOnce;
        let mut world = World::new();
        let grand = world
            .spawn((
                Transform::from_xyz(10.0, 0.0, 0.0),
                GlobalTransform::IDENTITY,
            ))
            .id();
        let parent = world
            .spawn((
                Transform::from_xyz(0.0, 1.0, 0.0),
                GlobalTransform::IDENTITY,
                ChildOf(grand),
            ))
            .id();
        let at = world
            .run_system_once(
                move |transforms: Query<&Transform>,
                      parents: Query<&ChildOf>,
                      globals: Query<&GlobalTransform>| {
                    current_global(
                        parent,
                        |entity| transforms.get(entity).ok().copied(),
                        &parents,
                        &globals,
                    )
                },
            )
            .unwrap()
            .unwrap();
        assert!(Vec3::from(at.translation).abs_diff_eq(Vec3::new(10.0, 1.0, 0.0), 1e-6));
    }

    /// A transform control on the fixture's robot as a whole.
    fn on_robot() -> Control {
        Control::Transform {
            target: Target::Instance {
                instance: ROBOT.into(),
            },
            modes: None,
            local: false,
            colors: None,
            size: None,
            face_viewer: false,
        }
    }

    /// A control given no size takes its view's, one given a size keeps it
    /// over its view's, and a `configure` resizes it; a joint control takes
    /// its view's joint size.
    #[test]
    fn a_control_takes_its_views_size_unless_given_its_own() {
        let (mut app, body) = scene();
        let view = camera(app.world_mut());
        app.world_mut().entity_mut(view).insert(ControlSize {
            transform: Some(GizmoSize::Pixels(80.0)),
            joint: Some(GizmoSize::World(0.5)),
        });
        robot(&mut app, body);
        let whole = |size: Option<GizmoSize>| Control::Transform {
            target: Target::Instance {
                instance: ROBOT.into(),
            },
            modes: None,
            local: false,
            colors: None,
            size,
            face_viewer: false,
        };
        send(&mut app, add("viewed", whole(None)));
        send(&mut app, add("own", whole(Some(GizmoSize::World(1.0)))));
        send(&mut app, add("joint", joint_on(ROBOT)));
        let size_of = |app: &App, key: &str| {
            let gizmo = app.world().resource::<Controls>().gizmo(key).unwrap();
            app.world()
                .get::<TransformGizmo>(gizmo)
                .map(|gizmo| gizmo.size)
                .or_else(|| app.world().get::<JointGizmo>(gizmo).map(|gizmo| gizmo.size))
                .unwrap()
        };
        assert_eq!(size_of(&app, "viewed"), GizmoSize::Pixels(80.0));
        assert_eq!(size_of(&app, "own"), GizmoSize::World(1.0));
        assert_eq!(size_of(&app, "joint"), GizmoSize::World(0.5));

        send(
            &mut app,
            ControlCommand::Configure {
                key: "viewed".into(),
                color: None,
                colors: None,
                local: None,
                opacity: None,
                interactive: None,
                size: Some(GizmoSize::Pixels(40.0)),
                face_viewer: None,
            },
        );
        assert_eq!(size_of(&app, "viewed"), GizmoSize::Pixels(40.0));
        // The view's size no longer reaches a control given its own.
        app.world_mut().entity_mut(view).insert(ControlSize {
            transform: Some(GizmoSize::Pixels(120.0)),
            joint: None,
        });
        app.update();
        assert_eq!(size_of(&app, "viewed"), GizmoSize::Pixels(40.0));

        // A view's size taken away goes back to the size each was made at.
        let joint_made_at = app.world().resource::<Controls>().0["joint"]
            .made_at
            .expect("a resolved joint control knows the size it was made at");
        assert_ne!(
            joint_made_at,
            GizmoSize::World(0.5),
            "the host's, not the view's"
        );
        send(&mut app, add("viewed-again", whole(None)));
        assert_eq!(size_of(&app, "viewed-again"), GizmoSize::Pixels(120.0));
        app.world_mut().entity_mut(view).remove::<ControlSize>();
        app.update();
        assert_eq!(
            size_of(&app, "viewed-again"),
            crate::transform_gizmo::GIZMO_SIZE
        );
        assert_eq!(size_of(&app, "joint"), joint_made_at);
        assert_eq!(
            size_of(&app, "viewed"),
            GizmoSize::Pixels(40.0),
            "its own stays"
        );
    }

    /// A driver writes a size in pixels or in metres.
    #[test]
    fn a_size_reads_in_pixels_or_metres() {
        let read = |json: serde_json::Value| -> GizmoSize { serde_json::from_value(json).unwrap() };
        assert_eq!(
            read(serde_json::json!({ "pixels": 80 })),
            GizmoSize::Pixels(80.0)
        );
        assert_eq!(
            read(serde_json::json!({ "metres": 0.25 })),
            GizmoSize::World(0.25)
        );
    }

    fn goal_at(position: [f32; 3]) -> Control {
        Control::Transform {
            target: Target::Goal {
                pose: Pose {
                    position,
                    rotation: no_turn(),
                },
                view: None,
            },
            modes: None,
            local: false,
            colors: None,
            size: None,
            face_viewer: false,
        }
    }

    #[test]
    fn a_pose_set_in_the_frame_a_goal_is_added_is_not_lost() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let moved_to = Pose {
            position: [0.0, 0.0, 2.0],
            rotation: no_turn(),
        };
        queue(
            &mut app,
            [
                add("goal", goal_at([0.0, 0.0, 1.0])),
                ControlCommand::SetPose {
                    key: "goal".into(),
                    pose: moved_to,
                    scale: None,
                },
            ],
        );
        app.update();
        app.update();
        let gizmo = only::<TransformGizmo>(&mut app).unwrap();
        let goal = app.world().get::<TransformGizmo>(gizmo).unwrap().target;
        let at = app.world().get::<Transform>(goal).unwrap();
        assert!(at
            .translation
            .abs_diff_eq(moved_to.to_world(&y_up()).translation, 1e-5));
    }

    #[test]
    fn a_control_removed_before_the_robot_arrives_never_appears() {
        let (mut app, body) = scene();
        queue(
            &mut app,
            [
                add(TILT, joint_on(ROBOT)),
                ControlCommand::Show {
                    key: TILT.into(),
                    visible: false,
                },
            ],
        );
        app.update();
        send(&mut app, ControlCommand::Remove { key: TILT.into() });
        robot(&mut app, body);
        app.update();
        assert!(only::<JointGizmo>(&mut app).is_none());
        assert!(app.world().resource::<ControlCommands>().0.is_empty());
    }

    #[test]
    fn a_hidden_control_is_marked_as_the_driver_s_doing() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        send(&mut app, add(TILT, joint_on(ROBOT)));
        let gizmo = only::<JointGizmo>(&mut app).unwrap();
        let show = |visible| ControlCommand::Show {
            key: TILT.into(),
            visible,
        };
        send(&mut app, show(false));
        assert!(app.world().get::<HiddenByDriver>(gizmo).is_some());
        send(&mut app, show(true));
        assert!(app.world().get::<HiddenByDriver>(gizmo).is_none());
        assert_eq!(
            app.world().get::<Visibility>(gizmo),
            Some(&Visibility::Inherited)
        );
    }

    #[test]
    fn a_y_up_world_s_up_is_the_driver_s_z() {
        let pose = Pose::from_world(&Transform::from_xyz(1.0, 2.0, 3.0), &y_up());
        assert!(Vec3::from(pose.position).abs_diff_eq(Vec3::new(1.0, -3.0, 2.0), 1e-6));
    }

    #[test]
    fn a_turn_about_a_y_up_world_s_up_is_a_turn_about_the_driver_s_z() {
        let turn = Quat::from_rotation_y(0.7);
        let pose = Pose::from_world(&Transform::from_rotation(turn), &y_up());
        assert!(Quat::from_array(pose.rotation).abs_diff_eq(Quat::from_rotation_z(0.7), 1e-6));
    }

    /// With no frame given, the world is the driver's own: Z up, and a pose
    /// is where the transform says.
    #[test]
    fn a_z_up_world_is_the_driver_s_frame_unchanged() {
        let frame = DriverFrame::default();
        let transform =
            Transform::from_xyz(1.0, 2.0, 3.0).with_rotation(Quat::from_rotation_z(0.4));
        let pose = Pose::from_world(&transform, &frame);
        assert_eq!(pose.position, [1.0, 2.0, 3.0]);
        assert!(Quat::from_array(pose.rotation).abs_diff_eq(transform.rotation, 1e-6));
        assert_eq!(frame.axes(), [Dir3::X, Dir3::Y, Dir3::Z]);
    }

    /// A transform control's handles are labelled as the driver names the
    /// axes, whichever way up the world is.
    #[test]
    fn handles_are_labelled_from_the_driver_s_frame() {
        assert_eq!(y_up().axes(), TransformGizmo::Z_UP);
        let (mut app, body) = scene();
        robot(&mut app, body);
        let gizmo = add_goal(&mut app, "goal", None);
        assert_eq!(
            app.world().get::<TransformGizmo>(gizmo).unwrap().axes,
            TransformGizmo::Z_UP
        );
    }

    fn add_goal(app: &mut App, key: &str, colors: Option<[String; 3]>) -> Entity {
        let mut control = goal_at([0.0, 0.0, 1.0]);
        if let Control::Transform { colors: c, .. } = &mut control {
            *c = colors;
        }
        send(app, add(key, control));
        only::<TransformGizmo>(app).expect("a transform control")
    }

    fn red() -> Color {
        Srgba::hex("#ff0000").unwrap().into()
    }

    fn drag_goal(app: &mut App, gizmo: Entity, phase: DragPhase, gesture: u32) {
        let target = app.world().get::<TransformGizmo>(gizmo).unwrap().target;
        app.world_mut().trigger(GizmoDrag {
            entity: gizmo,
            target,
            handle: GizmoHandle::Arrow(0),
            phase,
            gesture,
            transform: Transform::default(),
        });
    }

    /// A drag in hand on a control that is removed, or replaced under its
    /// key, closes with one `cancel` under its key and gesture, carrying the
    /// modifiers held and no value. A drag that has already closed sends
    /// nothing more when its control goes.
    #[test]
    fn a_drag_on_a_control_taken_away_is_cancelled() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        app.world_mut().resource_mut::<Modifiers>().shift = true;
        let cancel = |gesture| ControlEvent {
            key: "goal".into(),
            phase: Phase::Cancel,
            gesture,
            modifiers: Modifiers {
                shift: true,
                ..default()
            },
            value: None,
            pose: None,
            scale: None,
        };
        for replace in [false, true] {
            let gizmo = add_goal(&mut app, "goal", None);
            drag_goal(&mut app, gizmo, DragPhase::Start, 7);
            let started = events(&mut app);
            assert_eq!(started.len(), 1);
            assert_eq!((started[0].phase, started[0].gesture), (Phase::Start, 7));
            assert!(started[0].modifiers.shift, "the modifiers held");

            if replace {
                send(&mut app, add("goal", goal_at([1.0, 0.0, 1.0])));
            } else {
                send(&mut app, ControlCommand::Remove { key: "goal".into() });
            }
            assert_eq!(events(&mut app), [cancel(7)], "replaced: {replace}");
            app.update();
            assert!(events(&mut app).is_empty(), "closed once");
            if replace {
                send(&mut app, ControlCommand::Remove { key: "goal".into() });
            }
        }

        // Cancelled by its gizmo: no value, and nothing more when it goes.
        let gizmo = add_goal(&mut app, "goal", None);
        drag_goal(&mut app, gizmo, DragPhase::Start, 8);
        drag_goal(&mut app, gizmo, DragPhase::Cancel, 8);
        let sent = events(&mut app);
        assert_eq!(sent.last(), Some(&cancel(8)));
        send(&mut app, ControlCommand::Remove { key: "goal".into() });
        assert!(events(&mut app).is_empty(), "already closed");

        // Ended: nothing more when it goes.
        let gizmo = add_goal(&mut app, "goal", None);
        drag_goal(&mut app, gizmo, DragPhase::Start, 9);
        drag_goal(&mut app, gizmo, DragPhase::End, 9);
        events(&mut app);
        send(&mut app, ControlCommand::Remove { key: "goal".into() });
        assert!(events(&mut app).is_empty(), "already ended");
    }

    /// A transform control keeps its arrows on the positive side unless
    /// added or configured to face the viewer, and a configure can turn
    /// that back off.
    #[test]
    fn a_transform_control_faces_the_viewer_only_when_asked() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let gizmo = add_goal(&mut app, "goal", None);
        let facing = |app: &App| {
            app.world()
                .get::<TransformGizmo>(gizmo)
                .unwrap()
                .face_viewer
        };
        assert!(!facing(&app), "positive by default");
        let configure = |face_viewer| ControlCommand::Configure {
            key: "goal".into(),
            color: None,
            colors: None,
            local: None,
            opacity: None,
            interactive: None,
            size: None,
            face_viewer,
        };
        send(&mut app, configure(Some(true)));
        assert!(facing(&app));
        send(&mut app, configure(None));
        assert!(facing(&app), "an absent setting changes nothing");
        send(&mut app, configure(Some(false)));
        assert!(!facing(&app));

        let mut control = goal_at([0.0, 0.0, 1.0]);
        if let Control::Transform { face_viewer, .. } = &mut control {
            *face_viewer = true;
        }
        send(&mut app, add("facing", control));
        let added = app.world().resource::<Controls>().0["facing"].gizmo;
        assert!(
            app.world()
                .get::<TransformGizmo>(added)
                .unwrap()
                .face_viewer,
            "asked on its add"
        );
    }

    /// A control takes the host's colours when added, and changes them, and
    /// a transform control its axes, when configured; a bad colour changes
    /// nothing.
    #[test]
    fn a_control_takes_the_host_s_colours_and_axes() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let hex = |s: &str| s.to_string();
        let gizmo = add_goal(
            &mut app,
            "goal",
            Some([hex("#ff0000"), hex("#00ff00"), hex("#0000ff")]),
        );
        assert_eq!(
            app.world().get::<TransformGizmo>(gizmo).unwrap().colours[0],
            red()
        );
        send(
            &mut app,
            ControlCommand::Configure {
                key: "goal".into(),
                color: None,
                colors: Some([hex("#0000ff"), hex("not a colour"), hex("#ff0000")]),
                local: Some(true),
                opacity: None,
                interactive: None,
                size: None,
                face_viewer: None,
            },
        );
        let configured = app.world().get::<TransformGizmo>(gizmo).unwrap();
        assert_eq!(configured.colours[0], red(), "a bad colour changes none");
        assert_eq!(configured.space, GizmoSpace::Local);

        send(&mut app, add(TILT, joint_on(ROBOT)));
        let joint = only::<JointGizmo>(&mut app).unwrap();
        send(
            &mut app,
            ControlCommand::Configure {
                key: TILT.into(),
                color: Some(hex("#00ff00")),
                colors: None,
                local: None,
                opacity: None,
                interactive: None,
                size: None,
                face_viewer: None,
            },
        );
        assert_eq!(
            app.world().get::<JointGizmo>(joint).unwrap().colour,
            Srgba::hex("#00ff00").unwrap().into()
        );
    }

    /// A highlight lights one handle, or every one, until it is turned off;
    /// on a joint control it shows the control in hand.
    #[test]
    fn a_highlight_lights_a_handle_or_the_whole_control() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let gizmo = add_goal(&mut app, "goal", None);
        let highlight = |app: &mut App, handle, on| {
            send(
                app,
                ControlCommand::Highlight {
                    key: "goal".into(),
                    handle,
                    on,
                },
            );
            app.world()
                .get::<TransformGizmo>(gizmo)
                .unwrap()
                .highlight
                .clone()
        };
        let ring = GizmoHandle::Ring(1);
        assert_eq!(highlight(&mut app, Some(ring), true), vec![ring]);
        assert_eq!(highlight(&mut app, Some(ring), true), vec![ring], "once");
        assert_eq!(highlight(&mut app, Some(ring), false), vec![]);
        assert_eq!(
            highlight(&mut app, None, true).len(),
            GizmoHandle::ALL.len()
        );
        assert_eq!(highlight(&mut app, None, false), vec![]);

        send(&mut app, add(TILT, joint_on(ROBOT)));
        send(
            &mut app,
            ControlCommand::Highlight {
                key: TILT.into(),
                handle: None,
                on: true,
            },
        );
        let joint = only::<JointGizmo>(&mut app).unwrap();
        assert!(app.world().get::<JointGizmo>(joint).unwrap().show_in_hand);
    }

    /// A highlight or a change sent in the frame its control is added applies
    /// once the control exists, rather than being lost.
    #[test]
    fn a_change_sent_with_the_add_is_not_lost() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        queue(
            &mut app,
            [
                add("goal", goal_at([0.0, 0.0, 1.0])),
                ControlCommand::Highlight {
                    key: "goal".into(),
                    handle: None,
                    on: true,
                },
                ControlCommand::Configure {
                    key: "goal".into(),
                    color: None,
                    colors: None,
                    local: Some(true),
                    opacity: None,
                    interactive: None,
                    size: None,
                    face_viewer: None,
                },
            ],
        );
        app.update();
        app.update();
        let gizmo = only::<TransformGizmo>(&mut app).unwrap();
        let gizmo = app.world().get::<TransformGizmo>(gizmo).unwrap();
        assert_eq!(gizmo.highlight.len(), GizmoHandle::ALL.len());
        assert_eq!(gizmo.space, GizmoSpace::Local);
    }

    /// A highlight or a change for a key nothing will add is dropped, rather
    /// than waiting, and asking for a frame, for good; one for a key whose add
    /// is still coming waits for it.
    #[test]
    fn a_change_for_a_key_nothing_will_add_is_dropped() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        app.update();
        frames_asked(&mut app);
        send(
            &mut app,
            ControlCommand::Highlight {
                key: "nobody".into(),
                handle: None,
                on: true,
            },
        );
        send(
            &mut app,
            ControlCommand::Configure {
                key: "nobody".into(),
                color: Some("#ff0000".into()),
                colors: None,
                local: None,
                opacity: None,
                interactive: None,
                size: None,
                face_viewer: None,
            },
        );
        send(
            &mut app,
            ControlCommand::SetPose {
                key: "nobody".into(),
                pose: Pose {
                    position: [0.0, 0.0, 1.0],
                    rotation: no_turn(),
                },
                scale: None,
            },
        );
        assert!(app.world().resource::<ControlCommands>().0.is_empty());
        // Read before another update: a frame request outlives only two.
        assert_eq!(
            frames_asked(&mut app),
            0,
            "a dropped change asks for no frame"
        );
        app.update();
        app.update();
        assert_eq!(frames_asked(&mut app), 0, "the view can idle");

        // Before the robot, a highlight waits for the add it will follow.
        let (mut app, body) = scene();
        send(
            &mut app,
            ControlCommand::Highlight {
                key: TILT.into(),
                handle: None,
                on: true,
            },
        );
        assert_eq!(app.world().resource::<ControlCommands>().0.len(), 1);
        robot(&mut app, body);
        send(&mut app, add(TILT, joint_on(ROBOT)));
        app.update();
        let joint = only::<JointGizmo>(&mut app).unwrap();
        assert!(app.world().get::<JointGizmo>(joint).unwrap().show_in_hand);
    }

    fn configure(key: &str, opacity: Option<f32>, interactive: Option<bool>) -> ControlCommand {
        ControlCommand::Configure {
            key: key.into(),
            color: None,
            colors: None,
            local: None,
            opacity,
            interactive,
            size: None,
            face_viewer: None,
        }
    }

    fn frame_length(app: &mut App, secs: f32) {
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
            secs,
        )));
    }

    /// A control fades to the opacity its host asks for, over the fade's
    /// length rather than at once, and stays there; a frame that comes long
    /// after the last takes one step of it, not the whole fade.
    #[test]
    fn a_control_fades_to_the_opacity_asked_for() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        frame_length(&mut app, FADE_SECS / 4.0);
        let gizmo = add_goal(&mut app, "goal", None);
        let opacity = |app: &App| app.world().get::<TransformGizmo>(gizmo).unwrap().opacity;
        assert_eq!(opacity(&app), 1.0);

        send(&mut app, configure("goal", Some(0.0), None));
        let first = opacity(&app);
        assert!(first < 1.0 && first > 0.0, "part of the way: {first}");
        // A view may draw only on change, so a fade asks for its own frames
        // while it runs, and none once it has arrived.
        frames_asked(&mut app);
        app.update();
        assert!(frames_asked(&mut app) > 0, "a frame asked for mid-fade");
        for _ in 0..8 {
            app.update();
        }
        assert_eq!(opacity(&app), 0.0, "and there in the end");
        frames_asked(&mut app);
        app.update();
        assert_eq!(frames_asked(&mut app), 0, "none once it has arrived");

        // A long quiet spell before the next change is one step, not all.
        frame_length(&mut app, 5.0);
        send(&mut app, configure("goal", Some(1.0), None));
        let after_a_pause = opacity(&app);
        assert!(after_a_pause < 1.0, "{after_a_pause}");
    }

    /// A small change takes the whole fade, as a large one does, and a fade
    /// sent somewhere new part way starts over from where it had got to.
    #[test]
    fn every_fade_takes_the_fade_s_length() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        // Under the thirtieth of a second a step is held to.
        frame_length(&mut app, FADE_SECS / 8.0);
        let gizmo = add_goal(&mut app, "goal", None);
        let opacity = |app: &App| app.world().get::<TransformGizmo>(gizmo).unwrap().opacity;
        let near = |app: &App, wanted: f32| {
            let now = opacity(app);
            assert!((now - wanted).abs() < 1e-4, "{now}, not {wanted}");
        };
        let frames = |app: &mut App, count: usize| {
            for _ in 0..count {
                app.update();
            }
        };

        send(&mut app, configure("goal", Some(0.9), None));
        near(&app, 0.9875);
        frames(&mut app, 3);
        near(&app, 0.95);
        frames(&mut app, 4);
        assert_eq!(opacity(&app), 0.9, "there at the fade's length");
        // Exactly there, though the steps on the way round.
        send(&mut app, configure("goal", Some(0.1), None));
        frames(&mut app, 7);
        assert_eq!(opacity(&app), 0.1, "exactly what was asked for");
        send(&mut app, configure("goal", Some(0.9), None));
        frames(&mut app, 7);
        assert_eq!(opacity(&app), 0.9);

        // Sent somewhere new half way, it starts over from where it had got
        // to: half way from 0.9 to 0.1 is 0.5, so a fade to 0.5 holds there.
        send(&mut app, configure("goal", Some(0.1), None));
        frames(&mut app, 3);
        near(&app, 0.5);
        send(&mut app, configure("goal", Some(0.5), None));
        near(&app, 0.5);

        // Asked again for where it is already going, it runs on.
        send(&mut app, configure("goal", Some(1.0), None));
        near(&app, 0.5625);
        send(&mut app, configure("goal", Some(1.0), None));
        near(&app, 0.625);
    }

    /// A control sent with a change in the same batch as its add first
    /// shows with it: at the opacity asked for, not faded down from full,
    /// and inert from its first frame. A joint control resolved by its host
    /// between the passes does too.
    #[test]
    fn a_control_added_dimmed_and_inert_first_shows_so() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        queue(
            &mut app,
            [
                add("goal", goal_at([0.0, 0.0, 1.0])),
                add(TILT, joint_on(ROBOT)),
                configure("goal", Some(0.25), Some(false)),
                configure(TILT, Some(0.25), Some(false)),
            ],
        );
        app.update();
        let goal = only::<TransformGizmo>(&mut app).unwrap();
        let joint = only::<JointGizmo>(&mut app).unwrap();
        let goal = app.world().get::<TransformGizmo>(goal).unwrap();
        assert_eq!((goal.opacity, goal.interactive), (0.25, false));
        let joint = app.world().get::<JointGizmo>(joint).unwrap();
        assert_eq!((joint.opacity, joint.interactive), (0.25, false));

        // Once shown, a change fades as ever.
        send(&mut app, configure("goal", Some(1.0), None));
        let goal = only::<TransformGizmo>(&mut app).unwrap();
        let now = app.world().get::<TransformGizmo>(goal).unwrap().opacity;
        assert!(now > 0.25 && now < 1.0, "{now}");
    }

    /// The commands run before the gizmos' own systems, so a control made
    /// inert is inert to the drag handling of the same frame. Ambiguity is
    /// an error here, so the order has to be stated rather than happen to
    /// fall out.
    #[test]
    fn commands_run_before_the_gizmos_own_systems() {
        use bevy::ecs::schedule::{LogLevel, ScheduleBuildSettings};

        #[derive(Resource, Default)]
        struct Seen(Vec<bool>);

        let (mut app, body) = scene();
        robot(&mut app, body);
        add_goal(&mut app, "goal", None);
        app.init_resource::<Seen>().add_systems(
            Update,
            (|gizmos: Query<&TransformGizmo>, mut seen: ResMut<Seen>| {
                seen.0.extend(gizmos.iter().map(|gizmo| gizmo.interactive));
            })
            .in_set(TransformGizmoSystems),
        );
        app.edit_schedule(Update, |schedule| {
            schedule.set_build_settings(ScheduleBuildSettings {
                ambiguity_detection: LogLevel::Error,
                ..default()
            });
        });
        app.update();
        assert_eq!(app.world().resource::<Seen>().0, [true]);
        send(&mut app, configure("goal", None, Some(false)));
        assert_eq!(app.world().resource::<Seen>().0, [true, false]);
    }

    /// A host's systems in each [`InstanceSystems`] stage run in the stages'
    /// order, with control commands after indexing and before values are
    /// applied, and none of it ambiguous.
    #[test]
    fn control_commands_run_between_indexing_and_applying() {
        use bevy::ecs::schedule::{LogLevel, ScheduleBuildSettings};

        #[derive(Resource, Default)]
        struct Order(Vec<&'static str>);

        let mut app = bare();
        app.init_resource::<Order>();
        for (name, set) in [
            ("derive", InstanceSystems::Derive),
            ("apply", InstanceSystems::Apply),
            ("index", InstanceSystems::Index),
            ("intake", InstanceSystems::Intake),
        ] {
            app.add_systems(
                Update,
                (move |mut order: ResMut<Order>| order.0.push(name)).in_set(set),
            );
        }
        app.add_systems(
            Update,
            (|mut order: ResMut<Order>| order.0.push("commands"))
                .in_set(ControlSystems::Commands)
                .after(run_passes),
        );
        app.edit_schedule(Update, |schedule| {
            schedule.set_build_settings(ScheduleBuildSettings {
                ambiguity_detection: LogLevel::Error,
                ..default()
            });
        });
        app.update();
        assert_eq!(
            app.world().resource::<Order>().0,
            ["intake", "index", "commands", "apply", "derive"]
        );
    }

    /// Hidden part way through a fade and then given another opacity, a
    /// control takes it at once, and the fade it was on goes.
    #[test]
    fn a_control_hidden_part_way_through_a_fade_takes_a_new_opacity_at_once() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        frame_length(&mut app, FADE_SECS / 8.0);
        let gizmo = add_goal(&mut app, "goal", None);
        let opacity = |app: &App| app.world().get::<TransformGizmo>(gizmo).unwrap().opacity;
        send(&mut app, configure("goal", Some(0.5), None));
        let part_way = opacity(&app);
        assert!(part_way > 0.5 && part_way < 1.0, "{part_way}");
        send(
            &mut app,
            ControlCommand::Show {
                key: "goal".into(),
                visible: false,
            },
        );
        send(&mut app, configure("goal", Some(0.25), None));
        for _ in 0..8 {
            assert_eq!(opacity(&app), 0.25);
            app.update();
        }
    }

    /// Hidden and dimmed in one batch, a control takes the opacity at once,
    /// as it would a frame later.
    #[test]
    fn a_control_hidden_and_dimmed_together_takes_the_opacity_at_once() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let gizmo = add_goal(&mut app, "goal", None);
        queue(
            &mut app,
            [
                ControlCommand::Show {
                    key: "goal".into(),
                    visible: false,
                },
                configure("goal", Some(0.25), None),
            ],
        );
        app.update();
        let opacity = app.world().get::<TransformGizmo>(gizmo).unwrap().opacity;
        assert_eq!(opacity, 0.25);
    }

    /// A control added to a ghosted instance is hidden from its first frame,
    /// so an opacity sent the frame after takes hold at once; a gizmo a host
    /// spawned for itself is left as it is.
    #[test]
    fn a_control_added_on_a_ghost_is_hidden_and_takes_an_opacity_at_once() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        app.world_mut().entity_mut(body).insert(Ghost);
        let camera = camera(app.world_mut());
        let own = app
            .world_mut()
            .spawn((TransformGizmo::new(body, camera), Visibility::Visible))
            .id();
        send(&mut app, add(TILT, joint_on(ROBOT)));
        let joint = only::<JointGizmo>(&mut app).unwrap();
        assert_eq!(
            app.world().get::<Visibility>(joint),
            Some(&Visibility::Hidden),
            "hidden in the frame it is added"
        );
        send(&mut app, configure(TILT, Some(0.25), None));
        assert_eq!(app.world().get::<JointGizmo>(joint).unwrap().opacity, 0.25);
        assert_eq!(
            app.world().get::<Visibility>(own),
            Some(&Visibility::Visible),
            "the host's own gizmo is the host's"
        );

        // And shown again once the ghosting goes.
        app.world_mut().entity_mut(body).remove::<Ghost>();
        app.update();
        assert_eq!(
            app.world().get::<Visibility>(joint),
            Some(&Visibility::Inherited)
        );
    }

    /// A joint control fades as a transform control does.
    #[test]
    fn a_joint_control_fades() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        frame_length(&mut app, FADE_SECS / 8.0);
        send(&mut app, add(TILT, joint_on(ROBOT)));
        let joint = only::<JointGizmo>(&mut app).unwrap();
        let opacity = |app: &App| app.world().get::<JointGizmo>(joint).unwrap().opacity;
        send(&mut app, configure(TILT, Some(0.5), None));
        let part_way = opacity(&app);
        assert!(part_way > 0.5 && part_way < 1.0, "{part_way}");
        for _ in 0..7 {
            app.update();
        }
        assert_eq!(opacity(&app), 0.5);
    }

    /// An instance that arrives through commands before the passes leaves
    /// every pass to a control waiting on it: a change sent with its add
    /// holds from the control's first frame.
    #[test]
    fn a_control_waiting_on_its_robot_first_shows_as_asked() {
        #[derive(Resource)]
        struct Arriving(Option<Entity>);

        let (mut app, body) = scene();
        queue(
            &mut app,
            [
                add("whole", on_robot()),
                configure("whole", Some(0.25), Some(false)),
            ],
        );
        app.update();
        assert!(only::<TransformGizmo>(&mut app).is_none(), "no robot yet");

        app.insert_resource(Arriving(Some(body)));
        app.add_systems(
            Update,
            (|mut arriving: ResMut<Arriving>, mut commands: Commands| {
                if let Some(body) = arriving.0.take() {
                    commands.queue(move |world: &mut World| spawn_robot(world, body));
                }
            })
            .before(ControlSystems::Commands),
        );
        app.update();
        let gizmo = only::<TransformGizmo>(&mut app).expect("added with the robot");
        let gizmo = app.world().get::<TransformGizmo>(gizmo).unwrap();
        assert_eq!((gizmo.opacity, gizmo.interactive), (0.25, false));
    }

    /// A control dimmed while hidden takes the opacity at once, so it first
    /// shows at it, not fading down from where it was.
    #[test]
    fn a_control_dimmed_while_hidden_shows_at_it() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        queue(
            &mut app,
            [
                add("goal", goal_at([0.0, 0.0, 1.0])),
                ControlCommand::Show {
                    key: "goal".into(),
                    visible: false,
                },
            ],
        );
        app.update();
        app.update();
        let gizmo = only::<TransformGizmo>(&mut app).unwrap();
        queue(
            &mut app,
            [
                configure("goal", Some(0.25), None),
                ControlCommand::Show {
                    key: "goal".into(),
                    visible: true,
                },
            ],
        );
        app.update();
        let opacity = app.world().get::<TransformGizmo>(gizmo).unwrap().opacity;
        assert_eq!(opacity, 0.25);
    }

    /// Hidden by anything but the host, as a ghosted target's controls are, a
    /// control takes an opacity at once as well.
    #[test]
    fn a_control_hidden_by_a_ghosting_takes_an_opacity_at_once() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let gizmo = add_goal(&mut app, "goal", None);
        let goal = app.world().get::<TransformGizmo>(gizmo).unwrap().target;
        app.world_mut().entity_mut(goal).insert(Ghost);
        app.update();
        send(&mut app, configure("goal", Some(0.25), None));
        let opacity = app.world().get::<TransformGizmo>(gizmo).unwrap().opacity;
        assert_eq!(opacity, 0.25);
    }

    /// An opacity outside 0 to 1 is held to it, rather than reaching the
    /// gizmo's materials as an alpha they cannot draw.
    #[test]
    fn an_opacity_out_of_range_is_held_to_it() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        frame_length(&mut app, 1.0 / 60.0);
        let gizmo = add_goal(&mut app, "goal", None);
        let opacity = |app: &App| app.world().get::<TransformGizmo>(gizmo).unwrap().opacity;
        let settle = |app: &mut App| {
            for _ in 0..60 {
                app.update();
            }
        };
        send(&mut app, configure("goal", Some(-3.0), None));
        settle(&mut app);
        assert_eq!(opacity(&app), 0.0);
        send(&mut app, configure("goal", Some(7.0), None));
        settle(&mut app);
        assert_eq!(opacity(&app), 1.0);
    }

    /// `interactive` reaches either kind of control, and one command can
    /// carry it for both without a complaint about the other's settings.
    #[test]
    fn a_control_can_be_made_inert_and_live_again() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let goal = add_goal(&mut app, "goal", None);
        send(&mut app, add(TILT, joint_on(ROBOT)));
        let joint = only::<JointGizmo>(&mut app).unwrap();
        for key in ["goal", TILT] {
            send(&mut app, configure(key, None, Some(false)));
        }
        assert!(!app.world().get::<TransformGizmo>(goal).unwrap().interactive);
        assert!(!app.world().get::<JointGizmo>(joint).unwrap().interactive);
        send(&mut app, configure("goal", None, Some(true)));
        assert!(app.world().get::<TransformGizmo>(goal).unwrap().interactive);
    }

    /// A control on an instance is used through that instance's camera; a
    /// goal through the camera the host marks.
    #[test]
    fn a_control_is_used_through_its_instance_s_camera() {
        let (mut app, body) = scene();
        let own = app.world_mut().spawn_empty().id();
        let face = app
            .world_mut()
            .spawn((rig(body), instance("face", own)))
            .id();
        let marked = camera(app.world_mut());
        send(
            &mut app,
            add(
                "whole",
                Control::Transform {
                    target: Target::Instance {
                        instance: "face".into(),
                    },
                    modes: None,
                    local: false,
                    colors: None,
                    size: None,
                    face_viewer: false,
                },
            ),
        );
        send(&mut app, add("goal", goal_at([0.0, 0.0, 1.0])));
        send(&mut app, add(TILT, joint_on("face")));
        let controls = app.world().resource::<Controls>();
        let (whole, goal, joint) = (
            controls.gizmo("whole").unwrap(),
            controls.gizmo("goal").unwrap(),
            controls.gizmo(TILT).unwrap(),
        );
        let whole = app.world().get::<TransformGizmo>(whole).unwrap();
        assert_eq!((whole.target, whole.camera), (face, own));
        assert_eq!(
            app.world().get::<TransformGizmo>(goal).unwrap().camera,
            marked
        );
        assert_eq!(app.world().get::<JointGizmo>(joint).unwrap().camera, own);
    }

    /// A goal added while no camera is marked waits for one, and a change
    /// sent with it holds once it shows.
    #[test]
    fn a_goal_waits_for_a_control_camera() {
        let mut app = bare();
        queue(
            &mut app,
            [
                add("goal", goal_at([0.0, 0.0, 1.0])),
                configure("goal", Some(0.25), None),
            ],
        );
        app.update();
        assert!(only::<TransformGizmo>(&mut app).is_none());
        let camera = app.world_mut().spawn(ControlCamera).id();
        app.update();
        let gizmo = only::<TransformGizmo>(&mut app).expect("added once marked");
        let gizmo = app.world().get::<TransformGizmo>(gizmo).unwrap();
        assert_eq!((gizmo.camera, gizmo.opacity), (camera, 0.25));
    }

    /// A goal waiting for a camera holds the commands behind it under its
    /// key as an add waiting for its instance does: a later add is the
    /// control left.
    #[test]
    fn a_later_add_outlasts_a_goal_waiting_for_a_camera() {
        let mut app = bare();
        app.add_systems(ControlPass, resolve_joints.in_set(ControlSystems::Joints));
        let own = app.world_mut().spawn_empty().id();
        let body = app.world_mut().spawn(Transform::default()).id();
        app.world_mut().spawn((rig(body), instance(ROBOT, own)));
        queue(
            &mut app,
            [
                add("k", goal_at([0.0, 0.0, 1.0])),
                add("k", joint_on(ROBOT)),
            ],
        );
        app.update();
        app.world_mut().spawn(ControlCamera);
        app.update();
        app.update();
        assert!(
            only::<TransformGizmo>(&mut app).is_none(),
            "the goal gave way"
        );
        assert!(only::<JointGizmo>(&mut app).is_some());
    }

    /// A control on an element lands on the element's node.
    #[test]
    fn an_element_control_lands_on_its_node() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let node = app.world_mut().spawn(Transform::default()).id();
        let mut instances = app.world_mut().query::<&mut EntityInstance>();
        let mut robot = instances.single_mut(app.world_mut()).unwrap();
        robot.bindings.elements.insert(
            "head".into(),
            ElementTargets {
                node,
                mesh: None,
                morph: None,
            },
        );
        let on = |id: &str| Control::Transform {
            target: Target::Element {
                instance: ROBOT.into(),
                id: id.into(),
            },
            modes: Some(vec![Mode::Rotate]),
            local: true,
            colors: None,
            size: None,
            face_viewer: false,
        };
        send(&mut app, add("head", on("head")));
        send(&mut app, add("none", on("no such element")));
        let gizmo = only::<TransformGizmo>(&mut app).unwrap();
        let gizmo = app.world().get::<TransformGizmo>(gizmo).unwrap();
        assert_eq!((gizmo.target, gizmo.space), (node, GizmoSpace::Local));
        assert!(gizmo.modes.rotate && !gizmo.modes.translate && !gizmo.modes.scale);
    }

    /// An instance being replaced gives way to its successor under the same
    /// id: a control added while both are there lands on the one that stays.
    #[test]
    fn a_control_lands_on_the_instance_that_stays() {
        let (mut app, _) = scene();
        let camera = camera(app.world_mut());
        app.world_mut()
            .spawn((instance("face", camera), Superseded, Transform::default()));
        let successor = app
            .world_mut()
            .spawn((instance("face", camera), Transform::default()))
            .id();
        send(
            &mut app,
            add(
                "whole",
                Control::Transform {
                    target: Target::Instance {
                        instance: "face".into(),
                    },
                    modes: None,
                    local: false,
                    colors: None,
                    size: None,
                    face_viewer: false,
                },
            ),
        );
        let gizmo = only::<TransformGizmo>(&mut app).unwrap();
        assert_eq!(
            app.world().get::<TransformGizmo>(gizmo).unwrap().target,
            successor
        );
    }

    /// Forgetting an instance drops the adds waiting for it and what waits
    /// behind them, up to a later add under the same key, which stays.
    #[test]
    fn forgetting_an_instance_drops_what_waits_for_it() {
        let show = |key: &str| ControlCommand::Show {
            key: key.into(),
            visible: false,
        };
        let mut commands = ControlCommands(vec![
            add("a", joint_on(ROBOT)),
            show("a"),
            add("b", joint_on("robot-1")),
            show("b"),
            add("a", joint_on("robot-1")),
            show("a"),
        ]);
        commands.forget_instance(ROBOT, &Controls::default());
        let left: Vec<(&str, bool)> = commands
            .0
            .iter()
            .map(|command| (command.key(), matches!(command, ControlCommand::Add { .. })))
            .collect();
        assert_eq!(left, [("b", true), ("b", false), ("a", true), ("a", false)]);
    }

    /// Two worlds standing different ways up share one app, each drawn by a
    /// camera carrying its own frame: one driver pose lands where each
    /// world's own frame puts it, each control's handles are labelled for
    /// its world, and a drag is reported back in the driver's frame from
    /// either. A camera with no frame of its own speaks in the app's.
    #[test]
    fn each_view_speaks_in_its_own_camera_s_frame() {
        let mut app = bare();
        let y_up_camera = app.world_mut().spawn(y_up()).id();
        let z_up_camera = app
            .world_mut()
            .spawn((DriverFrame::default(), ControlCamera))
            .id();
        let plain_camera = app.world_mut().spawn_empty().id();
        let robot = app
            .world_mut()
            .spawn((instance("robot", y_up_camera), Transform::default()))
            .id();
        let face = app
            .world_mut()
            .spawn((instance("face", z_up_camera), Transform::default()))
            .id();
        let plain = app
            .world_mut()
            .spawn((instance("plain", plain_camera), Transform::default()))
            .id();
        let whole = |instance: &str| Control::Transform {
            target: Target::Instance {
                instance: instance.into(),
            },
            modes: None,
            local: false,
            colors: None,
            size: None,
            face_viewer: false,
        };
        let pose = Pose {
            position: [1.0, 2.0, 3.0],
            rotation: no_turn(),
        };
        let set = |key: &str| ControlCommand::SetPose {
            key: key.into(),
            pose,
            scale: None,
        };
        queue(
            &mut app,
            [
                add("robot", whole("robot")),
                add("face", whole("face")),
                add("plain", whole("plain")),
                add("goal", goal_at([1.0, 2.0, 3.0])),
                set("robot"),
                set("face"),
                set("plain"),
            ],
        );
        app.update();
        let at =
            |app: &App, entity: Entity| app.world().get::<Transform>(entity).unwrap().translation;
        assert!(at(&app, robot).abs_diff_eq(Vec3::new(1.0, 3.0, -2.0), 1e-5));
        assert!(at(&app, face).abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-5));
        assert!(
            at(&app, plain).abs_diff_eq(Vec3::new(1.0, 3.0, -2.0), 1e-5),
            "a camera with no frame speaks in the app's"
        );

        let controls = app.world().resource::<Controls>();
        let gizmo = |key: &str| controls.gizmo(key).unwrap();
        let (robot_gizmo, face_gizmo, goal_gizmo) = (gizmo("robot"), gizmo("face"), gizmo("goal"));
        let axes = |entity: Entity| app.world().get::<TransformGizmo>(entity).unwrap().axes;
        assert_eq!(axes(robot_gizmo), TransformGizmo::Z_UP);
        assert_eq!(axes(face_gizmo), [Dir3::X, Dir3::Y, Dir3::Z]);
        let goal = app
            .world()
            .get::<TransformGizmo>(goal_gizmo)
            .unwrap()
            .target;
        assert!(
            at(&app, goal).abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-5),
            "a goal is in its control camera's frame"
        );

        // A drag one unit up each world's own up.
        for (gizmo, target, up) in [(robot_gizmo, robot, Vec3::Y), (face_gizmo, face, Vec3::Z)] {
            let transform = Transform::from_translation(at(&app, target) + up);
            app.world_mut().trigger(GizmoDrag {
                entity: gizmo,
                target,
                handle: GizmoHandle::Arrow(2),
                phase: DragPhase::End,
                gesture: 0,
                transform,
            });
        }
        for event in events(&mut app) {
            let position = Vec3::from(event.pose.unwrap().position);
            assert!(
                position.abs_diff_eq(Vec3::new(1.0, 2.0, 4.0), 1e-5),
                "{}: {position}",
                event.key
            );
        }
    }

    /// An instance that is there but not yet indexed is still to come: an
    /// add on it waits, holding what follows under its key, and lands once
    /// it is indexed; an element it then turns out not to have is refused.
    #[test]
    fn a_control_on_an_instance_not_yet_indexed_waits_for_it() {
        let (mut app, _) = scene();
        let camera = camera(app.world_mut());
        let mut unindexed = instance("face", camera);
        unindexed.bindings.indexed = false;
        let face = app
            .world_mut()
            .spawn((unindexed, Transform::default()))
            .id();
        let on = |target: Target| Control::Transform {
            target,
            modes: None,
            local: false,
            colors: None,
            size: None,
            face_viewer: false,
        };
        let element = |id: &str| Target::Element {
            instance: "face".into(),
            id: id.into(),
        };
        queue(
            &mut app,
            [
                add("eye", on(element("eye"))),
                configure("eye", None, Some(false)),
                add(
                    "whole",
                    on(Target::Instance {
                        instance: "face".into(),
                    }),
                ),
                add("none", on(element("no such element"))),
            ],
        );
        app.update();
        assert!(only::<TransformGizmo>(&mut app).is_none(), "nothing yet");
        assert_eq!(app.world().resource::<ControlCommands>().0.len(), 4);

        let node = app.world_mut().spawn(Transform::default()).id();
        {
            let mut face = app.world_mut().get_mut::<EntityInstance>(face).unwrap();
            face.bindings.elements.insert(
                "eye".into(),
                ElementTargets {
                    node,
                    mesh: None,
                    morph: None,
                },
            );
            face.bindings.indexed = true;
        }
        app.update();
        let controls = app.world().resource::<Controls>();
        let (eye, whole) = (
            controls.gizmo("eye").expect("the element's control"),
            controls.gizmo("whole").expect("the instance's control"),
        );
        assert!(
            controls.gizmo("none").is_none(),
            "an unknown element is refused"
        );
        let eye = app.world().get::<TransformGizmo>(eye).unwrap();
        assert_eq!((eye.target, eye.interactive), (node, false));
        assert_eq!(
            app.world().get::<TransformGizmo>(whole).unwrap().target,
            face
        );
        assert!(app.world().resource::<ControlCommands>().0.is_empty());
    }

    /// A change waiting for the gizmo its add spawns holds what follows
    /// under its key: a later add replaces the control the change was for,
    /// and is left as it was added.
    #[test]
    fn a_waiting_change_holds_a_later_add_behind_it() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        queue(
            &mut app,
            [
                add("k", goal_at([0.0, 0.0, 1.0])),
                configure("k", None, Some(false)),
                add("k", goal_at([0.0, 0.0, 2.0])),
            ],
        );
        app.update();
        app.update();
        let gizmo = only::<TransformGizmo>(&mut app).unwrap();
        let gizmo = app.world().get::<TransformGizmo>(gizmo).unwrap();
        assert!(gizmo.interactive, "the later add is left as added");
        let goal = app.world().get::<Transform>(gizmo.target).unwrap();
        let wanted = Pose {
            position: [0.0, 0.0, 2.0],
            rotation: no_turn(),
        }
        .to_world(&y_up());
        assert!(goal.translation.abs_diff_eq(wanted.translation, 1e-5));
    }

    /// Changes sent with their adds hold from each control's first frame
    /// however many replacements a batch chains under one key: the control
    /// left draws its first frame dimmed and inert as asked, not at full
    /// strength fading down.
    #[test]
    fn a_change_holds_from_the_first_frame_through_chained_replacements() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        queue(
            &mut app,
            [
                add("k", goal_at([0.0, 0.0, 1.0])),
                configure("k", None, Some(true)),
                add("k", goal_at([0.0, 0.0, 2.0])),
                configure("k", Some(0.25), Some(false)),
            ],
        );
        app.update();
        let gizmo = only::<TransformGizmo>(&mut app).expect("the later control");
        let gizmo = app.world().get::<TransformGizmo>(gizmo).unwrap();
        assert_eq!((gizmo.opacity, gizmo.interactive), (0.25, false));
        let wanted = Pose {
            position: [0.0, 0.0, 2.0],
            rotation: no_turn(),
        }
        .to_world(&y_up());
        let goal = app.world().get::<Transform>(gizmo.target).unwrap();
        assert!(goal.translation.abs_diff_eq(wanted.translation, 1e-5));
        assert!(app.world().resource::<ControlCommands>().0.is_empty());
    }

    /// A control whose target and camera went, without its host taking its
    /// gizmo, is forgotten: the gizmo goes, the key is free, and what is sent
    /// under it next acts as for a key with nothing placed — a pose sent
    /// before the next add acts on the control that add makes.
    #[test]
    fn a_control_whose_target_went_is_forgotten_with_its_gizmo() {
        let (mut app, _) = scene();
        let own = app.world_mut().spawn_empty().id();
        let face = app
            .world_mut()
            .spawn((instance("face", own), Transform::default()))
            .id();
        let camera = camera(app.world_mut());
        let other = app
            .world_mut()
            .spawn((instance("other", camera), Transform::default()))
            .id();
        let whole = |instance: &str| Control::Transform {
            target: Target::Instance {
                instance: instance.into(),
            },
            modes: None,
            local: false,
            colors: None,
            size: None,
            face_viewer: false,
        };
        send(&mut app, add("k", whole("face")));
        let old = only::<TransformGizmo>(&mut app).expect("on the face");
        app.world_mut().despawn(face);
        app.world_mut().despawn(own);
        queue(
            &mut app,
            [
                ControlCommand::SetPose {
                    key: "k".into(),
                    pose: Pose {
                        position: [5.0, 0.0, 0.0],
                        rotation: no_turn(),
                    },
                    scale: None,
                },
                add("k", whole("other")),
            ],
        );
        app.update();
        assert!(app.world().get_entity(old).is_err(), "the old gizmo went");
        let gizmo = only::<TransformGizmo>(&mut app).expect("the new control");
        assert_eq!(
            app.world().get::<TransformGizmo>(gizmo).unwrap().target,
            other
        );
        let at = app.world().get::<Transform>(other).unwrap().translation;
        assert!(at.abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-5), "{at}");
        assert!(app.world().resource::<ControlCommands>().0.is_empty());
    }

    /// Changes sent before an add belong to that add's control, and act on it
    /// before anything sent after the add: a pose sent between two adds
    /// moves the first control's target, and the second replaces it after.
    #[test]
    fn changes_sent_before_an_add_act_on_its_control_before_what_follows() {
        let (mut app, _) = scene();
        let camera = camera(app.world_mut());
        let a = app
            .world_mut()
            .spawn((instance("a", camera), Transform::default()))
            .id();
        let b = app
            .world_mut()
            .spawn((instance("b", camera), Transform::default()))
            .id();
        let whole = |instance: &str| Control::Transform {
            target: Target::Instance {
                instance: instance.into(),
            },
            modes: None,
            local: false,
            colors: None,
            size: None,
            face_viewer: false,
        };
        let to = Pose {
            position: [5.0, 0.0, 0.0],
            rotation: no_turn(),
        };
        queue(
            &mut app,
            [
                ControlCommand::Show {
                    key: "k".into(),
                    visible: false,
                },
                add("k", whole("a")),
                ControlCommand::SetPose {
                    key: "k".into(),
                    pose: to,
                    scale: None,
                },
                add("k", whole("b")),
            ],
        );
        app.update();
        let at = |entity: Entity| *app.world().get::<Transform>(entity).unwrap();
        assert!(
            at(a)
                .translation
                .abs_diff_eq(to.to_world(&y_up()).translation, 1e-5),
            "the pose landed on a's control's target: {:?}",
            at(a).translation
        );
        assert_eq!(at(b), Transform::default(), "and not on b's");
        let gizmo = only::<TransformGizmo>(&mut app).expect("b's control");
        assert_eq!(app.world().get::<TransformGizmo>(gizmo).unwrap().target, b);
        assert!(
            app.world().get::<HiddenByDriver>(gizmo).is_none(),
            "the hide was a's control's"
        );
        assert!(app.world().resource::<ControlCommands>().0.is_empty());
    }

    /// A batch chaining more replacements under a key than a frame's passes
    /// carry out stops at the limit rather than running on, and the rest
    /// lands, in order, in the frames after.
    #[test]
    fn a_chain_longer_than_a_frame_s_passes_finishes_in_the_next() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let links = MOST_PASSES + 3;
        queue(
            &mut app,
            (0..links).flat_map(|link| {
                [
                    add("k", goal_at([0.0, 0.0, link as f32])),
                    configure("k", Some(0.25), Some(link.is_multiple_of(2))),
                ]
            }),
        );
        app.update();
        assert!(
            !app.world().resource::<ControlCommands>().0.is_empty(),
            "the rest waits for the next frame"
        );
        app.update();
        app.update();
        assert!(app.world().resource::<ControlCommands>().0.is_empty());
        let gizmo = only::<TransformGizmo>(&mut app).unwrap();
        let gizmo = app.world().get::<TransformGizmo>(gizmo).unwrap();
        assert_eq!(
            (gizmo.opacity, gizmo.interactive),
            (0.25, (links - 1).is_multiple_of(2))
        );
        let goal = app.world().get::<Transform>(gizmo.target).unwrap();
        let last = Pose {
            position: [0.0, 0.0, (links - 1) as f32],
            rotation: no_turn(),
        }
        .to_world(&y_up());
        assert!(goal.translation.abs_diff_eq(last.translation, 1e-5));
    }

    /// A joint control added in a later pass of a frame is resolved by its
    /// host before the changes sent with it apply, in the same frame.
    #[test]
    fn a_joint_added_in_a_later_pass_is_resolved_in_the_frame() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        queue(
            &mut app,
            [
                add("k", goal_at([0.0, 0.0, 1.0])),
                configure("k", None, Some(true)),
                add("k", joint_on(ROBOT)),
                configure("k", Some(0.25), Some(false)),
            ],
        );
        app.update();
        assert!(only::<TransformGizmo>(&mut app).is_none(), "replaced");
        let joint = only::<JointGizmo>(&mut app).expect("resolved in the frame");
        let joint = app.world().get::<JointGizmo>(joint).unwrap();
        assert_eq!(
            (joint.key.as_str(), joint.opacity, joint.interactive),
            ("k", 0.25, false)
        );
    }

    /// A show for a key nothing will add is dropped, rather than waiting,
    /// and asking for a frame, for good.
    #[test]
    fn a_show_for_a_key_nothing_will_add_is_dropped() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        app.update();
        frames_asked(&mut app);
        send(
            &mut app,
            ControlCommand::Show {
                key: "nobody".into(),
                visible: false,
            },
        );
        assert!(app.world().resource::<ControlCommands>().0.is_empty());
        assert_eq!(
            frames_asked(&mut app),
            0,
            "a dropped show asks for no frame"
        );
        app.update();
        app.update();
        assert_eq!(frames_asked(&mut app), 0, "the view can idle");
    }

    /// The JSON a host sends for a fade and an inert control reads as meant.
    #[test]
    fn opacity_and_interactive_read_from_json() {
        let command: ControlCommand = serde_json::from_value(serde_json::json!({
            "op": "configure", "key": "k", "opacity": 0.3, "interactive": false
        }))
        .unwrap();
        assert!(matches!(
            command,
            ControlCommand::Configure {
                opacity: Some(o),
                interactive: Some(false),
                ..
            } if o == 0.3
        ));
    }

    /// A host's press on a control's handle is reported by the control's
    /// key, with where it was.
    #[test]
    fn a_press_with_another_button_is_reported_by_key() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let gizmo = add_goal(&mut app, "goal", None);
        app.world_mut().trigger(GizmoPress {
            entity: gizmo,
            handle: GizmoHandle::Arrow(2),
            button: PointerButton::Secondary,
            position: Vec2::new(40.0, 30.0),
        });
        let presses = std::mem::take(&mut app.world_mut().resource_mut::<ControlPresses>().0);
        assert_eq!(presses.len(), 1);
        assert_eq!(
            (
                presses[0].key.as_str(),
                presses[0].handle,
                presses[0].button,
                presses[0].position
            ),
            (
                "goal",
                GizmoHandle::Arrow(2),
                PressButton::Secondary,
                [40.0, 30.0]
            )
        );
        let json = serde_json::to_value(&presses[0]).unwrap();
        assert_eq!(json["handle"], serde_json::json!({ "Arrow": 2 }));
        assert_eq!(json["button"], "secondary");
    }

    /// The JSON a host sends reads as the commands it means.
    #[test]
    fn commands_read_from_the_host_s_json() {
        let read =
            |json: serde_json::Value| -> ControlCommand { serde_json::from_value(json).unwrap() };
        assert!(matches!(
            read(serde_json::json!({
                "op": "add",
                "key": "k",
                "control": { "kind": "joint", "instance": "arm", "joint": "tilt" },
            })),
            ControlCommand::Add {
                control: Control::Joint { color: None, .. },
                ..
            }
        ));
        assert!(matches!(
            read(serde_json::json!({
                "op": "add",
                "key": "k",
                "control": { "kind": "transform", "target": { "kind": "instance", "instance": "arm" } },
            })),
            ControlCommand::Add {
                control: Control::Transform {
                    target: Target::Instance { .. },
                    ..
                },
                ..
            }
        ));
        assert!(matches!(
            read(
                serde_json::json!({ "op": "highlight", "key": "k", "handle": { "Ring": 0 }, "on": true })
            ),
            ControlCommand::Highlight {
                handle: Some(GizmoHandle::Ring(0)),
                on: true,
                ..
            }
        ));
        assert!(matches!(
            read(serde_json::json!({ "op": "highlight", "key": "k", "on": false })),
            ControlCommand::Highlight {
                handle: None,
                on: false,
                ..
            }
        ));
        assert!(matches!(
            read(serde_json::json!({ "op": "configure", "key": "k", "faceViewer": true })),
            ControlCommand::Configure {
                face_viewer: Some(true),
                ..
            }
        ));
        assert!(matches!(
            read(serde_json::json!({
                "op": "add",
                "key": "k",
                "control": { "kind": "transform", "target": { "kind": "goal", "pose": { "position": [0, 0, 0] } }, "faceViewer": true },
            })),
            ControlCommand::Add {
                control: Control::Transform {
                    face_viewer: true,
                    ..
                },
                ..
            }
        ));
        assert!(matches!(
            read(serde_json::json!({ "op": "configure", "key": "k", "local": true })),
            ControlCommand::Configure {
                local: Some(true),
                color: None,
                colors: None,
                ..
            }
        ));
    }

    #[test]
    fn a_pose_survives_the_round_trip() {
        let pose = Pose {
            position: [0.4, -1.2, 0.9],
            rotation: Quat::from_euler(EulerRot::ZYX, 0.3, -0.2, 1.1).to_array(),
        };
        for frame in [DriverFrame::default(), y_up()] {
            let back = Pose::from_world(&pose.to_world(&frame), &frame);
            assert!(Vec3::from(back.position).abs_diff_eq(Vec3::from(pose.position), 1e-5));
            assert!(
                Quat::from_array(back.rotation).abs_diff_eq(Quat::from_array(pose.rotation), 1e-5)
            );
        }
    }

    /// Every transform in the world, so a test can say none is broken.
    fn all_finite(app: &mut App) -> bool {
        let mut transforms = app.world_mut().query::<&Transform>();
        transforms.iter(app.world()).all(|transform| {
            transform.translation.is_finite()
                && transform.rotation.is_finite()
                && transform.scale.is_finite()
        })
    }

    /// A pose with no turn in it — a zero quaternion — or with a value that is
    /// not a number cannot stand anywhere: a goal added at one is refused, and
    /// a pose set to one moves nothing, rather than putting a transform the
    /// renderer cannot draw into the scene.
    #[test]
    fn a_pose_that_is_not_one_is_refused() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let mut control = goal_at([0.0, 0.0, 1.0]);
        if let Control::Transform {
            target: Target::Goal { pose, .. },
            ..
        } = &mut control
        {
            pose.rotation = [0.0; 4];
        }
        send(&mut app, add("bad", control));
        assert!(only::<TransformGizmo>(&mut app).is_none(), "refused");
        assert!(all_finite(&mut app));

        let gizmo = add_goal(&mut app, "goal", None);
        let goal = app.world().get::<TransformGizmo>(gizmo).unwrap().target;
        let before = *app.world().get::<Transform>(goal).unwrap();
        for pose in [
            Pose {
                position: [f32::NAN, 0.0, 0.0],
                rotation: no_turn(),
            },
            Pose {
                position: [0.0, 0.0, 0.0],
                rotation: [0.0; 4],
            },
            Pose {
                position: [0.0, 0.0, 0.0],
                rotation: [f32::INFINITY, 0.0, 0.0, 1.0],
            },
            // Finite, but its length overflows: normalized, it is zero.
            Pose {
                position: [0.0, 0.0, 0.0],
                rotation: [1e20, 0.0, 0.0, 0.0],
            },
        ] {
            assert!(!pose.is_valid(), "{pose:?}");
            send(
                &mut app,
                ControlCommand::SetPose {
                    key: "goal".into(),
                    pose,
                    scale: None,
                },
            );
        }
        assert_eq!(*app.world().get::<Transform>(goal).unwrap(), before);
        assert!(all_finite(&mut app));
        assert!(app.world().resource::<ControlCommands>().0.is_empty());
    }

    fn show(key: &str, visible: bool) -> ControlCommand {
        ControlCommand::Show {
            key: key.into(),
            visible,
        }
    }

    /// The kinds of the commands left, by key, to compare a queue against.
    fn left(commands: &ControlCommands) -> Vec<(&str, &'static str)> {
        commands
            .0
            .iter()
            .map(|command| {
                let kind = match command {
                    ControlCommand::Add { control, .. } => match control.instance() {
                        Some("robot-1") => "add on robot-1",
                        Some(_) => "add on another",
                        None => "add",
                    },
                    ControlCommand::Remove { .. } => "remove",
                    ControlCommand::Show { .. } => "show",
                    ControlCommand::SetPose { .. } => "pose",
                    ControlCommand::Configure { .. } => "configure",
                    ControlCommand::Highlight { .. } => "highlight",
                };
                (command.key(), kind)
            })
            .collect()
    }

    /// A change sent before an add waits for that add and is its, so
    /// forgetting the add's instance takes the change too, rather than leaving
    /// it to land on the next add's control.
    #[test]
    fn forgetting_an_instance_takes_the_changes_waiting_for_its_add() {
        let mut commands = ControlCommands(vec![
            show("k", false),
            add("k", joint_on(ROBOT)),
            add("k", joint_on("robot-1")),
            // A change after an add is that add's, so stays with robot-1's.
            add("j", joint_on("robot-1")),
            show("j", false),
            add("j", joint_on(ROBOT)),
        ]);
        commands.forget_instance(ROBOT, &Controls::default());
        assert_eq!(
            left(&commands),
            [
                ("k", "add on robot-1"),
                ("j", "add on robot-1"),
                ("j", "show")
            ]
        );
    }

    /// Forgetting a view takes the goal adds that name it, with their
    /// changes, and leaves a goal on another view and adds on instances.
    #[test]
    fn forgetting_a_view_takes_the_goals_waiting_for_it() {
        let goal_on = |view: &str| {
            let mut control = goal_at([0.0, 0.0, 1.0]);
            if let Control::Transform {
                target: Target::Goal { view: named, .. },
                ..
            } = &mut control
            {
                *named = Some(view.into());
            }
            control
        };
        let mut commands = ControlCommands(vec![
            add("g", goal_on("face-a")),
            show("g", false),
            add("h", goal_on("face-b")),
            add("k", joint_on("robot-1")),
        ]);
        commands.forget_view("face-a", &Controls::default());
        assert_eq!(left(&commands), [("h", "add"), ("k", "add on robot-1")]);
    }

    /// A remove is never held, so one behind a forgotten add stays, and ends
    /// what is dropped with it: what follows the remove is not the add's.
    #[test]
    fn forgetting_an_instance_keeps_a_remove_behind_its_add() {
        let mut commands = ControlCommands(vec![
            add("k", joint_on(ROBOT)),
            show("k", false),
            ControlCommand::Remove { key: "k".into() },
            show("k", true),
            add("k", joint_on("robot-1")),
        ]);
        commands.forget_instance(ROBOT, &Controls::default());
        assert_eq!(
            left(&commands),
            [("k", "remove"), ("k", "show"), ("k", "add on robot-1")]
        );
    }

    /// A change sent under a key with a control placed acts on that control,
    /// whatever add follows it, so forgetting the add's instance leaves it.
    #[test]
    fn forgetting_an_instance_keeps_the_changes_for_a_placed_control() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        add_goal(&mut app, "k", None);
        let mut commands = ControlCommands(vec![
            show("k", false),
            add("k", joint_on("loading")),
            show("free", false),
            add("free", joint_on("loading")),
        ]);
        commands.forget_instance("loading", app.world().resource::<Controls>());
        assert_eq!(
            left(&commands),
            [("k", "show")],
            "the placed control's show stays; a show for no control was the add's"
        );
    }

    /// A goal at something that is not a pose is refused at once, whichever
    /// view it names: it does not wait for a view, holding the adds after it
    /// under its key.
    #[test]
    fn a_goal_at_no_pose_is_refused_before_its_view_is_looked_for() {
        let mut app = bare();
        app.world_mut().spawn(ControlCamera);
        let mut bad = goal_at([0.0, 0.0, 1.0]);
        if let Control::Transform {
            target: Target::Goal { pose, view },
            ..
        } = &mut bad
        {
            pose.rotation = [0.0; 4];
            *view = Some("nowhere".into());
        }
        queue(
            &mut app,
            [add("k", bad), add("k", goal_at([0.0, 0.0, 2.0]))],
        );
        app.update();
        assert!(app.world().resource::<ControlCommands>().0.is_empty());
        assert!(only::<TransformGizmo>(&mut app).is_some(), "the later add");
    }

    /// A control hidden, by its driver or by a ghosting, is not interactive
    /// by the end of the frame it is hidden in, so it takes no press after
    /// it and a drag in hand ends. Shown again, it takes what its driver last
    /// asked for.
    #[test]
    fn a_hidden_control_takes_no_input_and_keeps_what_its_driver_asked() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let gizmo = add_goal(&mut app, "goal", None);
        let interactive = |app: &App| {
            app.world()
                .get::<TransformGizmo>(gizmo)
                .unwrap()
                .interactive
        };
        assert!(interactive(&app));
        send(&mut app, show("goal", false));
        assert!(!interactive(&app), "hidden, it takes nothing");
        send(&mut app, show("goal", true));
        assert!(interactive(&app), "shown, as it was");

        send(&mut app, show("goal", false));
        send(&mut app, configure("goal", None, Some(false)));
        send(&mut app, show("goal", true));
        assert!(!interactive(&app), "inert as its driver last asked");
        send(&mut app, configure("goal", None, Some(true)));
        assert!(interactive(&app));

        let target = app.world().get::<TransformGizmo>(gizmo).unwrap().target;
        app.world_mut().entity_mut(target).insert(Ghost);
        app.update();
        assert!(!interactive(&app), "a ghost's control takes nothing");
        app.world_mut().entity_mut(target).remove::<Ghost>();
        app.update();
        assert!(interactive(&app));
    }

    /// A change waiting on an add is that add's: when the add fails — a
    /// joint the host cannot resolve, an element the instance lacks — the
    /// change goes with it, rather than landing on the next add's control.
    #[test]
    fn changes_waiting_on_a_failed_add_go_with_it() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let missing = || Control::Joint {
            instance: ROBOT.into(),
            joint: "nothing drives this".into(),
            color: None,
            size: None,
        };
        let no_element = Control::Transform {
            target: Target::Element {
                instance: ROBOT.into(),
                id: "no such element".into(),
            },
            modes: None,
            local: false,
            colors: None,
            size: None,
            face_viewer: false,
        };
        queue(
            &mut app,
            [
                add("joint", missing()),
                configure("joint", Some(0.25), Some(false)),
                add("joint", goal_at([0.0, 0.0, 1.0])),
                show("element", false),
                add("element", no_element),
                configure("element", Some(0.25), Some(false)),
                add("element", goal_at([0.0, 0.0, 2.0])),
            ],
        );
        app.update();
        let controls = app.world().resource::<Controls>();
        for key in ["joint", "element"] {
            let gizmo = controls.gizmo(key).expect("the goal");
            let gizmo = app.world().get::<TransformGizmo>(gizmo).unwrap();
            assert_eq!(
                (gizmo.opacity, gizmo.interactive),
                (1.0, true),
                "{key}: the failed add's changes went with it"
            );
            assert!(
                app.world()
                    .get::<HiddenByDriver>(controls.gizmo(key).unwrap())
                    .is_none(),
                "{key}"
            );
        }
        assert!(app.world().resource::<ControlCommands>().0.is_empty());
    }

    /// A joint add's changes go with it when it fails, however late in a
    /// frame's passes it fails: past the passes a frame runs, what it leaves
    /// for the next frame is not given to the control left under its key.
    #[test]
    fn a_failed_joint_s_changes_go_with_it_past_a_frame_s_passes() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let mut batch = Vec::new();
        for link in 0..7 {
            batch.push(add("k", goal_at([0.0, 0.0, link as f32])));
            batch.push(configure("k", None, Some(true)));
        }
        batch.push(add(
            "k",
            Control::Joint {
                instance: ROBOT.into(),
                joint: "nothing drives this".into(),
                color: None,
                size: None,
            },
        ));
        batch.push(configure("k", Some(0.25), Some(false)));
        queue(&mut app, batch);
        for _ in 0..4 {
            app.update();
        }
        assert!(app.world().resource::<ControlCommands>().0.is_empty());
        // The control the joint would have replaced is the key's still, as
        // it was: the joint's configure went with the joint.
        let gizmo = only::<TransformGizmo>(&mut app).expect("the last goal");
        let gizmo = app.world().get::<TransformGizmo>(gizmo).unwrap();
        assert_eq!(
            (gizmo.opacity, gizmo.interactive),
            (1.0, true),
            "the failed joint's configure went with it"
        );
        let goal = app.world().get::<Transform>(gizmo.target).unwrap();
        let last = Pose {
            position: [0.0, 0.0, 6.0],
            rotation: no_turn(),
        }
        .to_world(&y_up());
        assert!(goal.translation.abs_diff_eq(last.translation, 1e-5));
    }

    /// A joint add replaces the control under its key only once its host
    /// resolves it: one the host cannot resolve leaves that control as it
    /// was, as any refused add does.
    #[test]
    fn a_joint_its_host_cannot_resolve_leaves_the_control_it_would_replace() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let old = add_goal(&mut app, "k", None);
        let goal = app.world().get::<TransformGizmo>(old).unwrap().target;
        send(
            &mut app,
            add(
                "k",
                Control::Joint {
                    instance: ROBOT.into(),
                    joint: "nothing drives this".into(),
                    color: None,
                    size: None,
                },
            ),
        );
        app.update();
        assert_eq!(app.world().resource::<Controls>().gizmo("k"), Some(old));
        assert!(app.world().get_entity(old).is_ok(), "its gizmo stays");
        assert!(app.world().get_entity(goal).is_ok(), "and its goal");
        send(&mut app, configure("k", None, Some(false)));
        assert!(
            !app.world().get::<TransformGizmo>(old).unwrap().interactive,
            "and it is the key's control still"
        );

        // One its host resolves replaces it.
        send(&mut app, add("k", joint_on(ROBOT)));
        assert!(app.world().get_entity(old).is_err(), "replaced");
        assert!(app.world().get_entity(goal).is_err(), "goal and all");
        let joint = only::<JointGizmo>(&mut app).expect("the joint control");
        assert_eq!(app.world().resource::<Controls>().gizmo("k"), Some(joint));
    }

    /// A goal is the control's own, so it goes when the control does, even
    /// when a host takes the control's gizmo away itself.
    #[test]
    fn a_goal_goes_when_a_host_takes_its_gizmo() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let gizmo = add_goal(&mut app, "goal", None);
        let goal = app.world().get::<TransformGizmo>(gizmo).unwrap().target;
        app.world_mut().despawn(gizmo);
        app.update();
        assert!(app.world().get_entity(goal).is_err(), "the goal went too");
        assert_eq!(app.world().resource::<Controls>().keys().count(), 0);
    }

    /// An add waiting for what no frame of the controls' own can bring — an
    /// instance loading, a camera not yet marked — asks for no frames: a
    /// view that draws on demand can idle until the host's change comes.
    #[test]
    fn an_add_waiting_asks_for_no_frames() {
        let mut app = bare();
        app.update();
        frames_asked(&mut app);
        send(&mut app, add("goal", goal_at([0.0, 0.0, 1.0])));
        app.update();
        assert_eq!(frames_asked(&mut app), 0, "waiting for a camera");
        assert_eq!(app.world().resource::<ControlCommands>().0.len(), 1);

        let (mut app, _) = scene();
        app.update();
        frames_asked(&mut app);
        send(&mut app, add(TILT, joint_on(ROBOT)));
        app.update();
        assert_eq!(frames_asked(&mut app), 0, "waiting for its instance");
        assert_eq!(app.world().resource::<ControlCommands>().0.len(), 1);
    }

    /// A joint control goes when the instance it is on goes, though its body
    /// and camera are left.
    #[test]
    fn a_joint_control_goes_with_its_instance() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        send(&mut app, add(TILT, joint_on(ROBOT)));
        let gizmo = only::<JointGizmo>(&mut app).expect("a joint control");
        let root = app
            .world_mut()
            .query_filtered::<Entity, With<Rig>>()
            .single(app.world())
            .unwrap();
        app.world_mut().despawn(root);
        app.update();
        assert!(app.world().get_entity(gizmo).is_err(), "its gizmo went");
        assert_eq!(app.world().resource::<Controls>().keys().count(), 0);
    }

    /// A pose for a placed control whose target has no transform to set is
    /// refused, rather than waiting, asking for frames and holding the key,
    /// for good.
    #[test]
    fn a_pose_for_a_target_with_no_transform_is_refused() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        // The fixture's robot root carries no transform of its own.
        send(&mut app, add("whole", on_robot()));
        app.update();
        frames_asked(&mut app);
        queue(
            &mut app,
            [
                ControlCommand::SetPose {
                    key: "whole".into(),
                    pose: Pose {
                        position: [1.0, 0.0, 0.0],
                        rotation: no_turn(),
                    },
                    scale: None,
                },
                add("whole", goal_at([0.0, 0.0, 1.0])),
            ],
        );
        app.update();
        assert!(app.world().resource::<ControlCommands>().0.is_empty());
        let gizmo = only::<TransformGizmo>(&mut app).expect("the goal, not held");
        let target = app.world().get::<TransformGizmo>(gizmo).unwrap().target;
        assert!(app.world().get::<Transform>(target).is_some(), "a goal");
    }

    /// Only a control's own gizmo reports as that control: a gizmo a host
    /// spawned for itself, under a key of its own for the overlay or even
    /// under a control's key, reports nothing.
    #[test]
    fn only_a_control_s_own_gizmo_reports() {
        let (mut app, body) = scene();
        robot(&mut app, body);
        let own = add_goal(&mut app, "k", None);
        let camera = camera(app.world_mut());
        let mut hosts = Vec::new();
        for key in ["host", "k"] {
            hosts.push(
                app.world_mut()
                    .spawn(TransformGizmo {
                        key: key.into(),
                        ..TransformGizmo::new(body, camera)
                    })
                    .id(),
            );
        }
        for gizmo in hosts.iter().copied().chain([own]) {
            app.world_mut().trigger(GizmoDrag {
                entity: gizmo,
                target: body,
                handle: GizmoHandle::Arrow(0),
                phase: DragPhase::Move,
                gesture: 0,
                transform: Transform::default(),
            });
            app.world_mut().trigger(GizmoPress {
                entity: gizmo,
                handle: GizmoHandle::Arrow(0),
                button: PointerButton::Secondary,
                position: Vec2::ZERO,
            });
        }
        let joint = app
            .world_mut()
            .spawn(JointGizmo {
                key: "host-joint".into(),
                ..JointGizmo::new(
                    body,
                    camera,
                    JointMotion::Turn,
                    Dir3::X,
                    -1.0,
                    1.0,
                    0.0,
                    Color::WHITE,
                )
            })
            .id();
        app.world_mut().trigger(JointDrag {
            entity: joint,
            target: body,
            phase: DragPhase::Move,
            gesture: 0,
            value: 0.5,
        });
        let events = events(&mut app);
        let presses = std::mem::take(&mut app.world_mut().resource_mut::<ControlPresses>().0);
        assert_eq!(events.len(), 1, "{events:?}");
        assert_eq!(presses.len(), 1);
        assert_eq!(
            (events[0].key.as_str(), presses[0].key.as_str()),
            ("k", "k")
        );
    }

    /// A goal is used through the view it names, which waits until a camera
    /// carries that name. Naming none, it takes the one camera marked for
    /// goals; with several marked it is refused rather than given a guess.
    #[test]
    fn a_goal_is_used_through_the_view_it_names() {
        let named = |view: Option<&str>| Control::Transform {
            target: Target::Goal {
                pose: Pose {
                    position: [0.0, 0.0, 1.0],
                    rotation: no_turn(),
                },
                view: view.map(String::from),
            },
            modes: None,
            local: false,
            colors: None,
            size: None,
            face_viewer: false,
        };
        let mut app = bare();
        app.world_mut().spawn(ControlCamera);
        app.world_mut().spawn(ControlCamera);
        send(&mut app, add("unnamed", named(None)));
        assert!(only::<TransformGizmo>(&mut app).is_none(), "refused");
        assert!(app.world().resource::<ControlCommands>().0.is_empty());

        send(&mut app, add("face", named(Some("face"))));
        assert!(
            only::<TransformGizmo>(&mut app).is_none(),
            "no such view yet"
        );
        assert_eq!(app.world().resource::<ControlCommands>().0.len(), 1);
        let face = app
            .world_mut()
            .spawn((ControlView("face".into()), DriverFrame::default()))
            .id();
        app.update();
        let gizmo = only::<TransformGizmo>(&mut app).expect("on the face's view");
        let gizmo = app.world().get::<TransformGizmo>(gizmo).unwrap();
        assert_eq!(gizmo.camera, face);
        assert_eq!(gizmo.axes, [Dir3::X, Dir3::Y, Dir3::Z], "in its frame");

        let read = |json: serde_json::Value| -> Target { serde_json::from_value(json).unwrap() };
        let pose = serde_json::json!({ "position": [0.0, 0.0, 1.0] });
        assert!(matches!(
            read(serde_json::json!({ "kind": "goal", "pose": pose })),
            Target::Goal { view: None, .. }
        ));
        assert!(matches!(
            read(serde_json::json!({ "kind": "goal", "pose": pose, "view": "face" })),
            Target::Goal { view: Some(view), .. } if view == "face"
        ));
    }

    /// A drag on a control whose target's parent moved this frame reports
    /// where the target would stand under the parent as it now stands, not
    /// as it was last propagated.
    #[test]
    fn a_drag_is_reported_against_its_parent_as_it_now_stands() {
        let (mut app, _) = scene();
        let camera = camera(app.world_mut());
        let parent = app
            .world_mut()
            .spawn((
                Transform::from_xyz(10.0, 0.0, 0.0),
                GlobalTransform::IDENTITY,
            ))
            .id();
        let face = app
            .world_mut()
            .spawn((
                instance("face", camera),
                Transform::default(),
                ChildOf(parent),
            ))
            .id();
        send(
            &mut app,
            add(
                "whole",
                Control::Transform {
                    target: Target::Instance {
                        instance: "face".into(),
                    },
                    modes: None,
                    local: false,
                    colors: None,
                    size: None,
                    face_viewer: false,
                },
            ),
        );
        let gizmo = only::<TransformGizmo>(&mut app).unwrap();
        app.world_mut().trigger(GizmoDrag {
            entity: gizmo,
            target: face,
            handle: GizmoHandle::Arrow(0),
            phase: DragPhase::End,
            gesture: 0,
            transform: Transform::default(),
        });
        let pose = events(&mut app)[0].pose.unwrap();
        let at = Vec3::from(pose.position);
        assert!(at.abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), 1e-5), "{at}");
    }
}
