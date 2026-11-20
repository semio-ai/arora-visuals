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
//! Every host takes the same commands, through [`ControlCommands`], under the
//! same rules, and reports the same events. What differs between hosts is
//! told to this module rather than decided by it, in four places.
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
//! # What is still to come: [`Loading`]
//!
//! A control on an instance that is still loading waits for it, and one on an
//! instance nothing is loading is refused. Only the host knows which is
//! which, so it says: it puts [`Loading`] on the entity it is loading an
//! instance into until that entity carries its [`EntityInstance`].
//!
//! An instance that is there but not yet indexed
//! ([`SceneBindings::indexed`](crate::binding::SceneBindings::indexed)) is
//! still to come as well: until it is, an element it lacks cannot be told
//! from one not yet joined, so a control on it waits rather than being
//! refused. A host that spawns an instance and indexes it later needs no
//! [`Loading`] for that gap.
//!
//! # Which camera a control is used through: [`ControlCamera`]
//!
//! A control on an instance is used through that instance's own
//! [`camera`](EntityInstance::camera), so a face drawn by a camera of its own
//! has its controls sized and picked through that camera. A goal is on no
//! instance; it is used through the camera the host marks [`ControlCamera`],
//! and a goal added while no camera is marked waits for one. The camera also
//! decides which [`DriverFrame`] the control's poses are in.
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
//! 2. In [`ControlSystems::Joints`], the host's resolver puts a
//!    [`JointGizmo`] on each request entity it can resolve, built from its rig
//!    and the request's [`camera`](JointRequest::camera).
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
//!   makes, not the one that replaces it. A change sent before its add waits
//!   for that add, holding later changes but not the add itself.
//! - **A remove is not held**: it cancels a waiting add and what waits behind
//!   it, and takes away the control the add would have replaced.
//! - **A key names one control in the view**; an add under a key in use
//!   replaces its control.
//! - A change for a key — a show, a pose, a configure, a highlight — waits for
//!   its control's gizmo, which an add spawns at the end of its pass, or for
//!   an add still to come. A change for a key nothing will add is dropped, so
//!   it does not ask for frames for good.
//! - Commands are carried out in two passes a frame, the gizmos an add spawns
//!   being there for the second, so a change sent with its control's add
//!   takes hold the frame the control first shows.
//! - A control whose gizmo went with its instance is forgotten, and its key is
//!   free again.

use std::collections::{HashMap, HashSet};

use bevy::math::Affine3A;
use bevy::picking::pointer::PointerButton;
use bevy::prelude::*;
use bevy::window::RequestRedraw;
use serde::{Deserialize, Serialize};

use crate::ghost::Ghost;
use crate::instance::{EntityInstance, Loading, Superseded};
use crate::joint_gizmo::{JointDrag, JointGizmo};
use crate::transform_gizmo::{
    DragPhase, GizmoDrag, GizmoHandle, GizmoPress, GizmoSpace, TransformGizmo,
    TransformGizmoSystems, TransformModes,
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
    Goal { pose: Pose },
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
    /// `color`, a transform control's `colors` and whether it is `local`, and
    /// for either its `opacity` and whether it is `interactive`. What is
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
    /// Drops the adds waiting for `instance`, and the commands waiting behind
    /// each under its key, for an instance a host has taken away before it
    /// arrived: they would otherwise wait for a control that never comes, up
    /// to a later add under that key, which is another control's and stays.
    pub fn forget_instance(&mut self, instance: &str) {
        let mut dropping = HashSet::new();
        self.0.retain(|command| {
            if let ControlCommand::Add { key, control } = command {
                if control.instance() == Some(instance) {
                    dropping.insert(key.clone());
                } else {
                    dropping.remove(key);
                }
            }
            !dropping.contains(command.key())
        });
    }
}

/// On a control the driver has hidden. A view also hides controls on its own
/// account (on a ghosted instance, say), and shows a control again only when
/// neither wants it hidden.
#[derive(Component)]
pub struct HiddenByDriver;

/// On the camera a goal's control is used through. A host that offers goals
/// marks one; a control on an instance is used through that instance's own.
#[derive(Component, Default)]
pub struct ControlCamera;

/// A joint control waiting for its host to resolve it
/// ([module docs](self#joint-controls-jointrequest)).
///
/// The host's resolver, in [`ControlSystems::Joints`], puts a [`JointGizmo`]
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

/// Where a drag is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Phase {
    Start,
    Move,
    End,
}

impl From<DragPhase> for Phase {
    fn from(phase: DragPhase) -> Self {
        match phase {
            DragPhase::Start => Self::Start,
            DragPhase::Move => Self::Move,
            DragPhase::End => Self::End,
        }
    }
}

/// A drag on a control: what it asks its target to become. Nothing has moved;
/// the driver writes the change if it wants it.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ControlEvent {
    pub key: String,
    pub phase: Phase,
    /// A joint control's value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<f32>,
    /// A transform control's target's pose, in the world.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pose: Option<Pose>,
    /// A transform control's target's scale, along its own axes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<[f32; 3]>,
}

/// Drags since a host last drained them.
#[derive(Resource, Default)]
pub struct ControlEvents(pub Vec<ControlEvent>);

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
#[derive(Clone, Debug, Serialize)]
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
    /// The opacity the host asked for, which the gizmo's fades toward.
    opacity: f32,
    /// The fade toward `opacity` under way, if one is.
    fade: Option<Fade>,
    /// Whether its gizmo shows: drawn, and hidden neither by the host nor by
    /// its instance's being ghosted. While it does not, an opacity asked for
    /// is taken at once rather than faded to, so a control added dimmed, or
    /// dimmed while hidden, never shows at full strength on its way there.
    shown: bool,
}

impl Placed {
    fn new(gizmo: Entity, target: Option<Entity>, goal: Option<Entity>, camera: Entity) -> Self {
        Self {
            gizmo,
            target,
            goal,
            camera,
            opacity: 1.0,
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
    /// Everything that carries out commands: both passes, and
    /// [`Joints`](Self::Joints) between them.
    Commands,
    /// Where a host resolves joint requests: after the first pass has spawned
    /// them, before they are finished for the second.
    Joints,
    /// The fade toward the opacity each control was asked for.
    Fade,
}

/// The two passes, either side of the joints.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
enum Pass {
    First,
    Second,
}

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
        app.init_resource::<ControlCommands>()
            .init_resource::<ControlEvents>()
            .init_resource::<ControlPresses>()
            .init_resource::<Controls>()
            .init_resource::<DefaultDriverFrame>()
            .add_message::<RequestRedraw>()
            // Twice, the gizmos an add spawns being there for the second: a
            // change sent with its control's add then takes hold the frame
            // the control first shows, rather than one frame later, live and
            // at full strength in between. Both before the gizmos' own drag
            // handling, so a left press in the frame a control is made inert
            // does not take it. A press with another button is reported in
            // `PreUpdate`, before either pass, so one in that frame can still
            // reach the host: a press and a change arriving between the same
            // two frames have no order to keep.
            .configure_sets(
                Update,
                (Pass::First, ControlSystems::Joints, Pass::Second)
                    .chain()
                    .in_set(ControlSystems::Commands),
            )
            .configure_sets(
                Update,
                (ControlSystems::Commands, ControlSystems::Fade)
                    .chain()
                    .before(TransformGizmoSystems),
            )
            .add_systems(
                Update,
                (
                    carry_out.in_set(Pass::First),
                    (finish_joints, carry_out).chain().in_set(Pass::Second),
                    // After the commands, so a control added this frame is
                    // hidden in it, and before the fade reads whether a
                    // control shows.
                    hide_ghosted
                        .after(ControlSystems::Commands)
                        .before(ControlSystems::Fade),
                    fade.in_set(ControlSystems::Fade),
                ),
            )
            .add_observer(report_transform_drag)
            .add_observer(report_joint_drag)
            .add_observer(report_press);
    }
}

/// Where `entity` stands in the world as the transforms say now. Its
/// `GlobalTransform` lags a transform written earlier in the frame — an
/// instance a host moved this frame — until propagation in `PostUpdate`, so
/// the chain is composed here, falling back to the propagated transform of an
/// ancestor with none of its own.
fn current_global(
    mut entity: Entity,
    transforms: &Query<&mut Transform>,
    parents: &Query<&ChildOf>,
    globals: &Query<&GlobalTransform>,
) -> Option<Affine3A> {
    let mut affine = Affine3A::IDENTITY;
    loop {
        let Ok(transform) = transforms.get(entity) else {
            return globals.get(entity).ok().map(|g| g.affine() * affine);
        };
        affine = transform.compute_affine() * affine;
        match parents.get(entity) {
            Ok(parent) => entity = parent.parent(),
            Err(_) => return Some(affine),
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
fn finish_joints(
    mut commands: Commands,
    mut requests: Query<(Entity, &JointRequest, Option<&mut JointGizmo>)>,
    instances: Query<&EntityInstance>,
) {
    for (entity, request, gizmo) in &mut requests {
        match gizmo {
            Some(mut gizmo) => {
                gizmo.key = request.key.clone();
                gizmo.apply = false;
                if let Some(colour) = request.colour {
                    gizmo.colour = colour;
                }
                commands.entity(entity).remove::<JointRequest>();
            }
            None => {
                let instance = instances
                    .get(request.instance)
                    .map_or("an instance gone", |instance| instance.id.as_str());
                log::warn!(
                    "no joint of {instance} is driven by {:?}; no control added for {:?}",
                    request.joint,
                    request.key
                );
                commands.entity(entity).despawn();
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
    cameras: Query<Entity, With<ControlCamera>>,
    instances: Query<(Entity, &EntityInstance, Has<Superseded>)>,
    loading: Query<&Loading>,
    parents: Query<&ChildOf>,
    globals: Query<&GlobalTransform>,
    mut transforms: Query<&mut Transform>,
    placeable: Query<(), Placeable>,
    mut transform_gizmos: Query<&mut TransformGizmo>,
    mut joint_gizmos: Query<&mut JointGizmo>,
    mut redraw: MessageWriter<RequestRedraw>,
) {
    // A control whose gizmo is gone went with the instance it was on, or was
    // a joint its host could not resolve; its key is free again.
    controls
        .0
        .retain(|_, placed| placeable.contains(placed.gizmo));
    if pending.0.is_empty() {
        return;
    }
    // The instance under a host's id, if one is there, indexed or not. One
    // being replaced gives way to its successor, which is the one a control
    // should stay on.
    let instance = |name: &str| {
        instances
            .iter()
            .filter(|(_, instance, _)| instance.id == name)
            .min_by_key(|(_, _, superseded)| *superseded)
            .map(|(entity, instance, _)| (entity, instance))
    };
    // Whether an instance not ready to be addressed is on its way: loading,
    // or there but not yet indexed, so its elements cannot be told from ones
    // it lacks. What a control on it waits for.
    let coming = |name: &str| loading.iter().any(|loading| loading.instance == name);
    let asked = std::mem::take(&mut pending.0);
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
    let any_loading = !loading.is_empty()
        || instances
            .iter()
            .any(|(_, instance, _)| !instance.bindings.indexed);
    let worth_waiting = |key: &str, controls: &Controls| {
        controls.0.contains_key(key) || adding.contains(key) || any_loading
    };
    let mut waiting = Vec::new();
    // Keys with a command waiting on what is already asked for: an add for
    // its instance or camera, or a change for the gizmo its control's add is
    // spawning. Every later command under the same key waits behind it, so
    // they land in the order they were asked: a later add is the control
    // left, and a change acts on the control the add before it makes, not
    // one that replaces it.
    let mut held: HashSet<String> = HashSet::new();
    // Keys with a change waiting for an add not yet made, sent before it.
    // Later changes wait behind it, keeping their order, but an add does
    // not: the add is what the change waits for, and holding it would hold
    // both for good.
    let mut held_changes: HashSet<String> = HashSet::new();
    // Whether to draw another frame: for a command that changed something, or
    // one waiting on a gizmo spawned at the end of this pass.
    let mut again = false;
    for command in asked {
        // A remove is not held: it cancels the waiting add and what waits
        // behind it, and takes away the control the add would have replaced.
        let key = command.key();
        let behind = match command {
            ControlCommand::Remove { .. } => false,
            ControlCommand::Add { .. } => held.contains(key),
            _ => held.contains(key) || held_changes.contains(key),
        };
        if behind {
            waiting.push(command);
            continue;
        }
        again |= matches!(
            command,
            ControlCommand::Add { .. } | ControlCommand::Remove { .. }
        );
        match command {
            ControlCommand::Add { key, control } => {
                // A control on an instance waits for it while it is loading
                // or not yet indexed; one on an instance nothing is loading
                // is dropped.
                let on = match control.instance() {
                    None => None,
                    Some(name) => match instance(name) {
                        Some((_, found)) if !found.bindings.indexed => {
                            defer(
                                &mut waiting,
                                &mut held,
                                ControlCommand::Add { key, control },
                            );
                            continue;
                        }
                        Some(found) => Some(found),
                        None if coming(name) => {
                            defer(
                                &mut waiting,
                                &mut held,
                                ControlCommand::Add { key, control },
                            );
                            continue;
                        }
                        None => {
                            log::warn!("no instance {name:?}; no control added for {key:?}");
                            continue;
                        }
                    },
                };
                // An instance's own camera, or for a goal the host's.
                let camera = match on {
                    Some((_, instance)) => instance.camera,
                    None => match cameras.iter().next() {
                        Some(camera) => camera,
                        None => {
                            log::debug!("no control camera yet; {key:?} waits for one");
                            defer(
                                &mut waiting,
                                &mut held,
                                ControlCommand::Add { key, control },
                            );
                            continue;
                        }
                    },
                };
                let frame = DriverFrame::of(camera, &frames, &app_frame);
                if let Some(old) = controls.0.remove(&key) {
                    despawn(&mut commands, old);
                }
                let placed = match control {
                    Control::Joint { joint, color, .. } => {
                        let Some((root, _)) = on else {
                            continue;
                        };
                        let request = JointRequest {
                            instance: root,
                            joint,
                            camera,
                            key: key.clone(),
                            colour: color.as_deref().and_then(colour),
                        };
                        Placed::new(commands.spawn(request).id(), None, None, camera)
                    }
                    Control::Transform {
                        target,
                        modes,
                        local,
                        colors,
                    } => {
                        let (target, goal) = match (target, on) {
                            (Target::Instance { .. }, Some((root, _))) => (root, None),
                            (Target::Element { id, .. }, Some((_, instance))) => {
                                match instance.bindings.elements.get(&id) {
                                    Some(element) => (element.node, None),
                                    None => {
                                        log::warn!(
                                            "no element {id:?} on {}; no control added for {key:?}",
                                            instance.id
                                        );
                                        continue;
                                    }
                                }
                            }
                            (Target::Instance { .. } | Target::Element { .. }, None) => continue,
                            (Target::Goal { pose }, _) => {
                                let goal = commands
                                    .spawn((pose.to_world(&frame), Visibility::default()))
                                    .id();
                                (goal, Some(goal))
                            }
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
                                ..base
                            })
                            .id();
                        Placed::new(gizmo, Some(target), goal, camera)
                    }
                };
                controls.0.insert(key, placed);
            }
            ControlCommand::Remove { key } => {
                if let Some(old) = controls.0.remove(&key) {
                    despawn(&mut commands, old);
                }
                // Whatever the driver asked of this key that is still
                // waiting goes too, or an add made before its instance
                // arrived would bring back a control the driver has since
                // removed.
                waiting.retain(|waiting| waiting.key() != key);
                held.remove(&key);
                held_changes.remove(&key);
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
                        &mut held_changes,
                        ControlCommand::Show { key, visible },
                    );
                } else {
                    log::debug!("no control {key:?} to show or hide");
                }
            }
            ControlCommand::SetPose { key, pose, scale } => {
                let placed = controls.0.get(&key);
                let target = placed.and_then(|p| p.target);
                let frame = placed.map(|p| DriverFrame::of(p.camera, &frames, &app_frame));
                // A goal added in this pass is spawned at the end of it, so it
                // has no transform yet; the pose waits for the second pass, as
                // one set before the add would.
                let (Some(target), Some(frame)) =
                    (target.filter(|t| transforms.contains(*t)), frame)
                else {
                    if target.is_none() && controls.0.contains_key(&key) {
                        log::warn!("{key:?} is a joint control; write its value instead");
                    } else if worth_waiting(&key, &controls) {
                        // Behind the gizmo its add is spawning, or behind
                        // an add still to come.
                        let placed = controls.0.contains_key(&key);
                        again |= placed;
                        defer(
                            &mut waiting,
                            if placed { &mut held } else { &mut held_changes },
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
                let local = match parents
                    .get(target)
                    .ok()
                    .and_then(|p| current_global(p.parent(), &transforms, &parents, &globals))
                {
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
            } => {
                // A control added in this pass is spawned at the end of it,
                // and a joint control resolved between the passes, so its
                // gizmo is not there yet; the change waits for the second
                // pass, as one made before the add would.
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
                            if placed { &mut held } else { &mut held_changes },
                            ControlCommand::Configure {
                                key,
                                color,
                                colors,
                                local,
                                opacity,
                                interactive,
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
                // Neither kind's own: one command may carry these for both.
                let common = opacity.is_some() || interactive.is_some();
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
                    if color.is_some() && colors.is_none() && local.is_none() && !common {
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
                    if color.is_none() && (colors.is_some() || local.is_some()) && !common {
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
                            if placed { &mut held } else { &mut held_changes },
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
    pending.0 = waiting;
    if again {
        redraw.write(RequestRedraw);
    }
}

/// Puts `command` off to a later pass, and holds every later command under
/// its key behind it.
fn defer(waiting: &mut Vec<ControlCommand>, held: &mut HashSet<String>, command: ControlCommand) {
    held.insert(command.key().to_string());
    waiting.push(command);
}

fn despawn(commands: &mut Commands, placed: Placed) {
    commands.entity(placed.gizmo).despawn();
    if let Some(goal) = placed.goal {
        commands.entity(goal).despawn();
    }
}

fn report_transform_drag(
    drag: On<GizmoDrag>,
    gizmos: Query<&TransformGizmo>,
    parents: Query<&ChildOf>,
    globals: Query<&GlobalTransform>,
    app_frame: Res<DefaultDriverFrame>,
    frames: Query<&DriverFrame>,
    mut events: ResMut<ControlEvents>,
) {
    let Ok(gizmo) = gizmos.get(drag.entity) else {
        return;
    };
    if gizmo.key.is_empty() {
        return;
    }
    // The drag gives the target's transform in its parent's frame.
    let world = match parents
        .get(drag.target)
        .ok()
        .and_then(|p| globals.get(p.parent()).ok())
    {
        Some(parent) => parent.mul_transform(drag.transform).compute_transform(),
        None => drag.transform,
    };
    events.0.push(ControlEvent {
        key: gizmo.key.clone(),
        phase: drag.phase.into(),
        value: None,
        pose: Some(Pose::from_world(
            &world,
            &DriverFrame::of(gizmo.camera, &frames, &app_frame),
        )),
        scale: Some(drag.transform.scale.to_array()),
    });
}

fn report_press(
    press: On<GizmoPress>,
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
    if key.is_empty() {
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
    gizmos: Query<&JointGizmo>,
    mut events: ResMut<ControlEvents>,
) {
    let Ok(gizmo) = gizmos.get(drag.entity) else {
        return;
    };
    if gizmo.key.is_empty() {
        return;
    }
    events.0.push(ControlEvent {
        key: gizmo.key.clone(),
        phase: drag.phase.into(),
        value: Some(drag.value),
        pose: None,
        scale: None,
    });
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use bevy::time::TimeUpdateStrategy;

    use super::*;
    use crate::binding::{ElementTargets, SceneBindings};
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
        app.add_systems(Update, resolve_joints.in_set(ControlSystems::Joints));
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
                    target: Target::Goal { pose: at },
                    modes: None,
                    local: false,
                    colors: None,
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
                move |transforms: Query<&mut Transform>,
                      parents: Query<&ChildOf>,
                      globals: Query<&GlobalTransform>| {
                    current_global(parent, &transforms, &parents, &globals)
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
        }
    }

    fn goal_at(position: [f32; 3]) -> Control {
        Control::Transform {
            target: Target::Goal {
                pose: Pose {
                    position,
                    rotation: no_turn(),
                },
            },
            modes: None,
            local: false,
            colors: None,
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
    /// both passes to a control waiting on it: a change sent with its add
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
        app.add_systems(Update, resolve_joints.in_set(ControlSystems::Joints));
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
        commands.forget_instance(ROBOT);
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
}
