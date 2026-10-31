//! The editing camera's input: the bindings that drive Bevy's pan-orbit
//! controller, which ships none of its own.
//!
//! A host puts Bevy's `PanOrbitCamera` and a [`CameraBindings`] on a camera.
//! The bindings default to Studio's: the left button orbits, the right pans,
//! the wheel zooms.
//!
//! **A gesture is the camera's only if it begins on the scene.** A press that
//! lands on a UI node — a panel, a toggle, a slider — reaches that node and
//! not the window, so it never starts one; neither does a press that takes a
//! gizmo's handle. Once begun, a gesture is the camera's until its button
//! comes up, wherever the pointer goes, so nothing it passes over can take
//! it. A scroll over the scene zooms; over UI it does not.
//!
//! **Every gesture moves about the same point: the view's target.** It sits on
//! the camera's axis at the controller's anchor depth, which the host sets
//! when it frames the scene. An orbit turns about it, a pan carries it with
//! the view, and a zoom closes on it. Where the press lands does not matter,
//! on the robot or beside it, so a gesture behaves the same from anywhere in
//! the scene — as an orbit target does in three.js's OrbitControls.
//!
//! **A zoom is OrbitControls' zoom.** Each scroll moves the camera along its
//! axis at once, scaling its distance to the target by
//! `0.95^(|pixels| × 0.01 × zoom_speed)`: no smoothing to spread a scroll
//! over the frames after it, and no momentum of its own. What the camera has
//! moved is exactly what has been scrolled, so a zoom stops the moment the
//! scrolling does. As in OrbitControls, a scroll during an orbit or a pan is
//! ignored.
//!
//! **Inertia is one choice, as Studio's damping is.** With
//! [`CameraBindings::inertia`] off, an orbit or pan stops when its button
//! comes up, and a trackpad zoom stops when the fingers lift: macOS keeps
//! scrolling for a while afterward, as the fingers would have carried on,
//! and the camera ignores that. With it on, an orbit or pan carries on and
//! slows, and the trackpad's own inertia zooms on. A mouse wheel has no
//! inertia either way.
//!
//! Drags are measured as the cursor's movement in logical pixels rather than
//! the mouse's raw motion, so a pan keeps what was pressed under the pointer
//! as the controller means it to.

use bevy::camera_controller::pan_orbit_camera::controller::momentum::Velocity;
use bevy::camera_controller::pan_orbit_camera::controller::motion::CurrentMotion;
use bevy::camera_controller::pan_orbit_camera::prelude::{
    DefaultPanOrbitCameraPlugins, PanOrbitCamera,
};
use bevy::input::mouse::MouseScrollUnit;
use bevy::input::touch::TouchPhase;
use bevy::picking::events::{PointerPress, PointerScroll};
use bevy::picking::pointer::PointerButton;
use bevy::prelude::*;

use crate::transform_gizmo::GizmoPointer;

/// Which inputs move a camera ([module docs](self)).
#[derive(Component, Clone, Copy, Debug)]
#[require(PanOrbitCamera)]
pub struct CameraBindings {
    pub orbit: MouseButton,
    pub pan: MouseButton,
    /// How strongly a scroll zooms: OrbitControls' `zoomSpeed`.
    pub zoom_speed: f32,
    /// How many pixels of scrolling one line of a wheel counts as. A wheel
    /// reports lines and a trackpad pixels; this puts them on one scale.
    pub pixels_per_line: f32,
    /// Whether motion carries on after the input stops ([module docs](self)).
    pub inertia: bool,
}

impl Default for CameraBindings {
    /// Studio's: left orbits, right pans, the wheel zooms.
    fn default() -> Self {
        Self {
            orbit: MouseButton::Left,
            pan: MouseButton::Right,
            zoom_speed: 1.0,
            // A browser's wheel step, so a notch zooms as it does in Studio.
            pixels_per_line: 100.0,
            inertia: false,
        }
    }
}

/// The gesture in hand: its camera, the button holding it, and where the
/// cursor was last frame.
#[derive(Resource, Default)]
struct Gesture(Option<(Entity, MouseButton, Vec2)>);

/// Where a trackpad's scrolling stands, read from the phases its scrolls
/// carry. The fingers' scrolling runs from a start to an end; the inertia
/// that follows runs from a second start, straight after that end, to its own
/// end. A mouse wheel's scrolls carry no phase but movement, and leave this
/// as it is.
#[derive(Resource, Default)]
struct Trackpad {
    /// When the fingers last lifted.
    lifted: Option<f32>,
    /// Whether the scrolling now arriving is the inertia's.
    coasting: bool,
}

/// How soon after the fingers lift a start is the inertia's rather than a
/// new gesture's, in seconds. macOS begins the inertia at once, and fingers
/// cannot come back down that fast.
const INERTIA_FOLLOWS: f32 = 0.05;

impl Trackpad {
    /// Takes in a scroll's phase at `now`, and says whether the scroll is the
    /// inertia's.
    fn coasting(&mut self, phase: TouchPhase, now: f32) -> bool {
        match phase {
            TouchPhase::Started => {
                self.coasting = self
                    .lifted
                    .take()
                    .is_some_and(|lifted| now - lifted <= INERTIA_FOLLOWS);
                self.coasting
            }
            TouchPhase::Ended | TouchPhase::Canceled => {
                let coasting = self.coasting;
                if coasting {
                    self.coasting = false;
                } else {
                    self.lifted = Some(now);
                }
                coasting
            }
            TouchPhase::Moved => self.coasting,
        }
    }
}

pub struct CameraInputPlugin;

impl Plugin for CameraInputPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(DefaultPanOrbitCameraPlugins)
            .init_resource::<Gesture>()
            .init_resource::<Trackpad>()
            .add_observer(start_gesture)
            .add_observer(zoom)
            .add_systems(Update, drag);
    }
}

/// The cameras a gesture can move, with their bindings.
type Cameras<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static Camera,
        &'static CameraBindings,
        &'static mut PanOrbitCamera,
        &'static mut Transform,
    ),
>;

fn holds(position: Vec2, camera: &Camera) -> bool {
    camera
        .logical_viewport_rect()
        .is_some_and(|rect| rect.contains(position))
}

/// Starts an orbit or a pan on a press that reaches the scene.
fn start_gesture(
    press: On<PointerPress>,
    windows: Query<(), With<Window>>,
    gizmos: Option<Res<GizmoPointer>>,
    mut cameras: Cameras,
    mut gesture: ResMut<Gesture>,
) {
    // A press bubbles from where it landed; take it once, and only where it
    // landed on the window rather than on something drawn over the scene.
    if press.entity != press.original_event_target() || !windows.contains(press.entity) {
        return;
    }
    if gizmos.is_some_and(|gizmos| gizmos.engaged()) || gesture.0.is_some() {
        return;
    }
    let button = match press.button {
        PointerButton::Primary => MouseButton::Left,
        PointerButton::Secondary => MouseButton::Right,
        PointerButton::Middle => MouseButton::Middle,
    };
    let position = press.pointer.position;
    for (entity, camera, bindings, mut controller, _) in &mut cameras {
        if !holds(position, camera) {
            continue;
        }
        if button == bindings.orbit {
            controller.start_orbit(None);
        } else if button == bindings.pan {
            controller.start_pan(None);
        } else {
            continue;
        }
        gesture.0 = Some((entity, button, position));
        return;
    }
}

/// Feeds the gesture in hand the cursor's movement, and ends it when its
/// button comes up — at once, or carrying on, as the bindings ask.
fn drag(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    mut cameras: Query<(&mut PanOrbitCamera, &CameraBindings)>,
    mut gesture: ResMut<Gesture>,
) {
    let Some((entity, button, last)) = gesture.0 else {
        return;
    };
    let Ok((mut controller, bindings)) = cameras.get_mut(entity) else {
        gesture.0 = None;
        return;
    };
    if !buttons.pressed(button) {
        controller.end_move();
        if !bindings.inertia {
            controller.current_motion = CurrentMotion::Stationary;
        }
        gesture.0 = None;
        return;
    }
    // Off the window the cursor has no position; the gesture holds still
    // until it returns rather than ending.
    let Some(cursor) = windows.iter().find_map(Window::cursor_position) else {
        return;
    };
    controller.send_screenspace_input(cursor - last);
    gesture.0 = Some((entity, button, cursor));
}

/// The camera's distance to its target after scrolling `pixels` — upward
/// positive, closing on the target — as OrbitControls' wheel zoom scales it.
fn dolly(distance: f64, pixels: f32, zoom_speed: f32) -> f64 {
    let scale = 0.95_f64.powf(f64::from(zoom_speed * (pixels * 0.01).abs()));
    if pixels > 0.0 {
        distance * scale
    } else {
        distance / scale
    }
}

/// Zooms the camera under a scroll that reaches the scene toward its target,
/// unless an orbit or pan is in hand.
fn zoom(
    scroll: On<PointerScroll>,
    windows: Query<(), With<Window>>,
    mut cameras: Cameras,
    gesture: Res<Gesture>,
    time: Res<Time>,
    mut trackpad: ResMut<Trackpad>,
) {
    if scroll.entity != scroll.original_event_target() || !windows.contains(scroll.entity) {
        return;
    }
    // Read every scroll's phase, taken or not, so the inertia is still told
    // from the fingers when the scrolling around it was ignored.
    let coasting = trackpad.coasting(scroll.phase, time.elapsed_secs());
    if gesture.0.is_some() {
        return;
    }
    let position = scroll.pointer.position;
    for (_, camera, bindings, mut controller, mut transform) in &mut cameras {
        if !holds(position, camera) {
            continue;
        }
        if coasting && !bindings.inertia {
            return;
        }
        let pixels = scroll.y
            * match scroll.unit {
                MouseScrollUnit::Line => bindings.pixels_per_line,
                MouseScrollUnit::Pixel => 1.0,
            };
        if pixels == 0.0 {
            return;
        }
        let distance = controller.last_anchor_depth().abs();
        let closer = dolly(distance, pixels, bindings.zoom_speed);
        let forward = transform.forward().as_vec3();
        transform.translation += forward * (distance - closer) as f32;
        controller.last_anchor_depth = -closer;
        // An orbit or pan still carrying on turns about the target where it
        // now stands.
        if let CurrentMotion::Momentum {
            velocity: Velocity::Orbit { anchor, .. } | Velocity::Pan { anchor, .. },
            ..
        } = &mut controller.current_motion
        {
            anchor.z = -closer;
        }
        return;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scrolling that starts straight after the fingers lift is the
    /// inertia's, until it ends; a mouse wheel's is never.
    #[test]
    fn inertia_is_the_scrolling_that_follows_a_lift() {
        let mut pad = Trackpad::default();
        assert!(!pad.coasting(TouchPhase::Started, 0.0));
        assert!(!pad.coasting(TouchPhase::Moved, 0.1));
        assert!(!pad.coasting(TouchPhase::Ended, 0.2));
        assert!(pad.coasting(TouchPhase::Started, 0.2));
        assert!(pad.coasting(TouchPhase::Moved, 0.5));
        assert!(pad.coasting(TouchPhase::Ended, 0.9));
        assert!(!pad.coasting(TouchPhase::Moved, 1.0), "a wheel after it");
    }

    /// A scroll up closes on the target by OrbitControls' factor, and the same
    /// scroll down undoes it exactly.
    #[test]
    fn a_scroll_scales_the_distance_and_its_reverse_undoes_it() {
        let closer = dolly(2.0, 100.0, 1.0);
        assert!((closer - 2.0 * 0.95).abs() < 1e-9, "{closer}");
        assert!((dolly(closer, -100.0, 1.0) - 2.0).abs() < 1e-9);
        assert!((dolly(2.0, 100.0, 2.0) - 2.0 * 0.95 * 0.95).abs() < 1e-9);
    }

    /// Fingers that come back down after a pause start a gesture of their
    /// own, not inertia.
    #[test]
    fn a_later_start_is_a_new_gesture() {
        let mut pad = Trackpad::default();
        pad.coasting(TouchPhase::Started, 0.0);
        pad.coasting(TouchPhase::Ended, 0.2);
        assert!(!pad.coasting(TouchPhase::Started, 0.6));
        assert!(!pad.coasting(TouchPhase::Moved, 0.7));
    }
}
