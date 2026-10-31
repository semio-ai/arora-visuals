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
//! **Motion is about the point under the pointer.** Each gesture anchors on
//! the nearest element the pointer's ray meets, cast over the meshes as drawn
//! — morph targets applied — so the camera orbits the thing you pressed on
//! and zooms toward the thing you point at. Over empty space the controller
//! falls back to the depth of the last anchor, so a gesture begun beside the
//! robot does not fling the camera away.
//!
//! Drags are measured as the cursor's movement in logical pixels rather than
//! the mouse's raw motion, so a pan keeps what was pressed under the pointer
//! as the controller means it to.

use bevy::camera_controller::pan_orbit_camera::prelude::{
    DefaultPanOrbitCameraPlugins, PanOrbitCamera,
};
use bevy::input::mouse::MouseScrollUnit;
use bevy::math::DVec3;
use bevy::picking::events::{PointerPress, PointerScroll};
use bevy::picking::pointer::PointerButton;
use bevy::prelude::*;

use crate::instance::{EntityInstance, Superseded};
use crate::pick::ElementMeshes;
use crate::transform_gizmo::GizmoPointer;

/// Which inputs move a camera ([module docs](self)).
#[derive(Component, Clone, Copy, Debug)]
#[require(PanOrbitCamera)]
pub struct CameraBindings {
    pub orbit: MouseButton,
    pub pan: MouseButton,
    /// How far one line of wheel scrolling zooms, in the controller's units,
    /// which are roughly pixels of travel at the anchor.
    pub zoom_per_line: f32,
    /// How far one pixel of smooth scrolling zooms — a trackpad's.
    pub zoom_per_pixel: f32,
}

impl Default for CameraBindings {
    /// Studio's: left orbits, right pans, the wheel zooms.
    fn default() -> Self {
        Self {
            orbit: MouseButton::Left,
            pan: MouseButton::Right,
            zoom_per_line: 24.0,
            zoom_per_pixel: 1.0,
        }
    }
}

/// The gesture in hand: its camera, the button holding it, and where the
/// cursor was last frame.
#[derive(Resource, Default)]
struct Gesture(Option<(Entity, MouseButton, Vec2)>);

/// When a camera was last scrolled, so a zoom with nothing else held can end
/// once the scrolling has stopped.
#[derive(Resource, Default)]
struct Scrolled(Option<f32>);

/// How long scrolling must have stopped, in seconds, before a zoom ends. A
/// wheel delivers its scrolling in bursts with frames between them, and a zoom
/// ended in such a gap and begun again on the next burst loses the
/// controller's smoothing and re-anchors, so the zoom lurches; this bridges
/// the gaps while still freeing the camera promptly for the next gesture.
const ZOOM_QUIET: f32 = 0.25;

pub struct CameraInputPlugin;

impl Plugin for CameraInputPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(DefaultPanOrbitCameraPlugins)
            .init_resource::<Gesture>()
            .init_resource::<Scrolled>()
            .add_observer(start_gesture)
            .add_observer(zoom)
            .add_systems(Update, (drag, end_zoom));
    }
}

/// The camera whose viewport holds `position`, with its bindings, and the
/// anchor under the pointer in that camera's view space.
type Cameras<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static Camera,
        &'static GlobalTransform,
        &'static CameraBindings,
        &'static mut PanOrbitCamera,
    ),
>;

/// The nearest point of any drawn element the ray from `position` meets, in
/// the camera's view space.
fn anchor_under(
    position: Vec2,
    camera: &Camera,
    at: &GlobalTransform,
    instances: &Query<&EntityInstance, Without<Superseded>>,
    elements: &ElementMeshes,
) -> Option<DVec3> {
    let ray = camera.viewport_to_world(at, position).ok()?;
    let nearest = instances
        .iter()
        .filter(|instance| instance.ready)
        .flat_map(|instance| instance.bindings.elements.values())
        .filter_map(|targets| elements.drawn(targets).and_then(|mesh| mesh.hit(ray)))
        .min_by(f32::total_cmp)?;
    let world = ray.get_point(nearest);
    Some(at.affine().inverse().transform_point3(world).as_dvec3())
}

fn holds(position: Vec2, camera: &Camera) -> bool {
    camera
        .logical_viewport_rect()
        .is_some_and(|rect| rect.contains(position))
}

/// Starts an orbit or a pan on a press that reaches the scene.
#[allow(clippy::too_many_arguments)]
fn start_gesture(
    press: On<PointerPress>,
    windows: Query<(), With<Window>>,
    gizmos: Option<Res<GizmoPointer>>,
    mut cameras: Cameras,
    instances: Query<&EntityInstance, Without<Superseded>>,
    elements: ElementMeshes,
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
    for (entity, camera, at, bindings, mut controller) in &mut cameras {
        if !holds(position, camera) {
            continue;
        }
        let anchor = anchor_under(position, camera, at, &instances, &elements);
        if button == bindings.orbit {
            controller.start_orbit(anchor);
        } else if button == bindings.pan {
            controller.start_pan(anchor);
        } else {
            continue;
        }
        gesture.0 = Some((entity, button, position));
        return;
    }
}

/// Feeds the gesture in hand the cursor's movement, and ends it when its
/// button comes up.
fn drag(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    mut cameras: Query<&mut PanOrbitCamera>,
    mut gesture: ResMut<Gesture>,
) {
    let Some((entity, button, last)) = gesture.0 else {
        return;
    };
    let Ok(mut controller) = cameras.get_mut(entity) else {
        gesture.0 = None;
        return;
    };
    if !buttons.pressed(button) {
        controller.end_move();
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

/// Zooms the camera under a scroll that reaches the scene, toward the point
/// under the pointer, joining an orbit or pan in hand rather than ending it.
fn zoom(
    scroll: On<PointerScroll>,
    windows: Query<(), With<Window>>,
    mut cameras: Cameras,
    instances: Query<&EntityInstance, Without<Superseded>>,
    elements: ElementMeshes,
    time: Res<Time>,
    mut scrolled: ResMut<Scrolled>,
) {
    if scroll.entity != scroll.original_event_target() || !windows.contains(scroll.entity) {
        return;
    }
    let position = scroll.pointer.position;
    for (_, camera, at, bindings, mut controller) in &mut cameras {
        if !holds(position, camera) {
            continue;
        }
        let amount = scroll.y
            * match scroll.unit {
                MouseScrollUnit::Line => bindings.zoom_per_line,
                MouseScrollUnit::Pixel => bindings.zoom_per_pixel,
            };
        if !controller.current_motion.is_user_controlled() {
            let anchor = anchor_under(position, camera, at, &instances, &elements);
            controller.start_zoom(anchor);
        }
        controller.send_zoom_input(amount);
        scrolled.0 = Some(time.elapsed_secs());
        return;
    }
}

/// Ends a zoom with nothing else held once the scrolling has stopped for
/// [`ZOOM_QUIET`], so the controller is free to start the next gesture.
fn end_zoom(
    time: Res<Time>,
    mut cameras: Query<&mut PanOrbitCamera>,
    mut scrolled: ResMut<Scrolled>,
) {
    let Some(last) = scrolled.0 else {
        return;
    };
    if time.elapsed_secs() - last < ZOOM_QUIET {
        return;
    }
    for mut controller in &mut cameras {
        if controller.current_motion.is_zooming_only() {
            controller.end_move();
        }
    }
    scrolled.0 = None;
}
