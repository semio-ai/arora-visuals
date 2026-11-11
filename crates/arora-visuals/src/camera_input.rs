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
//! **A floor, when a host sets one, keeps the camera above it**
//! ([`CameraBindings::floor`]), as Studio keeps its camera above the ground:
//! wherever an orbit, pan or zoom leaves the camera below the floor, it is
//! raised back to it and turned to look at the target again, so an orbit
//! that dips toward the floor slides along it instead. A face editor has no
//! floor and sets none.
//!
//! Drags are measured as the cursor's movement in logical pixels rather than
//! the mouse's raw motion, so a pan keeps what was pressed under the pointer
//! as the controller means it to.
//!
//! **A drag carries on past the window's edge**, as Studio's does: the cursor
//! is followed through the moves the platform reports while the button is
//! held, which go on outside the window — a browser captures the pointer to
//! the canvas on the press — and not through the window's cursor position,
//! which goes empty outside it. Followed that way, a drag out and back would
//! stall at the edge and then jump by everything it travelled outside.

use bevy::camera::{NormalizedRenderTarget, RenderTarget};
use bevy::camera_controller::pan_orbit_camera::SyncCameraPosition;
use bevy::camera_controller::pan_orbit_camera::controller::momentum::Velocity;
use bevy::camera_controller::pan_orbit_camera::controller::motion::CurrentMotion;
use bevy::camera_controller::pan_orbit_camera::prelude::{
    DefaultPanOrbitCameraPlugins, PanOrbitCamera,
};
use bevy::ecs::system::SystemParam;
use bevy::input::mouse::MouseScrollUnit;
use bevy::input::touch::TouchPhase;
use bevy::math::DVec3;
use bevy::picking::events::{Pointer, PointerPress, PointerScroll};
use bevy::picking::pointer::PointerButton;
use bevy::prelude::*;
use bevy::window::{CursorMoved, PrimaryWindow};

use crate::pick::on_the_scene;
use crate::transform_gizmo::{GizmoPointer, TransformGizmoSystems};

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
    /// A height the camera may not go below ([module docs](self)).
    pub floor: Option<CameraFloor>,
    /// How near and how far a zoom may take the camera from its target:
    /// OrbitControls' `minDistance` and `maxDistance`. A zoom stops at either
    /// rather than passing into the target or shrinking it to nothing.
    pub min_distance: f32,
    pub max_distance: f32,
}

/// A height the camera is kept at or above, measured along `up` from the
/// world's origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraFloor {
    pub up: Dir3,
    pub height: f32,
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
            min_distance: 0.0,
            max_distance: f32::INFINITY,
            inertia: false,
            floor: None,
        }
    }
}

/// The gesture in hand.
#[derive(Resource, Default)]
struct Gesture(Option<Held>);

#[derive(Clone, Copy, Debug, PartialEq)]
struct Held {
    camera: Entity,
    button: MouseButton,
    /// The window the press landed in, whose cursor the gesture follows.
    window: Entity,
    /// Where that cursor was last frame, in logical pixels.
    last: Vec2,
}

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
            // After the gizmos' pointer handling, so a handle taken by the
            // same press that began a gesture ends the gesture before it
            // moves the view.
            .add_systems(Update, drag.after(TransformGizmoSystems))
            .add_systems(PreUpdate, keep_above_floor.after(SyncCameraPosition));
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

/// Every camera, with what it draws to, and the primary window, which a
/// camera drawing to "the primary window" means.
#[derive(SystemParam)]
struct AllCameras<'w, 's> {
    cameras: Query<'w, 's, (Entity, &'static Camera, &'static RenderTarget)>,
    primary: Query<'w, 's, Entity, With<PrimaryWindow>>,
}

/// The camera a gesture by `pointer` belongs to, if any: of the cameras
/// drawing to the window the pointer is in, the topmost active one whose
/// viewport holds it — bound or not, so a view drawn over a bound one
/// shields it — and only if that camera has bindings. An inactive camera, a
/// view beneath another, one in another window and one drawing to an image,
/// such as an outline's mask, never moves.
fn camera_under(pointer: &Pointer, all: &AllCameras, bound: &Cameras) -> Option<Entity> {
    let primary = all.primary.iter().next();
    topmost(
        pointer.position,
        all.cameras
            .iter()
            .filter(|(_, _, target)| target.normalize(primary).as_ref() == Some(&pointer.target))
            .map(|(entity, camera, _)| {
                (
                    entity,
                    camera.is_active,
                    camera.order,
                    camera.logical_viewport_rect(),
                )
            }),
    )
    .filter(|entity| bound.contains(*entity))
}

/// The topmost of `cameras` — each its entity, whether it is active, its
/// order and its viewport — whose viewport holds `position`.
fn topmost(
    position: Vec2,
    cameras: impl Iterator<Item = (Entity, bool, isize, Option<Rect>)>,
) -> Option<Entity> {
    cameras
        .filter(|(_, active, _, viewport)| {
            *active && viewport.is_some_and(|rect| rect.contains(position))
        })
        .max_by_key(|(_, _, order, _)| *order)
        .map(|(entity, ..)| entity)
}

/// Starts an orbit or a pan on a press that reaches the scene.
fn start_gesture(
    press: On<PointerPress>,
    windows: Query<(), With<Window>>,
    gizmos: Option<Res<GizmoPointer>>,
    all: AllCameras,
    mut cameras: Cameras,
    mut gesture: ResMut<Gesture>,
) {
    if !on_the_scene(press.entity, press.original_event_target(), &windows) {
        return;
    }
    // Only a drag already in hand keeps the gesture from starting. A handle
    // the pointer is merely over may never be taken — the gizmo takes only
    // the left button — and when this press does take one, the gesture gives
    // way to it at once (`drag`).
    if gizmos.is_some_and(|gizmos| gizmos.dragging.is_some()) || gesture.0.is_some() {
        return;
    }
    let button = match press.button {
        PointerButton::Primary => MouseButton::Left,
        PointerButton::Secondary => MouseButton::Right,
        PointerButton::Middle => MouseButton::Middle,
    };
    let position = press.pointer.position;
    let NormalizedRenderTarget::Window(window) = &press.pointer.target else {
        return;
    };
    let window = window.entity();
    let Some(entity) = camera_under(&press.pointer, &all, &cameras) else {
        return;
    };
    let Ok((_, _, bindings, mut controller, _)) = cameras.get_mut(entity) else {
        return;
    };
    if button == bindings.orbit {
        controller.start_orbit(None);
    } else if button == bindings.pan {
        controller.start_pan(None);
    } else {
        return;
    }
    gesture.0 = Some(Held {
        camera: entity,
        button,
        window,
        last: position,
    });
}

/// Feeds the gesture in hand the cursor's movement, and ends it when its
/// button comes up — at once, or carrying on, as the bindings ask.
fn drag(
    buttons: Res<ButtonInput<MouseButton>>,
    mut moves: MessageReader<CursorMoved>,
    gizmos: Option<Res<GizmoPointer>>,
    mut cameras: Query<(&mut PanOrbitCamera, &CameraBindings)>,
    mut gesture: ResMut<Gesture>,
) {
    // Where the latest move in the gesture's window put the cursor, wherever
    // that is relative to the window's bounds. Every move is read every
    // frame, so a gesture's first frame sees only the moves since the frame
    // before it, not ones queued while no gesture was in hand.
    let moved = moves.read().fold(None, |latest, moved| match gesture.0 {
        Some(held) if moved.window == held.window => Some(moved.position),
        _ => latest,
    });
    let Some(held) = gesture.0 else {
        return;
    };
    let Ok((mut controller, bindings)) = cameras.get_mut(held.camera) else {
        gesture.0 = None;
        return;
    };
    // A gizmo drag owns the pointer. A gesture can begin on the same press
    // that takes a handle — the first frame the pointer reaches the handle,
    // before it is known to be over it — and it yields at once, without
    // carrying on, so the view never turns under a handle in hand.
    if gizmos.is_some_and(|gizmos| gizmos.dragging.is_some()) {
        controller.end_move();
        controller.current_motion = CurrentMotion::Stationary;
        gesture.0 = None;
        return;
    }
    if !buttons.pressed(held.button) {
        controller.end_move();
        if !bindings.inertia {
            controller.current_motion = CurrentMotion::Stationary;
        }
        gesture.0 = None;
        return;
    }
    // The cursor is followed on the window or off it ([module docs](self)).
    // The movement is sent every frame, zero when the cursor held still: the
    // controller keeps applying the last input it was sent until another
    // replaces it, so a frame that sent nothing would turn the view again by
    // the previous frame's movement.
    let cursor = moved.unwrap_or(held.last);
    controller.send_screenspace_input(cursor - held.last);
    gesture.0 = Some(Held {
        last: cursor,
        ..held
    });
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
    all: AllCameras,
    mut cameras: Cameras,
    gesture: Res<Gesture>,
    time: Res<Time>,
    mut trackpad: ResMut<Trackpad>,
) {
    if !on_the_scene(scroll.entity, scroll.original_event_target(), &windows) {
        return;
    }
    // Read every scroll's phase, taken or not, so the inertia is still told
    // from the fingers when the scrolling around it was ignored.
    let coasting = trackpad.coasting(scroll.phase, time.elapsed_secs());
    if gesture.0.is_some() {
        return;
    }
    let Some(entity) = camera_under(&scroll.pointer, &all, &cameras) else {
        return;
    };
    if let Ok((_, _, bindings, mut controller, mut transform)) = cameras.get_mut(entity) {
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
        let closer = dolly(distance, pixels, bindings.zoom_speed).clamp(
            f64::from(bindings.min_distance),
            f64::from(bindings.max_distance),
        );
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
    }
}

/// Where a camera at `at`, `depth` from its target along its axis, stands
/// once kept above `floor`: raised to it if below, and turned back to the
/// target. `None` when it is not below.
fn above_floor(at: &Transform, depth: f32, floor: CameraFloor) -> Option<Transform> {
    let up = floor.up.as_vec3();
    let below = floor.height - at.translation.dot(up);
    if below <= 0.0 {
        return None;
    }
    let target = at.translation + at.forward().as_vec3() * depth;
    Some(Transform::from_translation(at.translation + up * below).looking_at(target, up))
}

/// Raises a camera the controller or a zoom has left below its floor.
fn keep_above_floor(mut cameras: Query<(&CameraBindings, &mut PanOrbitCamera, &mut Transform)>) {
    for (bindings, mut controller, mut transform) in &mut cameras {
        let Some(floor) = bindings.floor else {
            continue;
        };
        let depth = controller.last_anchor_depth().abs();
        let Some(raised) = above_floor(&transform, depth as f32, floor) else {
            continue;
        };
        // The controller's anchors are in view space; carry each through the
        // move in world space, so a motion in progress keeps its point.
        let (before, after) = (
            transform.to_matrix().as_dmat4(),
            raised.to_matrix().as_dmat4().inverse(),
        );
        let carry =
            |anchor: &mut DVec3| *anchor = after.transform_point3(before.transform_point3(*anchor));
        match &mut controller.current_motion {
            CurrentMotion::UserControlled { anchor, .. }
            | CurrentMotion::Momentum {
                velocity: Velocity::Orbit { anchor, .. } | Velocity::Pan { anchor, .. },
                ..
            } => carry(anchor),
            _ => {}
        }
        let target = transform.translation + transform.forward().as_vec3() * depth as f32;
        controller.last_anchor_depth = -f64::from(raised.translation.distance(target));
        *transform = raised;
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

    /// The topmost active camera holding the pointer is the gesture's; an
    /// inactive one, one beneath, and one whose viewport misses are not.
    #[test]
    fn a_gesture_goes_to_the_topmost_active_camera_under_it() {
        let [beneath, above, inactive, elsewhere] =
            [1, 2, 3, 4].map(|index| Entity::from_raw_u32(index).unwrap());
        let full = Some(Rect::new(0.0, 0.0, 800.0, 600.0));
        let cameras = [
            (beneath, true, 0, full),
            (above, true, 1, full),
            (inactive, false, 5, full),
            (
                elsewhere,
                true,
                9,
                Some(Rect::new(900.0, 0.0, 1000.0, 100.0)),
            ),
        ];
        assert_eq!(
            topmost(Vec2::new(10.0, 10.0), cameras.into_iter()),
            Some(above)
        );
        assert_eq!(topmost(Vec2::new(850.0, 10.0), cameras.into_iter()), None);
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

    /// A camera below the floor is raised onto it and still looks at its
    /// target; one above it is left alone.
    #[test]
    fn a_camera_below_the_floor_is_raised_and_still_faces_its_target() {
        let floor = CameraFloor {
            up: Dir3::Y,
            height: 0.01,
        };
        let target = Vec3::new(0.0, 0.7, 0.0);
        let below = Transform::from_xyz(0.0, -0.5, 3.0).looking_at(target, Vec3::Y);
        let depth = below.translation.distance(target);
        let raised = above_floor(&below, depth, floor).expect("it was below");
        assert!((raised.translation.y - 0.01).abs() < 1e-5);
        assert!((raised.translation.z - 3.0).abs() < 1e-5);
        let facing = (target - raised.translation).normalize();
        assert!(raised.forward().as_vec3().abs_diff_eq(facing, 1e-5));
        let above = Transform::from_xyz(0.0, 1.0, 3.0).looking_at(target, Vec3::Y);
        assert!(above_floor(&above, 3.0, floor).is_none());
    }

    /// A drag follows the cursor out past the window's edge and back, by
    /// the moves the window reports, and ignores another window's. A frame
    /// with no move sends no movement, rather than nothing, which would leave
    /// the controller applying the movement before it.
    #[test]
    fn a_drag_follows_the_cursor_off_the_window_and_back() {
        use bevy::camera_controller::pan_orbit_camera::controller::inputs::MotionInputs;

        fn last_input(world: &mut World, camera: Entity) -> Option<Vec2> {
            match &world.get::<PanOrbitCamera>(camera)?.current_motion {
                CurrentMotion::UserControlled {
                    motion_inputs:
                        MotionInputs::OrbitZoom {
                            screenspace_inputs, ..
                        },
                    ..
                } => screenspace_inputs
                    .iter_unsmoothed()
                    .next()
                    .map(|(_, input)| input),
                _ => None,
            }
        }

        let mut world = World::new();
        world.init_resource::<Messages<CursorMoved>>();
        let mut buttons = ButtonInput::<MouseButton>::default();
        buttons.press(MouseButton::Left);
        world.insert_resource(buttons);
        let window = world.spawn_empty().id();
        let other = world.spawn_empty().id();
        let mut controller = PanOrbitCamera::default();
        controller.start_orbit(None);
        let camera = world.spawn((controller, CameraBindings::default())).id();
        world.insert_resource(Gesture(Some(Held {
            camera,
            button: MouseButton::Left,
            window,
            last: Vec2::new(10.0, 10.0),
        })));
        // One system for the whole drag, as in an app, so its reader keeps its
        // place between frames rather than rereading the frame before.
        let drag = world.register_system(drag);
        let step = |world: &mut World, moves: &[(Entity, Vec2)]| {
            for &(window, position) in moves {
                world.write_message(CursorMoved {
                    window,
                    position,
                    delta: None,
                });
            }
            world.run_system(drag).unwrap();
            world.resource_mut::<Messages<CursorMoved>>().update();
            world.resource::<Gesture>().0.map(|held| held.last)
        };
        let off = Vec2::new(-40.0, 10.0);
        assert_eq!(
            step(&mut world, &[(window, off)]),
            Some(off),
            "followed off the window"
        );
        assert_eq!(last_input(&mut world, camera), Some(Vec2::new(-50.0, 0.0)));
        assert_eq!(
            step(
                &mut world,
                &[
                    (window, Vec2::new(-80.0, 12.0)),
                    (other, Vec2::new(5.0, 5.0))
                ]
            ),
            Some(Vec2::new(-80.0, 12.0)),
            "another window's move is not the gesture's",
        );
        assert_eq!(
            step(&mut world, &[]),
            Some(Vec2::new(-80.0, 12.0)),
            "no move is no movement"
        );
        assert_eq!(
            last_input(&mut world, camera),
            Some(Vec2::ZERO),
            "and says so"
        );
        let back = Vec2::new(30.0, 12.0);
        assert_eq!(
            step(&mut world, &[(window, back)]),
            Some(back),
            "and back on"
        );
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
