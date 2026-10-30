//! The view gizmo: the scene's axes drawn small in a corner, turning with the
//! camera, and a way to look straight down any of them.
//!
//! It follows Studio's, which is drei's `GizmoViewport`: an axis bar along
//! each positive axis, a ball labelled with the axis's letter at each positive
//! end, and a smaller, fainter, unlabelled ball at each negative one. A ball
//! grows under the pointer, and pressing it asks for the camera to look from
//! that ball's side toward what it orbits.
//!
//! **It is drawn as UI, not as a second scene.** Only the camera's rotation
//! matters — the gizmo is orthographic, as drei's is — so eleven flat nodes
//! placed each frame by projecting the axes through that rotation give the
//! same picture as a second camera and render target would, with text that is
//! crisp at any size, and pointer handling from UI picking. Siblings are
//! stacked by depth, so the axes pointing at the viewer draw over the ones
//! pointing away.
//!
//! **It asks for a view; it does not turn the camera.** A press triggers
//! [`LookFrom`], and whatever drives the camera answers it. That keeps the
//! gizmo the same whichever controller a renderer uses, and is why it carries
//! no notion of orbit, focus or distance.
//!
//! Which world direction each letter names is the [`ViewGizmo`]'s to say: a
//! renderer whose content is Z-up, as Studio's is, labels Bevy's +Y as Z.

use bevy::picking::events::{PointerClick, PointerOut, PointerOver};
use bevy::picking::Pickable;
use bevy::prelude::*;
use bevy::ui::UiTargetCamera;

/// A camera's view gizmo. Put it on the camera whose view it shows.
#[derive(Component, Clone, Debug)]
pub struct ViewGizmo {
    /// The world directions the gizmo labels X, Y and Z, in that order.
    pub axes: [Dir3; 3],
    /// The axes' colours, in the same order.
    pub colours: [Color; 3],
    /// The letters on the positive balls.
    pub labels: [String; 3],
    pub label_colour: Color,
    /// How far the gizmo's centre is from the right and bottom of the view,
    /// in logical pixels. A host whose panels cover part of the view adds what
    /// they cover, so the gizmo sits in the corner the user can see.
    pub inset: Vec2,
}

impl Default for ViewGizmo {
    /// Studio's: its axis colours, black letters, 80 pixels in, and Bevy's own
    /// axes. A Z-up renderer replaces `axes` with [`ViewGizmo::Z_UP`].
    fn default() -> Self {
        Self {
            axes: [Dir3::X, Dir3::Y, Dir3::Z],
            colours: [
                Color::srgb_u8(0xef, 0x44, 0x44),
                Color::srgb_u8(0x22, 0xc5, 0x5e),
                Color::srgb_u8(0x3b, 0x82, 0xf6),
            ],
            labels: ["X".into(), "Y".into(), "Z".into()],
            label_colour: Color::BLACK,
            inset: Vec2::splat(80.0),
        }
    }
}

impl ViewGizmo {
    /// The axes of a Z-up world as Bevy's Y-up one holds it; the same as the
    /// transform gizmo's, so the two agree on which way each axis points.
    pub const Z_UP: [Dir3; 3] = crate::transform_gizmo::TransformGizmo::Z_UP;
}

/// A request, from a press on the gizmo, for `camera` to look from the side
/// `from` points to, toward whatever it is looking at.
#[derive(Event, Clone, Copy, Debug)]
pub struct LookFrom {
    pub camera: Entity,
    pub from: Dir3,
}

/// Logical pixels per unit of the gizmo: drei's scale.
const UNIT: f32 = 40.0;
/// An axis bar runs from the centre to here, in units.
const BAR_LENGTH: f32 = 0.8;
const BAR_THICKNESS: f32 = 2.0;
/// A positive ball's radius, in pixels: a sixteen-pixel circle on drei's
/// sixty-four-pixel sprite one unit across.
const BALL_RADIUS: f32 = 10.0;
/// A negative ball's size and opacity against a positive one's.
const NEGATIVE: f32 = 0.75;
/// A ball under the pointer, against one that is not.
const HOVERED: f32 = 1.2;
/// drei's 18-pixel label on the same sprite.
const LABEL_SIZE: f32 = 18.0 * UNIT / 64.0;
/// How far a label is lowered, as a share of its size. Centring the text's
/// line box leaves a capital letter high, since the box keeps room below the
/// baseline for descenders the axis letters do not have.
const LABEL_DROP: f32 = 0.08;
/// Half the gizmo's box: wide enough for a hovered ball at full reach.
const HALF: f32 = UNIT + BALL_RADIUS * HOVERED + 2.0;

/// The root of a camera's gizmo.
#[derive(Component)]
struct GizmoRoot(Entity);

/// A part of a gizmo, by the axis it draws and which way along it.
#[derive(Component, Clone, Copy)]
enum Part {
    Bar(usize),
    Ball { axis: usize, positive: bool },
}

/// A ball the pointer is over.
#[derive(Component)]
struct Hovered;

pub struct ViewGizmoPlugin;

impl Plugin for ViewGizmoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (build_gizmos, place_gizmos).chain());
    }
}

/// Builds a gizmo for each camera that has gained a [`ViewGizmo`], replacing
/// one whose settings have changed.
fn build_gizmos(
    mut commands: Commands,
    cameras: Query<(Entity, &ViewGizmo), Changed<ViewGizmo>>,
    roots: Query<(Entity, &GizmoRoot)>,
) {
    for (camera, gizmo) in &cameras {
        for (root, of) in &roots {
            if of.0 == camera {
                commands.entity(root).despawn();
            }
        }
        commands
            .spawn((
                GizmoRoot(camera),
                UiTargetCamera(camera),
                Node {
                    position_type: PositionType::Absolute,
                    right: Val::Px(gizmo.inset.x - HALF),
                    bottom: Val::Px(gizmo.inset.y - HALF),
                    width: Val::Px(HALF * 2.0),
                    height: Val::Px(HALF * 2.0),
                    ..default()
                },
                // The box itself takes no pointer, so the scene behind its
                // empty corners can still be orbited and picked.
                Pickable::IGNORE,
            ))
            .with_children(|root| {
                for axis in 0..3 {
                    root.spawn((
                        Part::Bar(axis),
                        centred(UNIT * BAR_LENGTH, BAR_THICKNESS),
                        BackgroundColor(gizmo.colours[axis]),
                        Pickable::IGNORE,
                    ));
                }
                for axis in 0..3 {
                    for positive in [true, false] {
                        let size = BALL_RADIUS * 2.0 * if positive { 1.0 } else { NEGATIVE };
                        let colour = gizmo.colours[axis];
                        let colour = if positive {
                            colour
                        } else {
                            colour.with_alpha(NEGATIVE)
                        };
                        let mut ball = root.spawn((
                            Part::Ball { axis, positive },
                            Node {
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                border_radius: BorderRadius::MAX,
                                ..centred(size, size)
                            },
                            BackgroundColor(colour),
                        ));
                        if positive {
                            ball.with_child((
                                Text::new(gizmo.labels[axis].clone()),
                                TextFont::from_font_size(LABEL_SIZE),
                                Node {
                                    top: Val::Px(LABEL_SIZE * LABEL_DROP),
                                    ..default()
                                },
                                TextColor(gizmo.label_colour),
                                Pickable::IGNORE,
                            ));
                        }
                        let direction = if positive {
                            gizmo.axes[axis]
                        } else {
                            -gizmo.axes[axis]
                        };
                        ball.observe(|over: On<PointerOver>, mut commands: Commands| {
                            commands.entity(over.entity).insert(Hovered);
                        })
                        .observe(|out: On<PointerOut>, mut commands: Commands| {
                            commands.entity(out.entity).remove::<Hovered>();
                        })
                        .observe(
                            move |click: On<PointerClick>, mut commands: Commands| {
                                if click.button == PointerButton::Primary {
                                    commands.trigger(LookFrom {
                                        camera,
                                        from: direction,
                                    });
                                }
                            },
                        );
                    }
                }
            });
    }
}

/// A node of this size whose centre is the gizmo's centre, for a transform to
/// move from there.
fn centred(width: f32, height: f32) -> Node {
    Node {
        position_type: PositionType::Absolute,
        left: Val::Px(HALF - width / 2.0),
        top: Val::Px(HALF - height / 2.0),
        width: Val::Px(width),
        height: Val::Px(height),
        ..default()
    }
}

/// Places every part of every gizmo where its axis projects through its
/// camera's current rotation, and stacks the parts nearest the viewer on top.
fn place_gizmos(
    roots: Query<(&GizmoRoot, &Children)>,
    cameras: Query<(&GlobalTransform, &ViewGizmo)>,
    mut parts: Query<(&Part, &mut UiTransform, &mut ZIndex, Has<Hovered>)>,
) {
    for (root, children) in &roots {
        let Ok((camera, gizmo)) = cameras.get(root.0) else {
            continue;
        };
        let to_view = camera.rotation().inverse();
        // An axis's offset from the centre on screen, in units with y down,
        // and how far toward the viewer it points.
        let project = |axis: usize, sign: f32| {
            let v = to_view * (*gizmo.axes[axis] * sign);
            (Vec2::new(v.x, -v.y), v.z)
        };
        for child in children.iter() {
            let Ok((part, mut transform, mut z, hovered)) = parts.get_mut(child) else {
                continue;
            };
            let depth = match *part {
                Part::Bar(axis) => {
                    let (offset, depth) = project(axis, 1.0);
                    // The bar's node is its full length lying along +x about
                    // the centre; shortened by how far it is foreshortened,
                    // turned to point along the axis, and moved out so it
                    // starts at the centre.
                    let reach = offset.length();
                    *transform = UiTransform {
                        translation: Val2::px(
                            offset.x * UNIT * BAR_LENGTH / 2.0,
                            offset.y * UNIT * BAR_LENGTH / 2.0,
                        ),
                        scale: Vec2::new(reach.max(1e-3), 1.0),
                        rotation: Rot2::radians(offset.y.atan2(offset.x)),
                    };
                    depth * BAR_LENGTH / 2.0
                }
                Part::Ball { axis, positive } => {
                    let (offset, depth) = project(axis, if positive { 1.0 } else { -1.0 });
                    *transform = UiTransform {
                        translation: Val2::px(offset.x * UNIT, offset.y * UNIT),
                        scale: Vec2::splat(if hovered { HOVERED } else { 1.0 }),
                        rotation: Rot2::IDENTITY,
                    };
                    depth
                }
            };
            *z = ZIndex((depth * 1000.0) as i32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Z-up axes are a right-handed set, as the axes they stand for are:
    /// X cross Y is Z.
    #[test]
    fn z_up_axes_are_right_handed() {
        let [x, y, z] = ViewGizmo::Z_UP;
        assert!(x.cross(*y).abs_diff_eq(*z, 1e-6));
    }
}
