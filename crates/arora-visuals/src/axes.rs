//! Showing a reference frame: three arrows along an entity's own X, Y and Z,
//! from its origin, dashed lines reaching well past them, or both, so where
//! a body or a joint's frame sits and how it is turned can be seen.
//!
//! **A view, not a default.** Nothing shows axes until a host puts
//! [`FrameAxes`] on an entity — a body, a joint, an element's node, a gizmo's
//! target — and removing it takes them away again. Any number of entities may
//! show theirs at once.
//!
//! The arrows are the transform gizmo's, at rest and without its handles:
//! unlit, in its axis colours, and drawn over everything, so a frame inside a
//! body still shows. They follow the entity's position and turn but not its
//! scale, so a scaled node's axes keep their length; [`FrameAxes::size`] sets
//! that length, in the world or on screen, as a gizmo's size does. They are
//! never picked, cast no shadow, and are left alone by a ghost.
//!
//! The lines are [`AxisLine`]s, the guides a gizmo shows along a held
//! handle's axis, measured in the arrows' length. Run them both ways when the
//! arrows show too, since the arrows already say which way is positive, and
//! one way when the lines show alone.

use bevy::light::NotShadowCaster;
use bevy::prelude::*;

use crate::axis_line::{AxisLine, AxisLinePlugin};
use crate::transform_gizmo::{
    add_handle_material, unit_size, GizmoSize, HandleMaterial, HandleUniform, AXIS_COLOURS,
    CONE_LENGTH, CONE_RADIUS, SHAFT_LENGTH, TUBE,
};

/// Shows this entity's reference frame ([module docs](self)).
#[derive(Component, Clone, Debug)]
pub struct FrameAxes {
    /// How long each arrow is. A size on screen is measured through
    /// `camera`, and shows nothing without one.
    pub size: GizmoSize,
    pub camera: Option<Entity>,
    /// The entity's directions the arrows name X, Y and Z: its own by
    /// default, or [`Z_UP`](crate::transform_gizmo::TransformGizmo::Z_UP) to label a Y-up frame as a Z-up
    /// world names it, as the gizmos can.
    pub axes: [Dir3; 3],
    /// X's, Y's and Z's colours.
    pub colours: [Color; 3],
    /// Whether the arrows show.
    pub arrows: bool,
    /// The dashed lines along the axes, if they show.
    pub lines: Option<FrameLines>,
}

/// A frame's dashed lines ([module docs](self)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameLines {
    /// How far each reaches from the origin, in arrow lengths.
    pub reach: f32,
    /// Whether each runs back through the origin as far again.
    pub both_ways: bool,
}

impl Default for FrameLines {
    /// Four arrow lengths, both ways, as beside arrows.
    fn default() -> Self {
        Self {
            reach: 4.0,
            both_ways: true,
        }
    }
}

impl Default for FrameAxes {
    /// Studio's body axes: three tenths of a metre, in the gizmos' colours.
    fn default() -> Self {
        Self {
            size: GizmoSize::World(0.3),
            camera: None,
            axes: [Dir3::X, Dir3::Y, Dir3::Z],
            colours: AXIS_COLOURS,
            arrows: true,
            lines: None,
        }
    }
}

/// The arrows showing an entity's axes, held under it; `0` is the entity.
#[derive(Component)]
struct AxesOf(Entity);

/// One of the lines showing an entity's axes, under it; `0` is the entity.
#[derive(Component)]
struct LinesOf(Entity);

/// What an entity's axes were built as, so a change rebuilds them.
#[derive(Component, PartialEq)]
struct Built {
    axes: [Dir3; 3],
    colours: [Color; 3],
    arrows: bool,
    lines: Option<FrameLines>,
    size: GizmoSize,
    camera: Option<Entity>,
}

impl Built {
    fn of(axes: &FrameAxes) -> Self {
        Self {
            axes: axes.axes,
            colours: axes.colours,
            arrows: axes.arrows,
            lines: axes.lines,
            size: axes.size,
            camera: axes.camera,
        }
    }
}

pub struct FrameAxesPlugin;

impl Plugin for FrameAxesPlugin {
    fn build(&self, app: &mut App) {
        add_handle_material(app);
        if !app.is_plugin_added::<AxisLinePlugin>() {
            app.add_plugins(AxisLinePlugin);
        }
        app.add_systems(
            PostUpdate,
            (build_axes, (size_axes, drop_axes))
                .chain()
                .after(TransformSystems::Propagate),
        );
    }
}

/// Builds the arrows and lines for each entity newly showing its axes, or
/// showing them differently.
fn build_axes(
    mut commands: Commands,
    shown: Query<(Entity, &FrameAxes, Option<&Built>)>,
    holders: Query<(Entity, &AxesOf)>,
    lines: Query<(Entity, &LinesOf)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<HandleMaterial>>,
) {
    for (entity, axes, built) in &shown {
        let wanted = Built::of(axes);
        if built == Some(&wanted) {
            continue;
        }
        despawn_shown(&mut commands, entity, &holders, &lines);
        commands.entity(entity).insert(wanted);

        if let Some(style) = axes.lines {
            for (index, colour) in axes.colours.into_iter().enumerate() {
                commands.spawn((
                    LinesOf(entity),
                    AxisLine {
                        direction: axes.axes[index],
                        reach: style.reach,
                        both_ways: style.both_ways,
                        size: axes.size,
                        camera: axes.camera,
                        colour,
                    },
                    Transform::default(),
                    Visibility::default(),
                    ChildOf(entity),
                ));
            }
        }
        if !axes.arrows {
            continue;
        }

        let shaft = meshes.add(Cylinder::new(TUBE[0], SHAFT_LENGTH));
        // The head stands on the shaft's end, its base at the piece's origin.
        let head = meshes.add(
            Mesh::from(Cone {
                radius: CONE_RADIUS,
                height: CONE_LENGTH,
            })
            .translated_by(Vec3::Y * CONE_LENGTH / 2.0),
        );
        let holder = commands
            .spawn((
                AxesOf(entity),
                Transform::default(),
                Visibility::default(),
                ChildOf(entity),
            ))
            .id();
        for (index, colour) in axes.colours.into_iter().enumerate() {
            let material = materials.add(HandleMaterial {
                handle: HandleUniform {
                    colour: LinearRgba::from(colour).to_vec4(),
                },
            });
            // The meshes stand along Y; each arrow is turned onto its axis.
            let along = axes.axes[index].as_vec3();
            let onto = Quat::from_rotation_arc(Vec3::Y, along);
            for (mesh, from) in [(&shaft, SHAFT_LENGTH / 2.0), (&head, SHAFT_LENGTH)] {
                commands.spawn((
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(material.clone()),
                    Transform::from_translation(along * from).with_rotation(onto),
                    NotShadowCaster,
                    Pickable::IGNORE,
                    ChildOf(holder),
                ));
            }
        }
    }
}

/// Sizes each entity's arrows, undoing its scale so they keep their length.
fn size_axes(
    shown: Query<(&FrameAxes, &GlobalTransform)>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    mut holders: Query<(&AxesOf, &mut Transform, &mut Visibility)>,
) {
    for (of, mut transform, mut visibility) in &mut holders {
        let Ok((axes, at)) = shown.get(of.0) else {
            continue;
        };
        let camera = axes.camera.and_then(|camera| cameras.get(camera).ok());
        let unit = match (axes.size, camera) {
            (GizmoSize::World(metres), _) => Some(metres),
            (size, Some(camera)) => unit_size(size, camera, at.translation()),
            (GizmoSize::Pixels(_), None) => None,
        };
        let shown = if unit.is_some() {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        visibility.set_if_neq(shown);
        let Some(unit) = unit else {
            continue;
        };
        let scale = at.scale();
        let wanted = Vec3::splat(unit) / scale.max(Vec3::splat(f32::EPSILON));
        if transform.scale != wanted {
            transform.scale = wanted;
        }
    }
}

/// Takes the arrows and lines away from an entity no longer showing its
/// axes.
fn drop_axes(
    mut commands: Commands,
    mut removed: RemovedComponents<FrameAxes>,
    holders: Query<(Entity, &AxesOf)>,
    lines: Query<(Entity, &LinesOf)>,
) {
    for entity in removed.read() {
        if let Ok(mut built) = commands.get_entity(entity) {
            built.remove::<Built>();
        }
        despawn_shown(&mut commands, entity, &holders, &lines);
    }
}

/// Despawns whatever shows `entity`'s axes.
fn despawn_shown(
    commands: &mut Commands,
    entity: Entity,
    holders: &Query<(Entity, &AxesOf)>,
    lines: &Query<(Entity, &LinesOf)>,
) {
    let arrows = holders
        .iter()
        .filter(|(_, of)| of.0 == entity)
        .map(|(e, _)| e);
    let dashed = lines
        .iter()
        .filter(|(_, of)| of.0 == entity)
        .map(|(e, _)| e);
    for shown in arrows.chain(dashed) {
        commands.entity(shown).despawn();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default(), TransformPlugin))
            .init_asset::<Mesh>()
            .init_asset::<HandleMaterial>()
            .add_systems(
                PostUpdate,
                (build_axes, (size_axes, drop_axes))
                    .chain()
                    .after(TransformSystems::Propagate),
            );
        app
    }

    fn arrows(app: &mut App) -> usize {
        app.world_mut()
            .query_filtered::<(), With<MeshMaterial3d<HandleMaterial>>>()
            .iter(app.world())
            .count()
    }

    /// Showing a frame puts three arrows under it, each a shaft and a head,
    /// and taking it away removes them.
    #[test]
    fn showing_axes_adds_three_arrows_and_hiding_them_removes_them() {
        let mut app = app();
        let body = app
            .world_mut()
            .spawn((Transform::default(), FrameAxes::default()))
            .id();
        app.update();
        assert_eq!(arrows(&mut app), 6);
        app.world_mut().entity_mut(body).remove::<FrameAxes>();
        app.update();
        assert_eq!(arrows(&mut app), 0);
    }

    /// Under the Z-up labelling, Z's arrow — the third colour — stands along
    /// Bevy's up.
    #[test]
    fn a_z_up_frame_puts_z_up() {
        let mut app = app();
        app.world_mut().spawn((
            Transform::default(),
            FrameAxes {
                axes: crate::transform_gizmo::TransformGizmo::Z_UP,
                ..default()
            },
        ));
        app.update();
        let blue = LinearRgba::from(AXIS_COLOURS[2]).to_vec4();
        let world = app.world_mut();
        let pieces: Vec<(Vec3, Handle<HandleMaterial>)> = world
            .query::<(&Transform, &MeshMaterial3d<HandleMaterial>)>()
            .iter(world)
            .map(|(at, material)| (at.translation, material.0.clone()))
            .collect();
        let materials = world.resource::<Assets<HandleMaterial>>();
        let z: Vec<Vec3> = pieces
            .into_iter()
            .filter(|(_, material)| materials.get(material).unwrap().handle.colour == blue)
            .map(|(at, _)| at)
            .collect();
        assert_eq!(z.len(), 2, "a shaft and a head");
        for at in z {
            assert!(at.normalize().abs_diff_eq(Vec3::Y, 1e-5), "{at:?}");
        }
    }

    /// Lines alone show three dashed lines and no arrows; taking the axes
    /// away removes them too.
    #[test]
    fn lines_alone_show_three_lines_and_no_arrows() {
        let mut app = app();
        let body = app
            .world_mut()
            .spawn((
                Transform::default(),
                FrameAxes {
                    arrows: false,
                    lines: Some(FrameLines {
                        reach: 4.0,
                        both_ways: false,
                    }),
                    ..default()
                },
            ))
            .id();
        app.update();
        let count = |app: &mut App| {
            let world = app.world_mut();
            world.query::<&AxisLine>().iter(world).count()
        };
        assert_eq!(count(&mut app), 3);
        assert_eq!(arrows(&mut app), 0);
        app.world_mut().entity_mut(body).remove::<FrameAxes>();
        app.update();
        assert_eq!(count(&mut app), 0);
    }

    /// The arrows keep their length on a scaled node.
    #[test]
    fn a_scaled_frame_keeps_its_arrow_length() {
        let mut app = app();
        app.world_mut().spawn((
            Transform::from_scale(Vec3::new(2.0, 4.0, 0.5)),
            FrameAxes::default(),
        ));
        app.update();
        app.update();
        let global = *app
            .world_mut()
            .query_filtered::<&GlobalTransform, With<AxesOf>>()
            .single(app.world())
            .expect("one holder");
        assert!(
            global.scale().abs_diff_eq(Vec3::splat(0.3), 1e-5),
            "{:?}",
            global.scale()
        );
    }
}
