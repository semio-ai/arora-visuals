//! Showing a reference frame: three arrows along an entity's own X, Y and Z,
//! from its origin, so where a body or a joint's frame sits and how it is
//! turned can be seen.
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

use bevy::light::NotShadowCaster;
use bevy::prelude::*;

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
}

impl Default for FrameAxes {
    /// Studio's body axes: three tenths of a metre, in the gizmos' colours.
    fn default() -> Self {
        Self {
            size: GizmoSize::World(0.3),
            camera: None,
            axes: [Dir3::X, Dir3::Y, Dir3::Z],
            colours: AXIS_COLOURS,
        }
    }
}

/// The arrows showing an entity's axes, held under it; `0` is the entity.
#[derive(Component)]
struct AxesOf(Entity);

/// The directions and colours the arrows were built in, so a change
/// rebuilds them.
#[derive(Component, PartialEq)]
struct Built([Dir3; 3], [Color; 3]);

pub struct FrameAxesPlugin;

impl Plugin for FrameAxesPlugin {
    fn build(&self, app: &mut App) {
        add_handle_material(app);
        app.add_systems(
            PostUpdate,
            (build_axes, (size_axes, drop_axes))
                .chain()
                .after(TransformSystems::Propagate),
        );
    }
}

/// Builds the arrows for each entity newly showing its axes, or showing them
/// in new colours.
fn build_axes(
    mut commands: Commands,
    shown: Query<(Entity, &FrameAxes, Option<&Built>)>,
    holders: Query<(Entity, &AxesOf)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<HandleMaterial>>,
) {
    for (entity, axes, built) in &shown {
        let wanted = Built(axes.axes, axes.colours);
        if built == Some(&wanted) {
            continue;
        }
        for (holder, of) in &holders {
            if of.0 == entity {
                commands.entity(holder).despawn();
            }
        }
        commands.entity(entity).insert(wanted);

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

/// Takes the arrows away from an entity no longer showing its axes.
fn drop_axes(
    mut commands: Commands,
    mut removed: RemovedComponents<FrameAxes>,
    holders: Query<(Entity, &AxesOf)>,
) {
    for entity in removed.read() {
        if let Ok(mut built) = commands.get_entity(entity) {
            built.remove::<Built>();
        }
        for (holder, of) in &holders {
            if of.0 == entity {
                commands.entity(holder).despawn();
            }
        }
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
