//! A dashed guide line along one direction from an entity's origin: the
//! line a frame's axis or a held gizmo handle constrains to, reaching well
//! past whatever marks the axis itself.
//!
//! An [`AxisLine`] on an entity draws it, in the entity's own space. It runs
//! out along its direction, or both ways through the origin: both ways where
//! something else already says which way is positive — a frame's arrows, a
//! gizmo's handles — and one way where the line alone has to.
//!
//! **Its reach and its dashes take a size, as a gizmo does** ([`GizmoSize`]),
//! so a line can stay fixed in the world or the same on screen however far.
//! Its width does not: it is always [`WIDTH_PX`] across wherever a camera is
//! given to measure with, so a guide never swells into a bar up close or
//! thins to nothing far away. It is flat, fainter than a handle, drawn over
//! everything as the gizmos are, never picked and casts no shadow. It follows
//! the entity's position and turn but not its scale.

use bevy::asset::{embedded_asset, embedded_path, AssetPath};
use bevy::light::NotShadowCaster;
use bevy::mesh::{CylinderAnchor, CylinderMeshBuilder, MeshVertexBufferLayoutRef};
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, CompareFunction, RenderPipelineDescriptor, ShaderType,
    SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::transform_gizmo::{metres_per_pixel, unit_size, GizmoSize, TUBE};

/// A dashed line from this entity's origin ([module docs](self)).
#[derive(Component, Clone, Debug, PartialEq)]
pub struct AxisLine {
    /// Which way it runs, in the entity's own space.
    pub direction: Dir3,
    /// How far it reaches from the origin, in units of `size`.
    pub reach: f32,
    /// Whether it runs back through the origin as far again.
    pub both_ways: bool,
    /// The unit its reach and dashes are measured in. A size on screen is
    /// measured through `camera`, and shows nothing without one.
    pub size: GizmoSize,
    pub camera: Option<Entity>,
    pub colour: Color,
}

/// How wide a line is on screen, in logical pixels, when a camera is given.
pub const WIDTH_PX: f32 = 1.5;

/// One dash and the gap after it, in units of the line's size.
const DASH_PERIOD: f32 = 0.08;

/// How much of each period is dash.
const DUTY: f32 = 0.5;

/// How much fainter than its colour a line is drawn.
const OPACITY: f32 = 0.6;

#[derive(ShaderType, Clone, Debug, Default)]
pub struct AxisLineUniform {
    pub colour: Vec4,
    pub dashes: f32,
    pub duty: f32,
    pub _padding: Vec2,
}

/// A line's material: one flat colour in dashes, never hidden; see
/// `axis_line.wesl`.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug, Default)]
pub struct AxisLineMaterial {
    #[uniform(0)]
    pub line: AxisLineUniform,
}

impl Material for AxisLineMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(embedded_path!("axis_line.wesl")).with_source("embedded"),
        )
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // A thin tube seen from any side: no face of it is a back face.
        descriptor.primitive.cull_mode = None;
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_compare = Some(CompareFunction::Always);
            depth.depth_write_enabled = Some(false);
        }
        Ok(())
    }
}

/// Where lines are placed, in `PostUpdate` after transforms propagate. Anything
/// placing a line's parent by hand after propagation — a gizmo does — runs
/// before it, so a line never trails what it marks by a frame.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct AxisLineSystems;

/// One drawn half of a line: `1.0` out along its direction, `-1.0` back.
#[derive(Component)]
struct Half(f32);

/// What a line's halves were built for, so a change rebuilds them.
#[derive(Component, PartialEq)]
struct Built {
    both_ways: bool,
    material: Handle<AxisLineMaterial>,
}

/// The unit tube every half is drawn with: a radius of one, a length of one,
/// standing up from its base.
#[derive(Resource)]
struct Tube(Handle<Mesh>);

pub struct AxisLinePlugin;

impl Plugin for AxisLinePlugin {
    fn build(&self, app: &mut App) {
        if app.is_plugin_added::<MaterialPlugin<AxisLineMaterial>>() {
            return;
        }
        embedded_asset!(app, "axis_line.wesl");
        app.add_plugins(MaterialPlugin::<AxisLineMaterial>::default())
            .add_systems(Startup, make_tube)
            .add_systems(
                PostUpdate,
                (build_lines, place_lines)
                    .chain()
                    .in_set(AxisLineSystems)
                    .after(TransformSystems::Propagate),
            );
    }
}

fn make_tube(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    let tube = CylinderMeshBuilder::new(1.0, 1.0, 12)
        .without_caps()
        .anchor(CylinderAnchor::Bottom);
    commands.insert_resource(Tube(meshes.add(tube)));
}

/// Lines whose settings changed this frame, with what they were built as.
type ChangedLines<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static AxisLine,
        Option<&'static Built>,
        Option<&'static Children>,
    ),
    Changed<AxisLine>,
>;

/// Builds the halves of each line that is new or has changed which ways it
/// runs.
fn build_lines(
    mut commands: Commands,
    tube: Option<Res<Tube>>,
    lines: ChangedLines,
    halves: Query<(), With<Half>>,
    mut materials: ResMut<Assets<AxisLineMaterial>>,
) {
    let Some(tube) = tube else {
        return;
    };
    for (entity, line, built, children) in &lines {
        if built.is_some_and(|built| built.both_ways == line.both_ways) {
            continue;
        }
        for child in children.into_iter().flatten() {
            if halves.contains(*child) {
                commands.entity(*child).despawn();
            }
        }
        let material = materials.add(AxisLineMaterial::default());
        commands.entity(entity).insert(Built {
            both_ways: line.both_ways,
            material: material.clone(),
        });
        let signs: &[f32] = if line.both_ways { &[1.0, -1.0] } else { &[1.0] };
        for sign in signs {
            commands.spawn((
                Half(*sign),
                Mesh3d(tube.0.clone()),
                MeshMaterial3d(material.clone()),
                Transform::default(),
                Visibility::Hidden,
                NotShadowCaster,
                Pickable::IGNORE,
                ChildOf(entity),
            ));
        }
    }
}

/// Lays each line's halves along it at its reach and width, and sets its
/// dashes and colour.
fn place_lines(
    lines: Query<(&AxisLine, &GlobalTransform, &Built, &Children), Without<Half>>,
    cameras: Query<(&Camera, &GlobalTransform), Without<Half>>,
    mut halves: Query<(&Half, &mut Transform, &mut GlobalTransform, &mut Visibility)>,
    mut materials: ResMut<Assets<AxisLineMaterial>>,
) {
    for (line, at, built, children) in &lines {
        let origin = at.translation();
        let camera = line.camera.and_then(|camera| cameras.get(camera).ok());
        let unit = match (line.size, camera) {
            (GizmoSize::World(metres), _) => Some(metres),
            (size, Some(camera)) => unit_size(size, camera, origin),
            (GizmoSize::Pixels(_), None) => None,
        };
        let width = match camera {
            Some(camera) => metres_per_pixel(camera, origin).map(|metres| metres * WIDTH_PX),
            None => unit.map(|unit| 2.0 * TUBE[0] * unit),
        };
        let shown = match (unit, width) {
            (Some(unit), Some(width)) => Some((unit, width)),
            _ => None,
        };

        if shown.is_some() {
            let faint = line.colour.with_alpha(line.colour.alpha() * OPACITY);
            let wanted = AxisLineUniform {
                colour: LinearRgba::from(faint).to_vec4(),
                dashes: line.reach / DASH_PERIOD,
                duty: DUTY,
                _padding: Vec2::ZERO,
            };
            // Touched only on a change, since touching a material re-uploads it.
            let stale = materials.get(&built.material).is_some_and(|material| {
                material.line.colour != wanted.colour || material.line.dashes != wanted.dashes
            });
            if stale {
                if let Some(mut material) = materials.get_mut(&built.material) {
                    material.line = wanted;
                }
            }
        }

        // The line's own space may be scaled; the halves undo that, so the
        // line keeps its length and width.
        let scale = at.scale();
        let undo = 3.0 / (scale.x + scale.y + scale.z).max(f32::EPSILON);
        for child in children.iter() {
            let Ok((half, mut transform, mut global, mut visibility)) = halves.get_mut(child)
            else {
                continue;
            };
            let Some((unit, width)) = shown else {
                visibility.set_if_neq(Visibility::Hidden);
                continue;
            };
            visibility.set_if_neq(Visibility::Inherited);
            let along = line.direction.as_vec3() * half.0;
            *transform = Transform::from_rotation(Quat::from_rotation_arc(Vec3::Y, along))
                .with_scale(Vec3::new(width / 2.0, line.reach * unit, width / 2.0) * undo);
            // Placed by hand, after propagation, so the half stands where
            // its line is this frame.
            *global = at.mul_transform(*transform);
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
            .init_asset::<AxisLineMaterial>()
            .add_systems(Startup, make_tube)
            .add_systems(
                PostUpdate,
                (build_lines, place_lines)
                    .chain()
                    .after(TransformSystems::Propagate),
            );
        app
    }

    fn line(both_ways: bool) -> AxisLine {
        AxisLine {
            direction: Dir3::X,
            reach: 4.0,
            both_ways,
            size: GizmoSize::World(0.25),
            camera: None,
            colour: Color::WHITE,
        }
    }

    /// The far ends of a line's halves, in the world.
    fn ends(app: &mut App) -> Vec<Vec3> {
        let world = app.world_mut();
        world
            .query::<(&Half, &GlobalTransform)>()
            .iter(world)
            .map(|(_, at)| at.transform_point(Vec3::Y))
            .collect()
    }

    /// A one-way line reaches out along its direction only; a two-way one as
    /// far back again.
    #[test]
    fn a_line_reaches_one_way_or_both() {
        let mut app = app();
        let entity = app
            .world_mut()
            .spawn((Transform::from_xyz(1.0, 0.0, 0.0), line(false)))
            .id();
        app.update();
        app.update();
        let one = ends(&mut app);
        assert_eq!(one.len(), 1);
        assert!(
            one[0].abs_diff_eq(Vec3::new(2.0, 0.0, 0.0), 1e-5),
            "{one:?}"
        );

        app.world_mut().entity_mut(entity).insert(line(true));
        app.update();
        app.update();
        let mut both = ends(&mut app);
        both.sort_by(|a, b| a.x.total_cmp(&b.x));
        assert_eq!(both.len(), 2);
        assert!(
            both[0].abs_diff_eq(Vec3::new(0.0, 0.0, 0.0), 1e-5),
            "{both:?}"
        );
        assert!(
            both[1].abs_diff_eq(Vec3::new(2.0, 0.0, 0.0), 1e-5),
            "{both:?}"
        );
    }

    /// A scaled entity's line keeps its reach.
    #[test]
    fn a_scaled_line_keeps_its_reach() {
        let mut app = app();
        app.world_mut()
            .spawn((Transform::from_scale(Vec3::splat(3.0)), line(false)));
        app.update();
        app.update();
        let end = ends(&mut app)[0];
        assert!(end.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-5), "{end:?}");
    }
}
