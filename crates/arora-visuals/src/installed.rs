//! Plugins that two consumers can both add.
//!
//! A robot scene and the face on its screen can share one Bevy app, and each
//! adds the plugins of this crate it uses. Bevy panics when a unique plugin
//! is added twice, so every plugin here says it is not unique and installs
//! itself once, whoever adds it first.

use std::marker::PhantomData;

use bevy::prelude::*;

/// Marks plugin `P` installed. Private to this crate, so installing `P` is the
/// only way to create it.
#[derive(Resource)]
struct Installed<P: Send + Sync + 'static>(PhantomData<fn() -> P>);

/// Whether this is the first time `P` is installed in `app`, marking it
/// installed if so. A plugin returns from `build` at once when it is not.
pub(crate) fn first_install<P: Send + Sync + 'static>(app: &mut App) -> bool {
    if app.world().contains_resource::<Installed<P>>() {
        return false;
    }
    app.insert_resource(Installed::<P>(PhantomData));
    true
}

#[cfg(test)]
mod tests {
    use bevy::ecs::schedule::ScheduleLabel;
    use bevy::prelude::*;

    use crate::active_area::ActiveAreaPlugin;
    use crate::axes::FrameAxesPlugin;
    use crate::control::ControlsPlugin;
    use crate::drawn::DrawnPlugin;
    use crate::ghost::GhostPlugin;
    use crate::joint_gizmo::JointGizmoPlugin;
    use crate::outline::OutlinePlugin;
    use crate::overlay::OverlayPlugin;
    use crate::pick::PickPlugin;
    use crate::transform_gizmo::TransformGizmoPlugin;

    /// An app with the plugins one consumer adds, `times` over, beside
    /// Bevy's transform, camera and visibility systems, its frame's schedules
    /// built.
    fn with_plugins(times: usize) -> App {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            AssetPlugin::default(),
            TransformPlugin,
            bevy::camera::CameraPlugin,
        ));
        for _ in 0..times {
            app.add_plugins((
                ActiveAreaPlugin,
                ControlsPlugin,
                FrameAxesPlugin,
                DrawnPlugin,
                GhostPlugin,
                JointGizmoPlugin,
                OutlinePlugin,
                OverlayPlugin,
                PickPlugin,
                TransformGizmoPlugin,
            ));
            #[cfg(feature = "camera_input")]
            app.add_plugins(crate::camera_input::CameraInputPlugin);
            #[cfg(feature = "view_gizmo")]
            app.add_plugins(crate::view_gizmo::ViewGizmoPlugin);
        }
        for label in [Update.intern(), PostUpdate.intern(), Last.intern()] {
            let world = app.world_mut();
            let Some(mut schedule) = world.resource_mut::<Schedules>().remove(label) else {
                continue;
            };
            schedule.initialize(world).expect("an ordered schedule");
            world.resource_mut::<Schedules>().insert(schedule);
        }
        app
    }

    /// What an app runs: the systems in each of its schedules, and its
    /// observers.
    fn runs(app: &mut App) -> (Vec<usize>, usize) {
        let systems = [Update.intern(), PostUpdate.intern(), Last.intern()]
            .map(|label| {
                app.world()
                    .resource::<Schedules>()
                    .get(label)
                    .map_or(0, |schedule| schedule.systems_len())
            })
            .to_vec();
        let observers = app
            .world_mut()
            .query::<&bevy::ecs::observer::Observer>()
            .iter(app.world())
            .count();
        (systems, observers)
    }

    /// Two consumers sharing an app each add the plugins they use. Bevy
    /// panics on a unique plugin added again, and a plugin that built twice
    /// would run its systems and observers twice — two picks for one click.
    /// Added twice, the app runs exactly what it runs with them added once,
    /// and the order they put themselves in among Bevy's camera and
    /// visibility systems is free of cycles.
    #[test]
    fn two_consumers_can_add_the_same_plugins() {
        let once = runs(&mut with_plugins(1));
        let twice = runs(&mut with_plugins(2));
        assert_eq!(twice, once);
        assert!(once.0.iter().sum::<usize>() > 0);
    }
}
