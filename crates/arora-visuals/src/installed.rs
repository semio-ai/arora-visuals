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
    use crate::drawn::DrawnPlugin;
    use crate::ghost::GhostPlugin;
    use crate::joint_gizmo::JointGizmoPlugin;
    use crate::outline::OutlinePlugin;
    use crate::overlay::OverlayPlugin;
    use crate::pick::PickPlugin;
    use crate::transform_gizmo::TransformGizmoPlugin;

    /// The plugins two consumers sharing an app would each add, added twice:
    /// Bevy panics on a unique plugin added again, and a plugin that built
    /// twice would run its systems twice a frame. With Bevy's transform and
    /// visibility systems beside them, and the frame's schedules built, so
    /// the order the plugins put themselves in among Bevy's is free of
    /// cycles.
    #[test]
    fn two_consumers_can_add_the_same_plugins() {
        let consumer = || {
            (
                ActiveAreaPlugin,
                FrameAxesPlugin,
                DrawnPlugin,
                GhostPlugin,
                JointGizmoPlugin,
                OutlinePlugin,
                OverlayPlugin,
                PickPlugin,
                TransformGizmoPlugin,
            )
        };
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            AssetPlugin::default(),
            TransformPlugin,
            bevy::camera::visibility::VisibilityPlugin,
        ))
        .add_plugins(consumer())
        .add_plugins(consumer());
        for label in [Update.intern(), PostUpdate.intern(), Last.intern()] {
            let world = app.world_mut();
            let Some(mut schedule) = world.resource_mut::<Schedules>().remove(label) else {
                continue;
            };
            schedule.initialize(world).expect("an ordered schedule");
            world.resource_mut::<Schedules>().insert(schedule);
        }
        assert!(app.is_plugin_added::<OutlinePlugin>());
    }
}
