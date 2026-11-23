//! When an instance is drawable: once the GPU has finished a frame drawing
//! every mesh visible to its camera, each with its pipeline compiled.
//!
//! **Created is not drawable, and nothing but a finished frame says
//! otherwise.** Creating a pipeline returns before the driver has compiled it
//! on some backends — a browser's WebGL compiles in its GPU process, when a
//! draw first uses the program. An instance whose meshes all exist, all pass
//! culling and are all counted among the visible can still draw nothing. That
//! is true of any renderer on any asset, which is why this is shared.
//!
//! So an indexed instance first draws *under* its target: its camera is
//! ordered below the [`ClearCamera`](crate::instance::ClearCamera), which
//! clears what it drew. Its pipelines compile and its meshes draw with the
//! very view they will be seen with, while the target still shows what it
//! showed before — a reload's previous instance, or the background. The
//! target never shows an asset with some of its meshes missing.
//!
//! The render world watches each warming camera's view. In the first frame
//! where every mesh visible to it has a compiled pipeline and is queued, it
//! asks the GPU queue to call back once that frame's work is done. On the
//! callback the instance is ready
//! ([`EntityInstance::ready`](crate::instance::EntityInstance::ready)): its
//! camera moves above the clear, and the next frame draws it in its place.
//!
//! **Nothing holds an instance back for good**, because a wait with no bound
//! is indistinguishable from slowness and strands whatever is waiting on it. A
//! pipeline that failed permanently never draws and the renderer logs it, so
//! its mesh counts as settled. Anything else — a mesh whose specialization
//! failed, a material that never loads, a camera that draws no frame, a GPU
//! that never reports the frame done — is logged after [`HELD_FRAMES`] frames,
//! and after [`GIVE_UP_FRAMES`] the instance is made ready as it draws. A
//! reload never keeps the previous instance, nor a page its readiness,
//! waiting forever.

use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Mutex;

use bevy::pbr::{PendingMeshMaterialQueues, SpecializedMaterialPipelineCache};
use bevy::prelude::*;
use bevy::render::render_resource::{CachedPipelineState, PipelineCache};
use bevy::render::renderer::RenderQueue;
use bevy::render::view::{ExtractedView, RenderVisibleEntities, RetainedViewEntity};
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};
use bevy::shader::ShaderCacheError;

use crate::instance::{EntityInstance, Superseded, ViewOf};

/// Frames an instance may stay unready before what holds it back is logged.
///
/// An instance draws within a few frames on a GPU, and a page under a
/// software GPU runs few frames while its pipelines compile — the browser
/// holds its frames back meanwhile — so two seconds of frames at 60 Hz
/// without it drawn means something is keeping it from drawing, rather than
/// that it is merely slow.
const HELD_FRAMES: u32 = 120;

/// Frames after which an instance still not drawn is made ready as it draws —
/// ten seconds at 60 Hz, five times [`HELD_FRAMES`]. Long enough for a slow
/// pipeline compile on a CPU renderer, short enough that one which can never
/// draw whole does not hold a reload or a page for good.
const GIVE_UP_FRAMES: u32 = 600;

/// On the camera of an indexed instance that is not ready yet: the camera
/// draws under the clear, and the render world watches its view.
#[derive(Component)]
pub struct Warming;

/// What the render world reports about a warming camera.
enum Report {
    /// The GPU has finished a frame drawing every mesh visible to it.
    Drawn,
    /// [`HELD_FRAMES`] frames went by without that.
    Held,
    /// [`GIVE_UP_FRAMES`] frames went by without that.
    GaveUp,
}

/// A report on a warming camera, with what held its instance back as of the
/// last frame it drew.
struct Message {
    camera: Entity,
    report: Report,
    /// The visible meshes whose pipeline was not compiled or which were not
    /// queued.
    meshes: Vec<Entity>,
    /// Whether the camera drew any frame.
    seen: bool,
    /// Whether the frame drawing the whole instance was submitted, the GPU's
    /// report on it awaited.
    submitted: bool,
}

/// An instance made ready, for a host to tell its page: whether it was seen
/// drawn whole, or given up on after [`GIVE_UP_FRAMES`] frames and made ready
/// as it draws. Readiness alone cannot say which, and a host reporting load
/// times or waiting on a first frame needs to.
#[derive(bevy::prelude::Message, Clone, Debug, PartialEq, Eq)]
pub struct InstanceReady {
    pub instance: Entity,
    /// Its [`EntityInstance::id`].
    pub id: String,
    /// Whether the GPU finished a frame drawing every mesh of it its camera
    /// sees; false when it was given up on.
    pub drawn: bool,
}

/// The render world's reports, read by the main world. Unbounded, as it
/// carries at most three messages per warming camera.
#[derive(Resource)]
pub struct Reports(Mutex<Receiver<Message>>);

/// The render world's end of [`Reports`].
#[derive(Resource)]
struct ReportSender(Sender<Message>);

/// One warming camera's view, as the render world follows it.
#[derive(Default)]
struct Watch {
    /// Frames since its instance was indexed.
    frames: u32,
    seen: bool,
    submitted: bool,
    warned: bool,
    /// Whether it was given up on.
    done: bool,
    /// The meshes that held its instance back in the last frame it drew.
    meshes: Vec<Entity>,
}

/// The warming cameras' views.
#[derive(Resource, Default)]
struct Watched(HashMap<RetainedViewEntity, Watch>);

pub struct DrawnPlugin;

impl Plugin for DrawnPlugin {
    /// Both consumers sharing an app may add it; it installs once.
    fn is_unique(&self) -> bool {
        false
    }

    fn build(&self, app: &mut App) {
        if !crate::installed::first_install::<Self>(app) {
            return;
        }
        let (sender, receiver) = channel();
        app.insert_resource(Reports(Mutex::new(receiver)))
            .add_message::<InstanceReady>();
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .insert_resource(ReportSender(sender))
                .init_resource::<Watched>()
                .add_systems(ExtractSchedule, extract_warming)
                // After the frame is submitted, so the GPU callback is about
                // this frame's work.
                .add_systems(Render, watch_drawing.in_set(RenderSystems::Cleanup));
        }
    }
}

/// Follow the warming cameras, counting each one's frames.
fn extract_warming(cameras: Extract<Query<Entity, With<Warming>>>, mut watched: ResMut<Watched>) {
    let views: Vec<RetainedViewEntity> = cameras
        .iter()
        .map(|camera| RetainedViewEntity::new(camera.into(), None, 0))
        .collect();
    watched.0.retain(|view, _| views.contains(view));
    for view in views {
        watched.0.entry(view).or_default().frames += 1;
    }
}

/// For each warming view whose every visible mesh was drawn this frame,
/// have the GPU report when it has finished the frame; report a view held
/// back too long.
fn watch_drawing(
    views: Query<(&ExtractedView, &RenderVisibleEntities)>,
    mut watched: ResMut<Watched>,
    specialized: Res<SpecializedMaterialPipelineCache>,
    pending: Res<PendingMeshMaterialQueues>,
    pipelines: Res<PipelineCache>,
    queue: Res<RenderQueue>,
    sender: Res<ReportSender>,
) {
    for (view, visible) in &views {
        let id = view.retained_view_entity;
        let Some(watch) = watched.0.get_mut(&id) else {
            continue;
        };
        if watch.submitted || watch.done {
            continue;
        }
        watch.seen = true;
        let view_pipelines = specialized.get(&id);
        let pending = pending.get(&id).map(|queues| &queues.current_frame);
        // A view with no mesh visible has nothing to wait for.
        watch.meshes = visible
            .get::<Mesh3d>()
            .into_iter()
            .flat_map(|meshes| meshes.iter_visible())
            .filter(|(entity, main)| {
                let queued = !pending.is_some_and(|pending| pending.contains(&(**entity, **main)));
                let settled =
                    view_pipelines
                        .and_then(|ids| ids.get(*main))
                        .is_some_and(|pipeline| {
                            settled(pipelines.get_render_pipeline_state(*pipeline))
                        });
                !(queued && settled)
            })
            .map(|(_, main)| main.id())
            .collect();
        if !watch.meshes.is_empty() {
            continue;
        }
        watch.submitted = true;
        let sender = sender.0.clone();
        let camera = id.main_entity.id();
        queue.on_submitted_work_done(move || {
            // The app may be gone by the time the GPU finishes.
            let _ = sender.send(Message {
                camera,
                report: Report::Drawn,
                meshes: Vec::new(),
                seen: true,
                submitted: true,
            });
        });
    }
    for (id, watch) in &mut watched.0 {
        let report = if watch.done {
            continue;
        } else if watch.frames >= GIVE_UP_FRAMES {
            watch.done = true;
            Report::GaveUp
        } else if watch.frames >= HELD_FRAMES && !watch.warned {
            watch.warned = true;
            Report::Held
        } else {
            continue;
        };
        let _ = sender.0.send(Message {
            camera: id.main_entity.id(),
            report,
            meshes: watch.meshes.clone(),
            seen: watch.seen,
            submitted: watch.submitted,
        });
    }
}

/// Whether a pipeline is as it will stay: compiled, or failed for good. A
/// pipeline waiting on a shader to load fails and is queued again, so that
/// failure is not final.
fn settled(state: &CachedPipelineState) -> bool {
    match state {
        CachedPipelineState::Ok(_) => true,
        CachedPipelineState::Err(
            ShaderCacheError::ShaderNotLoaded(_) | ShaderCacheError::ShaderImportNotYetAvailable,
        ) => false,
        CachedPipelineState::Err(_) => true,
        CachedPipelineState::Queued | CachedPipelineState::Creating(_) => false,
    }
}

/// What the render world said of a superseded instance's camera: whether it
/// was seen drawn. Kept until the instance is current again — its successor
/// failed — or goes, since the render world reports a camera once and will
/// not say it again.
#[derive(Component, Clone, Copy, Debug)]
pub struct Verdict(bool);

/// Make ready the instances the render world reports drawn, or given up on:
/// each one's camera moves above the clear, where it draws in its place from
/// the next frame on, and an [`InstanceReady`] says which. An instance
/// superseded meanwhile stays under the clear, keeping what was said of it:
/// retired, it never draws; made current again, it is made ready by that.
#[allow(clippy::too_many_arguments)]
pub fn ready_drawn_instances(
    reports: Res<Reports>,
    mut cameras: Query<(Entity, &ViewOf, &mut Camera, Option<&Verdict>), With<Warming>>,
    mut instances: Query<(&mut EntityInstance, Has<Superseded>)>,
    names: Query<&Name>,
    parents: Query<&ChildOf>,
    mut ready: MessageWriter<InstanceReady>,
    mut commands: Commands,
) {
    let Ok(reports) = reports.0.lock() else {
        return;
    };
    // Cameras made ready in this run. `Warming` comes off through commands,
    // which land only after it, so a second terminal report for the same
    // camera — the GPU's drawn and the timeout's give-up can both arrive
    // before one drain — would still find it warming. The first says what
    // the instance became; the rest are about a transition already made.
    let mut made_ready = std::collections::HashSet::new();
    // A superseded instance made current again, by what was said of it.
    for (camera, &ViewOf(entity), mut placed, verdict) in &mut cameras {
        let Some(&Verdict(drawn)) = verdict else {
            continue;
        };
        let Ok((mut instance, false)) = instances.get_mut(entity) else {
            continue;
        };
        make_ready(
            camera,
            entity,
            &mut placed,
            &mut instance,
            drawn,
            &mut ready,
            &mut commands,
        );
        made_ready.insert(camera);
    }
    for message in reports.try_iter() {
        if made_ready.contains(&message.camera) {
            continue;
        }
        let Ok((_, &ViewOf(instance), mut placed, verdict)) = cameras.get_mut(message.camera)
        else {
            continue;
        };
        // Already said: the render world's first word stands.
        if verdict.is_some() {
            continue;
        }
        let entity = instance;
        let Ok((mut instance, superseded)) = instances.get_mut(entity) else {
            continue;
        };
        let drawn = match message.report {
            Report::Drawn => true,
            Report::Held => {
                log::warn!(
                    "{}: not drawn after {HELD_FRAMES} frames: {}",
                    instance.id,
                    holding(&message, &names, &parents)
                );
                continue;
            }
            Report::GaveUp => {
                log::warn!(
                    "{}: ready without being drawn whole after {GIVE_UP_FRAMES} frames: {}",
                    instance.id,
                    holding(&message, &names, &parents)
                );
                false
            }
        };
        made_ready.insert(message.camera);
        if superseded {
            commands.entity(message.camera).insert(Verdict(drawn));
            continue;
        }
        make_ready(
            message.camera,
            entity,
            &mut placed,
            &mut instance,
            drawn,
            &mut ready,
            &mut commands,
        );
    }
}

/// Moves `instance`'s camera above the clear and says it is ready.
fn make_ready(
    camera: Entity,
    entity: Entity,
    placed: &mut Camera,
    instance: &mut EntityInstance,
    drawn: bool,
    ready: &mut MessageWriter<InstanceReady>,
    commands: &mut Commands,
) {
    placed.order = instance.draw_order;
    instance.ready = true;
    commands.entity(camera).remove::<(Warming, Verdict)>();
    ready.write(InstanceReady {
        instance: entity,
        id: instance.id.clone(),
        drawn,
    });
    log::info!("{}: ready", instance.id);
}

/// What kept an instance from being drawn, in words: the meshes, by the name
/// of the nearest named node, or the camera or the GPU.
fn holding(message: &Message, names: &Query<&Name>, parents: &Query<&ChildOf>) -> String {
    if !message.seen {
        return "its camera drew no frame".into();
    }
    if message.submitted {
        return "the GPU did not report the frame drawing it done".into();
    }
    let named = |mesh: Entity| {
        std::iter::successors(Some(mesh), |entity| {
            parents.get(*entity).ok().map(ChildOf::parent)
        })
        .find_map(|entity| names.get(entity).ok())
        .map_or_else(|| format!("{mesh}"), |name| name.as_str().to_string())
    };
    let meshes: Vec<String> = message.meshes.iter().map(|mesh| named(*mesh)).collect();
    format!(
        "meshes without a compiled pipeline or not queued: {}",
        meshes.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binding::SceneBindings;

    /// An instance made ready says whether it was seen drawn or given up on,
    /// once, and a superseded one says nothing. Two terminal reports for one
    /// camera in one frame — drawn, and the timeout racing it — make it
    /// ready once, as the first said.
    #[test]
    fn a_ready_instance_says_whether_it_was_drawn() {
        let mut app = App::new();
        let (sender, receiver) = channel();
        app.add_plugins(MinimalPlugins)
            .insert_resource(Reports(Mutex::new(receiver)))
            .add_message::<InstanceReady>()
            .add_systems(Update, ready_drawn_instances);
        let mut spawn = |id: &str, superseded: bool| {
            let world = app.world_mut();
            let camera = world.spawn((Camera::default(), Warming)).id();
            let mut instance = world.spawn(EntityInstance {
                id: id.into(),
                asset_path: String::new(),
                meta: Default::default(),
                bindings: SceneBindings::default(),
                camera,
                draw_order: 0,
                ready: false,
            });
            if superseded {
                instance.insert(Superseded);
            }
            let instance = instance.id();
            world.entity_mut(camera).insert(ViewOf(instance));
            camera
        };
        let drawn = spawn("drawn", false);
        let gave_up = spawn("gave-up", false);
        let gone = spawn("superseded", true);
        for (camera, report) in [
            (drawn, Report::Drawn),
            (gave_up, Report::GaveUp),
            (gone, Report::Drawn),
            (drawn, Report::GaveUp),
        ] {
            sender
                .send(Message {
                    camera,
                    report,
                    meshes: Vec::new(),
                    seen: true,
                    submitted: true,
                })
                .unwrap();
        }
        app.update();

        let messages = app.world().resource::<Messages<InstanceReady>>();
        let said: Vec<(String, bool)> = messages
            .iter_current_update_messages()
            .map(|ready| (ready.id.clone(), ready.drawn))
            .collect();
        assert_eq!(
            said,
            vec![("drawn".to_string(), true), ("gave-up".to_string(), false)]
        );

        // Its successor failed: the superseded instance is current again,
        // and ready by what was said of it while it was not.
        // A give-up racing the drawn report lands a frame later; the first
        // word stands.
        sender
            .send(Message {
                camera: gone,
                report: Report::GaveUp,
                meshes: Vec::new(),
                seen: true,
                submitted: true,
            })
            .unwrap();
        app.update();
        let instance = app.world().get::<ViewOf>(gone).unwrap().0;
        app.world_mut().entity_mut(instance).remove::<Superseded>();
        let mut cursor = app
            .world()
            .resource::<Messages<InstanceReady>>()
            .get_cursor_current();
        app.update();
        let messages = app.world().resource::<Messages<InstanceReady>>();
        let said: Vec<(String, bool)> = cursor
            .read(messages)
            .map(|ready| (ready.id.clone(), ready.drawn))
            .collect();
        assert_eq!(said, vec![("superseded".to_string(), true)]);
        assert!(app.world().get::<EntityInstance>(instance).unwrap().ready);
        assert!(app.world().get::<Warming>(gone).is_none());
    }
}
