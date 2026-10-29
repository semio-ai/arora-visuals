//! Where a view draws, when it does not draw to a window.

use bevy_asset::Handle;
use bevy_ecs::resource::Resource;
use bevy_image::Image;

/// When present, every view camera renders into this image instead of a
/// window.
///
/// One resource rather than one per camera: the cameras sharing it are views
/// of the same scene, and a host that wanted them apart would be asking for
/// separate scenes. It is what the headless harness sets to capture a frame,
/// and what a host sets to put a view somewhere other than the screen — a
/// texture on a mesh, for one.
#[derive(Resource, Clone)]
pub struct OffscreenTarget(pub Handle<Image>);
