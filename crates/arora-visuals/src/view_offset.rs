//! The host's view offset: a view that renders as part of a larger notional
//! viewport, so its subject stays centred in the part of the canvas the user
//! can see while a panel covers the rest.
//!
//! Bevy's [`SubCameraView`] is the analogue of three's `setViewOffset`. The
//! host supplies it through a [`ViewOffsetFeed`] read every frame, since a
//! panel sliding in changes it continuously; it is applied to every live
//! instance's camera.
//!
//! A camera with an [`ActiveArea`] is framed by that instead, and this leaves it
//! alone: an active area is the higher-level answer to the same question, and
//! two writers of one camera's sub-view would undo each other every frame.

use bevy::camera::{CameraUpdateSystems, SubCameraView};
use bevy::prelude::*;

use crate::active_area::{ActiveArea, FramingSystems};
use crate::instance::{EntityInstance, Superseded};

/// The host's view offset, read each frame. `None` renders the whole view.
#[derive(Resource)]
pub struct ViewOffsetFeed(pub Box<dyn Fn() -> Option<SubCameraView> + Send + Sync>);

impl ViewOffsetFeed {
    pub fn new(feed: impl Fn() -> Option<SubCameraView> + Send + Sync + 'static) -> Self {
        Self(Box::new(feed))
    }
}

pub struct ViewOffsetPlugin;

impl Plugin for ViewOffsetPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            apply_view_offset
                .in_set(FramingSystems)
                .before(CameraUpdateSystems),
        );
    }
}

/// Applies the host's view offset to each instance's camera, writing only a
/// change so an unchanged offset does not mark the camera changed.
fn apply_view_offset(
    feed: Option<Res<ViewOffsetFeed>>,
    instances: Query<&EntityInstance, Without<Superseded>>,
    mut cameras: Query<&mut Camera, Without<ActiveArea>>,
) {
    let Some(feed) = feed else {
        return;
    };
    let offset = (feed.0)();
    for instance in &instances {
        if let Ok(mut camera) = cameras.get_mut(instance.camera) {
            if camera.sub_camera_view != offset {
                camera.sub_camera_view = offset;
            }
        }
    }
}
