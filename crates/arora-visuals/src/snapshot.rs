//! Headless offscreen rendering: draw a view to a texture without a window,
//! read the pixels back, save a PNG.
//!
//! This is how a Bevy view gets a regression test at all. **A frame is the
//! only artifact that answers the question a renderer is asked**, and every
//! cheaper signal is a proxy that can be right while the picture is wrong — a
//! mesh that spawns, passes culling and is counted among the visible still
//! draws nothing if its material has no shader. So the harness captures the
//! frame and leaves the comparison to the caller.
//!
//! No window and no event loop: `WinitPlugin` disabled, `ScheduleRunnerPlugin`
//! with manual `App::update`, a camera drawing into a `RenderTarget` image
//! carrying `COPY_SRC`, and `gpu_readback` delivering the pixels. The shape is
//! from ros-viz-rs `src/snapshot.rs`.
//!
//! Both waits here are bounded rather than open. A missing or unusable GPU
//! otherwise hangs the process with nothing to read, which in a test suite is
//! indistinguishable from slowness and costs far more than a prompt failure.

use std::time::Duration;

// Without anyhow's `Context`: Bevy's prelude carries a `ContextExt` offering the
// same `context`/`with_context`, so a bare call here is ambiguous. Add context
// through `anyhow::Error::new(e).context(..)`, which is inherent rather than a
// trait method and keeps the source chain the formatted form would flatten.
use anyhow::{bail, ensure, Result};
use bevy::app::{PluginsState, ScheduleRunnerPlugin};
use bevy::asset::{AssetMetaCheck, AssetPlugin};
use bevy::image::TextureFormatPixelInfo;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::render::renderer::RenderDevice;
use bevy::render::RenderPlugin;
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;
use image::RgbaImage;

use crate::target::OffscreenTarget;

/// Matches `TextureFormat::bevy_default()` on desktop, and keeps readback
/// bytes in plain RGBA order so a capture needs no channel swizzle.
pub const FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;

/// Bounded wait for the async GPU readback (~2 s at 60 fps) so a missing GPU
/// fails fast instead of hanging.
const MAX_READBACK_FRAMES: u32 = 120;

/// Latest raw (row-padded) frame from the GPU readback.
#[derive(Resource, Default)]
struct Frame(Option<Vec<u8>>);

/// Windowless app base: `DefaultPlugins` without a window or winit, driven by
/// manual updates. Inserts the offscreen target the view's cameras draw into.
pub struct SnapshotPlugin {
    pub width: u32,
    pub height: u32,
}

impl Plugin for SnapshotPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .set(RenderPlugin {
                    synchronous_pipeline_compilation: true,
                    ..default()
                })
                // An asset served from memory has no `.meta` file beside
                // it to probe for, and the probe is a per-asset round trip.
                .set(AssetPlugin {
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                })
                .disable::<WinitPlugin>()
                // The app owns logging (env_logger).
                .disable::<bevy::log::LogPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 60.0,
        )))
        .init_resource::<Frame>();

        // The render-target image the view camera draws into.
        let mut target = Image::new_target_texture(self.width, self.height, FORMAT, None);
        target.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        let handle = app.world_mut().resource_mut::<Assets<Image>>().add(target);
        app.insert_resource(OffscreenTarget(handle.clone()));
        app.world_mut().spawn(Readback::texture(handle)).observe(
            |event: On<ReadbackComplete>, mut frame: ResMut<Frame>| {
                frame.0 = Some(event.data.clone());
            },
        );
    }
}

/// Finish deferred plugin setup so manual `App::update` calls work. Idempotent.
pub fn ensure_ready(app: &mut App) {
    if app.plugins_state() != PluginsState::Cleaned {
        while app.plugins_state() == PluginsState::Adding {
            // The render plugin finishes on the task pools; on the web there
            // are none to tick, the loop spins until the pool's promise lands.
            #[cfg(not(target_arch = "wasm32"))]
            bevy::tasks::tick_global_task_pools_on_main_thread();
        }
        app.finish();
        app.cleanup();
    }
}

/// Updates until `ready` reports true, then a few settle frames, then captures
/// the next fresh frame.
///
/// `ready` is the caller's because only the caller knows what its scene has to
/// have done before a frame means anything — spawned, bound, morphed. A fixed
/// frame count in its place would be a guess that passes on a fast machine.
pub fn capture(
    app: &mut App,
    width: u32,
    height: u32,
    settle_frames: u32,
    mut ready: impl FnMut(&mut App) -> bool,
) -> Result<RgbaImage> {
    ensure_ready(app);

    const MAX_READY_FRAMES: u32 = 600;
    let mut was_ready = false;
    for _ in 0..MAX_READY_FRAMES {
        app.update();
        if ready(app) {
            was_ready = true;
            break;
        }
    }
    if !was_ready {
        bail!("scene did not become ready within {MAX_READY_FRAMES} updates");
    }

    for _ in 0..settle_frames {
        app.update();
    }

    app.world_mut().resource_mut::<Frame>().0 = None;
    for _ in 0..MAX_READBACK_FRAMES {
        app.update();
        if let Some(data) = app.world_mut().resource_mut::<Frame>().0.take() {
            return frame_to_image(&data, width, height);
        }
    }
    bail!("GPU readback did not deliver a frame within {MAX_READBACK_FRAMES} updates");
}

/// Strips wgpu's 256-byte row padding and builds the image.
fn frame_to_image(data: &[u8], width: u32, height: u32) -> Result<RgbaImage> {
    let pixel_size = FORMAT
        .pixel_size()
        .map_err(|e| anyhow::anyhow!("cannot get pixel size of {FORMAT:?}: {e:?}"))?;
    let unpadded_row = width as usize * pixel_size;
    let padded_row = RenderDevice::align_copy_bytes_per_row(unpadded_row);
    ensure!(
        data.len() == padded_row * height as usize,
        "unexpected readback size: got {} bytes for {width}x{height}",
        data.len(),
    );
    let mut pixels = Vec::with_capacity(unpadded_row * height as usize);
    for row in data.chunks_exact(padded_row) {
        pixels.extend_from_slice(&row[..unpadded_row]);
    }
    RgbaImage::from_raw(width, height, pixels)
        .ok_or_else(|| anyhow::anyhow!("readback buffer mismatch"))
}

/// Saves a captured frame as PNG.
pub fn save_png(img: &RgbaImage, path: &std::path::Path) -> Result<()> {
    img.save_with_format(path, image::ImageFormat::Png).map_err(|e| {
        anyhow::Error::new(e).context(format!("failed to save PNG to {}", path.display()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A caller deciding what to do about a failed save needs the cause, not a
    /// sentence containing it: whether the disk is full, the directory is gone
    /// or the path is unwritable are different situations with different
    /// answers. Formatting the error into a message reads the same and leaves
    /// nothing to branch on, so the chain is asserted rather than assumed.
    #[test]
    fn a_failed_save_keeps_the_image_error_it_failed_with() {
        let img = RgbaImage::new(1, 1);
        let unwritable = std::path::Path::new("/does/not/exist/frame.png");

        let error = save_png(&img, unwritable).expect_err("a missing directory cannot be written");

        assert!(
            error.to_string().contains("frame.png"),
            "the message should name the path: {error}"
        );
        let cause = error
            .downcast_ref::<image::ImageError>()
            .expect("the image error is preserved, not formatted away");
        assert!(
            matches!(cause, image::ImageError::IoError(_)),
            "a missing directory is an IO failure: {cause:?}"
        );
    }
}
