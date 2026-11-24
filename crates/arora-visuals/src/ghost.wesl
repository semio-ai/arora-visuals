// A ghost: an element drawn as an outline of light, Studio's
// `materials/ghost.tsx`.
//
// Its colour is the element's own washed halfway toward the wash colour —
// white unless the host chose another — unlit. Its opacity follows the rim:
// nearly clear where the surface faces the camera, rising steeply toward the
// silhouette, so the ghost reads by its outline and what stands behind it
// shows through. A ghost carrying an image — a screen —
// shows the image in the same wash at one opacity throughout, since a flat
// image has no rim to light. Either opacity is scaled by the element's own
// opacity (its colour's alpha) and by the ghost's, so an element faded solid
// is faded as a ghost too, and a host can fade the ghost itself.

import bevy_pbr::render::{forward_io::VertexOutput, mesh_view_bindings::view};

struct Ghost {
    colour: vec4<f32>,
    // What the colour is washed toward.
    wash: vec4<f32>,
    // How far toward the element's colour from the wash: a half is Studio's.
    tint: f32,
    // The rim falloff, as Studio has it: `pow(coefficient + n·v, power)`.
    power: f32,
    coefficient: f32,
    // One where the element carries an image, nought otherwise.
    image: f32,
    // How opaque the rim is against Studio's.
    rim: f32,
    // How opaque the whole ghost is.
    opacity: f32,
    _padding: vec2<f32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> ghost: Ghost;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var image_texture: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var image_sampler: sampler;

// An image's opacity on a ghost.
const IMAGE_OPACITY: f32 = 0.6;

// Below this opacity a depth-tested ghost draws nothing, so that only the
// surface it visibly draws writes depth: a nearly clear fragment, face-on to
// the camera, would otherwise hide what is drawn behind it after it.
const DEPTH_ALPHA: f32 = 0.05;

// The fragment's colour, discarding one a depth-tested ghost barely draws.
fn shown(colour: vec4<f32>) -> vec4<f32> {
    @if(GHOST_DEPTH_TESTED) {
        if colour.a < DEPTH_ALPHA {
            discard;
        }
    }
    return colour;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    var base = ghost.colour.rgb;
    let faded = ghost.colour.a * ghost.opacity;
    if ghost.image > 0.5 {
        var sampled = vec3(1.0);
        @if(VERTEX_UVS_A) {
            sampled = textureSample(image_texture, image_sampler, in.uv).rgb;
        }
        return shown(vec4(mix(ghost.wash.rgb, base * sampled, ghost.tint), IMAGE_OPACITY * faded));
    }
    // From the camera to the fragment, and the surface's outward normal: the
    // dot is -1 looking straight at the surface and 0 along its edge.
    let toward = normalize(in.world_position.xyz - view.world_position);
    let normal = normalize(in.world_normal);
    let rim = pow(max(ghost.coefficient + dot(normal, toward), 0.0), ghost.power);
    // Studio's mix runs past its end at the very rim, which the clamp keeps an
    // opacity.
    let alpha = clamp(mix(0.01, 0.25, rim) * ghost.rim, 0.0, 1.0) * faded;
    return shown(vec4(mix(ghost.wash.rgb, base, ghost.tint), alpha));
}
