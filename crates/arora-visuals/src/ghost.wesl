// A ghost: an element drawn as an outline of light, Studio's
// `materials/ghost.tsx`.
//
// Its colour is the element's own washed halfway toward the wash colour —
// white unless the host chose another — unlit. Its opacity follows the rim:
// nearly clear where the surface faces the camera, rising steeply toward the
// silhouette, so the ghost reads by its outline and what stands behind it
// shows through. A ghost carrying an image — a screen —
// shows the image in the same wash at one opacity throughout, since a flat
// image has no rim to light.

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
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> ghost: Ghost;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var image_texture: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var image_sampler: sampler;

// An image's opacity on a ghost.
const IMAGE_OPACITY: f32 = 0.6;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    var base = ghost.colour.rgb;
    if ghost.image > 0.5 {
        var sampled = vec3(1.0);
        @if(VERTEX_UVS_A) {
            sampled = textureSample(image_texture, image_sampler, in.uv).rgb;
        }
        return vec4(mix(ghost.wash.rgb, base * sampled, ghost.tint), IMAGE_OPACITY);
    }
    // From the camera to the fragment, and the surface's outward normal: the
    // dot is -1 looking straight at the surface and 0 along its edge.
    let toward = normalize(in.world_position.xyz - view.world_position);
    let normal = normalize(in.world_normal);
    let rim = pow(max(ghost.coefficient + dot(normal, toward), 0.0), ghost.power);
    // Studio's mix runs past its end at the very rim, which the clamp keeps an
    // opacity.
    let alpha = clamp(mix(0.01, 0.25, rim), 0.0, 1.0);
    return vec4(mix(ghost.wash.rgb, base, ghost.tint), alpha);
}
