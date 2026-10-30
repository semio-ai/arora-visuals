// The selection outline, composited over the view.
//
// The mask holds the selection alone, drawn from the view camera's own eye.
// A fragment the selection covers takes only the fill, if there is one — the
// line sits outside it. A fragment within the outline's width of it takes the
// outline colour, weighted by how much of the selection that reach finds, so
// the line's outer edge is soft rather than stepped.

import bevy_pbr::render::{forward_io::VertexOutput, mesh_view_bindings::view};

struct Outline {
    colour: vec4<f32>,
    // In physical pixels.
    width: f32,
    // How strongly the selection itself is filled, from not at all to solid.
    fill: f32,
    _padding: vec2<f32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> outline: Outline;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var mask: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var mask_sampler: sampler;

const RINGS: u32 = 3u;
const STEPS: u32 = 16u;
const TAU: f32 = 6.28318530718;

fn coverage(uv: vec2<f32>) -> f32 {
    // An explicit level, because a sample inside a loop has no derivatives to
    // choose one from.
    return textureSampleLevel(mask, mask_sampler, uv, 0.0).a;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = (in.position.xy - view.viewport.xy) / view.viewport.zw;
    let here = coverage(uv);
    if here > 0.5 {
        // Inside the selection: the fill, if there is one, and never the line.
        if outline.fill <= 0.0 {
            discard;
        }
        return vec4(outline.colour.rgb, outline.colour.a * outline.fill * here);
    }
    let texel = 1.0 / vec2<f32>(textureDimensions(mask));
    var reach = 0.0;
    for (var ring = 1u; ring <= RINGS; ring++) {
        let radius = outline.width * f32(ring) / f32(RINGS);
        for (var step = 0u; step < STEPS; step++) {
            let angle = TAU * f32(step) / f32(STEPS);
            let offset = vec2(cos(angle), sin(angle)) * radius * texel;
            reach = max(reach, coverage(uv + offset));
        }
    }
    if reach <= 0.0 {
        discard;
    }
    return vec4(outline.colour.rgb, outline.colour.a * reach);
}
