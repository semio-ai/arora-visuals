// A dashed guide line: one flat colour, drawn over whatever stands in front of
// it, in dashes counted outward from the line's origin.
//
// The mesh is a tube whose v runs from 0 at the origin to 1 at the far end,
// so every half of a line dashes the same way out from the shared origin and
// a line running both ways is symmetric about it.

import bevy_pbr::render::forward_io::VertexOutput;

struct AxisLine {
    colour: vec4<f32>,
    // How many dash periods fit along the half, and how much of each period
    // is drawn.
    dashes: f32,
    duty: f32,
    _padding: vec2<f32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> line: AxisLine;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if fract(in.uv.y * line.dashes) > line.duty {
        discard;
    }
    return line.colour;
}
