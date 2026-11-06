// A dashed guide line: one flat colour, drawn over whatever stands in front of
// it, in dashes counted outward from the line's origin.
//
// The mesh is a tube whose v runs from 0 at the origin to 1 at the far end,
// so every half of a line dashes the same way out from the shared origin and
// a line running both ways is symmetric about it. The same v fades the line
// out toward its end, so it trails off rather than stopping on a hard edge;
// one smoothstep per fragment, nothing more drawn.

import bevy_pbr::render::forward_io::VertexOutput;

struct AxisLine {
    colour: vec4<f32>,
    // How many dash periods fit along the half, and how much of each period
    // is drawn.
    dashes: f32,
    duty: f32,
    // Where along the half the fade begins, as a fraction of its length.
    fade_from: f32,
    _padding: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> line: AxisLine;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if fract(in.uv.y * line.dashes) > line.duty {
        discard;
    }
    let fade = 1.0 - smoothstep(line.fade_from, 1.0, in.uv.y);
    return vec4<f32>(line.colour.rgb, line.colour.a * fade);
}
