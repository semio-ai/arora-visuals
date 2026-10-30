// A gizmo handle: one flat colour, drawn over whatever stands in front of it.
//
// Flat because a handle is a control, not an object in the scene: shading
// would make its colour depend on the lights, and its colour is what says
// which axis it is.

import bevy_pbr::render::forward_io::VertexOutput;

struct Handle {
    colour: vec4<f32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> handle: Handle;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return handle.colour;
}
