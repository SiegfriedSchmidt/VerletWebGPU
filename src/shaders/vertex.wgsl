struct Circle {
    pos: vec2f,
    last_pos: vec2f,
    accel: vec2f,
    color: vec3f,
    radius: f32
}

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(flat) id: u32
}

@group(0) @binding(0) var<storage> circle: array<Circle>;
@group(0) @binding(1) var<uniform> res: vec2f;

@vertex
fn main(@location(0) pos: vec2f, @builtin(instance_index) id: u32) -> VertexOutput {
    let offset = circle[id].pos / vec2f(res) * 2 - 1;
    return VertexOutput(vec4f(pos / (res / circle[id].radius) * 2 + offset, 0, 1), id);
}