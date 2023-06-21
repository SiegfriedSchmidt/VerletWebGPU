struct Circle {
    pos: vec2f,
    vel: vec2f
}

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) center: vec2f
}

@group(0) @binding(0) var<storage> circles: array<Circle>;

@vertex
fn main(@location(0) pos: vec2f, @builtin(instance_index) i: u32) -> VertexOutput {
    let offset = circles[i].pos;
    return VertexOutput(vec4f(pos + offset, 0, 1), offset);
}