struct Circle {
    pos: vec2f,
    last_pos: vec2f,
    accel: vec2f,
    color: vec3f,
    radius: f32
}

@group(0) @binding(0) var<storage> circle: array<Circle>;
@group(0) @binding(1) var<uniform> res: vec2f;

@fragment
fn main(@builtin(position) pos: vec4f, @location(0) @interpolate(flat) id: u32) -> @location(0) vec4f {
    let pos2 = vec2f(pos.x, res.y - pos.y);
    let dis = length(pos2 - circle[id].pos);
    if (dis > circle[id].radius) {
        discard;
    }
    var color = circle[id].color;
//    color *= (1 - 1 / (circle[id].radius - dis));
    return vec4f(color, 1);
}
