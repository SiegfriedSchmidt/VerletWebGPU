const PI = 3.14159265359;

struct Circle {
    pos: vec2f,
    vel: vec2f
}

struct Global {
    circle_radius: f32
}

@group(0) @binding(1) var<uniform> time: f32;
@group(0) @binding(2) var<uniform> grid_res: f32;
@group(0) @binding(3) var<uniform> global: Global;
@group(0) @binding(4) var<storage, read_write> circles: array<Circle>;
@group(0) @binding(5) var<storage, read_write> grid: array<f32>;

fn hash(state: f32) -> f32 {
    var s = u32(state);
    s ^= 2747636419;
    s *= 2654435769;
    s ^= s >> 16;
    s *= 2654435769;
    s ^= s >> 16;
    s *= 2654435769;
    return f32(s) / 4294967295;
}

@compute @workgroup_size(64)
fn update_circles(@builtin(global_invocation_id) id: vec3u) {
//    circles[id.x].pos = vec2f(-1, -1);
}

@compute @workgroup_size(8)
fn solve_collisions(@builtin(global_invocation_id) pos: vec3u) {

}
