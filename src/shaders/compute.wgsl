const PI = 3.14159265359;

struct Circle {
    pos: vec2f,
    lpos: vec2f,
    accel: vec2f,
    color: vec3f,
    radius: f32
}

struct Global {
    circleCount: f32,
    dt: f32,
}

@group(0) @binding(0) var<uniform> res: vec2f;
@group(0) @binding(1) var<uniform> time: u32;
@group(0) @binding(2) var<uniform> grid_res: vec3u;
@group(0) @binding(3) var<uniform> global: Global;
@group(0) @binding(4) var<storage, read_write> circle: array<Circle>;
@group(0) @binding(5) var<storage, read_write> grid: array<u32>;

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

fn apply_constaint(i: u32) {
    if (circle[i].pos.y > res.y - circle[i].radius) {
        circle[i].pos.y = res.y - circle[i].radius;
    }
    if (circle[i].pos.y < circle[i].radius) {
        circle[i].pos.y = circle[i].radius;
    }
    if (circle[i].pos.x > res.x - circle[i].radius) {
        circle[i].pos.x = res.x - circle[i].radius;
    }
    if (circle[i].pos.x < circle[i].radius) {
        circle[i].pos.x = circle[i].radius;
    }
}

fn get_pos(pos: vec2f) -> u32 {
    return (u32(pos[0]) + u32(pos[1]) * grid_res.y) * grid_res.z;
}

fn write_grid(i: u32) {
    let grid_pos = get_pos(circle[i].pos);
    var grid_i: u32 = 0;
    if (grid[grid_pos] == time) {
        grid_i = grid[grid_pos + 1];
        grid[grid_pos + 1] += 1;
    } else {
        grid[grid_pos] = time;
        grid[grid_pos + 1] = 1;
    }
    grid[grid_i + 2] = i;
}

@compute @workgroup_size(64)
fn update_circles(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= u32(global.circleCount)) {
        return;
    }
    let i = id.x;

    let displacement = circle[i].pos - circle[i].lpos;
    circle[i].lpos = circle[i].pos;
    circle[i].pos += displacement + circle[i].accel * global.dt * global.dt;
    apply_constaint(i);
    write_grid(i);
}

@compute @workgroup_size(8, 8)
fn solve_collisions(@builtin(global_invocation_id) pos: vec3u) {

}
