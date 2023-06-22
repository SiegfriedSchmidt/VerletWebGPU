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

fn get_cell(pos: vec2f) -> vec2u {
    return vec2u(floor(pos / (res / vec2f(grid_res.xy))));
}

fn get_pos(pos: vec2u) -> u32 {
    return (pos.x + pos.y * grid_res.x) * grid_res.z;
}

fn write_grid(i: u32) {
    let grid_pos = get_pos(get_cell(circle[i].pos));
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

fn solve_one_collision(i1: u32, i2: u32) {
    let collision_axis = circle[i1].pos - circle[i2].pos;
    let dist = length(collision_axis);
    let min_dist = circle[i1].radius + circle[i2].radius;
    circle[i1].color = vec3f(1, 1, 1);
    if (dist < min_dist) {

        let v = collision_axis / dist;
        let delta_v = v * ((min_dist - dist) * 0.5);
        circle[i1].pos += delta_v;
        circle[i2].pos -= delta_v;
    }
}

@compute @workgroup_size(64)
fn update_circles(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= u32(global.circleCount)) {
        return;
    }
    let i = id.x;

    let displacement = circle[i].pos - circle[i].lpos;
    circle[i].lpos = circle[i].pos;
    apply_constaint(i);
    circle[i].pos += displacement + circle[i].accel * global.dt * global.dt;
    write_grid(i);
}

@compute @workgroup_size(8, 8)
fn solve_collisions(@builtin(global_invocation_id) pos: vec3u) {
    if (pos.x >= grid_res.x - 1 || pos.x < 1 || pos.y >= grid_res.y - 1 || pos.y < 1) {
        return;
    }

    let grid_pos1 = get_pos(pos.xy);
    if (grid[grid_pos1] == time) {
        for (var x: u32 = pos.x - 1; x <= pos.x + 1; x++) {
            for (var y: u32 = pos.y - 1; y <= pos.y + 1; y++) {
                let grid_pos2 = get_pos(vec2u(x, y));
                if (grid[grid_pos2] == time) {
                    let c1 = grid[grid_pos1 + 1] + 2;
                    let c2 = grid[grid_pos2 + 1] + 2;
                    for (var i1: u32 = 2; i1 < c1; i1++) {
                        for (var i2: u32 = 2; i2 < c2; i2++) {
                            let id_1 = grid[grid_pos1 + i1];
                            let id_2 = grid[grid_pos2 + i2];
                            if (id_1 != id_2) {
                                solve_one_collision(id_1, id_2);
                            }
                        }
                    }
                }
            }
        }
    }
}
