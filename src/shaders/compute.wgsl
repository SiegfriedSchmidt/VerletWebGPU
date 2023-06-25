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
@group(0) @binding(1) var<uniform> time: i32;
@group(0) @binding(2) var<uniform> grid_res: vec3i;
@group(0) @binding(3) var<uniform> global: Global;
@group(0) @binding(4) var<storage, read_write> circle: array<Circle>;
@group(0) @binding(5) var<storage, read_write> grid: array<i32>;

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

fn get_cell(pos: vec2f) -> vec2i {
    return vec2i(floor(pos / (res / vec2f(grid_res.xy))));
}

fn get_pos(pos: vec2i) -> i32 {
    return (pos.x + pos.y * grid_res.x) * grid_res.z;
}

fn write_grid(i: u32) {
    let grid_pos = get_pos(get_cell(circle[i].pos));
    var grid_i = 2;
    if (grid[grid_pos] == time) {
        grid_i = grid[grid_pos + 1];
    } else {
        grid[grid_pos] = time;
    }
    grid[grid_pos + grid_i] = i32(i);
    grid[grid_pos + 1] = grid_i + 1;
}

fn solve_collision(i1: i32, i2: i32) {
    let collision_axis = circle[i1].pos - circle[i2].pos;
    let dist = length(collision_axis);
    let min_dist = circle[i1].radius + circle[i2].radius;
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

    apply_constaint(i);
    let displacement = circle[i].pos - circle[i].lpos;
    circle[i].lpos = circle[i].pos;
    circle[i].pos += displacement + circle[i].accel * global.dt * global.dt;
//    write_grid(i);
}

@compute @workgroup_size(8, 8)
fn find_collisions(@builtin(global_invocation_id) workgroup_pos: vec3u) {
    let pos = vec2i(workgroup_pos.xy);
    if (pos.x >= grid_res.x || pos.y >= grid_res.y) {
        return;
    }

    let grid_pos1 = get_pos(pos.xy);
    if (grid[grid_pos1] == time) {
        let x_left = max(0, pos.x - 1);
        let x_right = min(grid_res.x - 1, pos.x + 1);
        let y_left = max(0, pos.y - 1);
        let y_right = min(grid_res.y - 1, pos.y + 1);

        for (var x = x_left; x <= x_right; x++) {
            for (var y = y_left; y <= y_right; y++) {
                let grid_pos2 = get_pos(vec2i(x, y));
                if (grid[grid_pos2] == time) {
                    let c1 = grid[grid_pos1 + 1];
                    let c2 = grid[grid_pos2 + 1];
                    for (var i1 = 2; i1 < c1; i1++) {
                        for (var i2 = 2; i2 < c2; i2++) {
                            let id_1 = grid[grid_pos1 + i1];
                            let id_2 = grid[grid_pos2 + i2];
                            if (id_1 != id_2) {
                                solve_collision(id_1, id_2);
                            }
                        }
                    }
                }
            }
        }
    }
}
