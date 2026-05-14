// heal_jacobi.wgsl — One Jacobi iteration of the Laplace solve.
// Masked pixels are updated to the average of their 4 neighbours.
// Non-masked pixels copy their value unchanged (preserving Dirichlet BCs).
// Ping-pong: bind diff_a as diff_in and diff_b as diff_out, then swap next iteration.

struct HealParams {
    dest_center:  vec2<i32>,
    src_center:   vec2<i32>,
    radius_px:    i32,
    _pad0:        i32,
    bbox_origin:  vec2<i32>,
    bbox_size:    vec2<i32>,
    global_offset:vec2<i32>,
    full_dims:    vec2<i32>,
    _pad1:        vec2<i32>,
}

@group(0) @binding(0) var<storage, read>       diff_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write>  diff_out: array<f32>;
@group(0) @binding(2) var<storage, read>       mask_buf: array<f32>;
@group(0) @binding(3) var<uniform>             params:   HealParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = i32(gid.x);
    let by = i32(gid.y);
    let bw = params.bbox_size.x;
    let bh = params.bbox_size.y;
    if (bx >= bw || by >= bh) { return; }

    let idx      = (by * bw + bx) * 4;
    let mask_idx =  by * bw + bx;

    if (mask_buf[mask_idx] == 0.0) {
        // Boundary pixel — preserve Dirichlet BC unchanged
        diff_out[idx + 0] = diff_in[idx + 0];
        diff_out[idx + 1] = diff_in[idx + 1];
        diff_out[idx + 2] = diff_in[idx + 2];
        diff_out[idx + 3] = 0.0;
        return;
    }

    var sum   = vec3<f32>(0.0);
    var count = 0.0;

    if (bx > 0)      { let n = (by * bw + (bx - 1)) * 4; sum += vec3<f32>(diff_in[n], diff_in[n+1], diff_in[n+2]); count += 1.0; }
    if (bx < bw - 1) { let n = (by * bw + (bx + 1)) * 4; sum += vec3<f32>(diff_in[n], diff_in[n+1], diff_in[n+2]); count += 1.0; }
    if (by > 0)      { let n = ((by - 1) * bw + bx) * 4; sum += vec3<f32>(diff_in[n], diff_in[n+1], diff_in[n+2]); count += 1.0; }
    if (by < bh - 1) { let n = ((by + 1) * bw + bx) * 4; sum += vec3<f32>(diff_in[n], diff_in[n+1], diff_in[n+2]); count += 1.0; }

    let avg = sum / count;
    diff_out[idx + 0] = avg.x;
    diff_out[idx + 1] = avg.y;
    diff_out[idx + 2] = avg.z;
    diff_out[idx + 3] = 0.0;
}
