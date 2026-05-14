// heal_init.wgsl — For one spot's bounding box: computes diff = dest_pixel - src_pixel
// and writes a circular mask.  Non-masked pixels retain their diff values as Dirichlet
// boundary conditions for the Jacobi solver.

struct HealParams {
    dest_center:  vec2<i32>,  // full-image pixel coords of dest centre
    src_center:   vec2<i32>,  // full-image pixel coords of source centre
    radius_px:    i32,
    _pad0:        i32,
    bbox_origin:  vec2<i32>,  // top-left corner of bounding box in full-image pixels
    bbox_size:    vec2<i32>,  // width × height of bounding box
    global_offset:vec2<i32>,  // tile offset (0,0 for non-tiled renders)
    full_dims:    vec2<i32>,  // full image width × height
    _pad1:        vec2<i32>,
}

@group(0) @binding(0) var  input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> diff_buf:  array<f32>;
@group(0) @binding(2) var<storage, read_write> mask_buf:  array<f32>;
@group(0) @binding(3) var<uniform>             params:    HealParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = i32(gid.x);
    let by = i32(gid.y);
    if (bx >= params.bbox_size.x || by >= params.bbox_size.y) { return; }

    let buf_idx  = (by * params.bbox_size.x + bx) * 4;
    let mask_idx =  by * params.bbox_size.x + bx;
    let idims    = vec2<i32>(textureDimensions(input_tex));

    // Full-image dest and source pixel coordinates
    let dest_full = params.bbox_origin + vec2<i32>(bx, by);
    let delta     = vec2<i32>(bx, by) - (params.dest_center - params.bbox_origin);
    let src_full  = params.src_center + delta;

    // Tile-local coordinates (clamped to texture)
    let dest_local = clamp(dest_full - params.global_offset, vec2<i32>(0), idims - 1);
    let src_local  = clamp(src_full  - params.global_offset, vec2<i32>(0), idims - 1);

    let dv = textureLoad(input_tex, dest_local, 0).rgb;
    let sv = textureLoad(input_tex, src_local,  0).rgb;

    diff_buf[buf_idx + 0] = dv.r - sv.r;
    diff_buf[buf_idx + 1] = dv.g - sv.g;
    diff_buf[buf_idx + 2] = dv.b - sv.b;
    diff_buf[buf_idx + 3] = 0.0;

    // Circular mask centred at dest centre within bbox
    let ref_pt = params.dest_center - params.bbox_origin;
    let cy = by - ref_pt.y;
    let cx = bx - ref_pt.x;
    let r  = params.radius_px;
    mask_buf[mask_idx] = select(0.0, 1.0, cy * cy + cx * cx <= r * r);
}
