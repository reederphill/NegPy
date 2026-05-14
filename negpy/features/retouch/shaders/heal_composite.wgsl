// heal_composite.wgsl — Composites the solved diff back into the heal output texture.
// For mask pixels: writes clamp(src + solved_diff, 0, 1) to tex_heal.
// For non-mask pixels: tex_heal was already initialised by heal_copy; nothing to do.

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

@group(0) @binding(0) var  input_tex:  texture_2d<f32>;
@group(0) @binding(1) var  output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<storage, read> diff_buf:  array<f32>;
@group(0) @binding(3) var<storage, read> mask_buf:  array<f32>;
@group(0) @binding(4) var<uniform>       params:    HealParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = i32(gid.x);
    let by = i32(gid.y);
    let bw = params.bbox_size.x;
    let bh = params.bbox_size.y;
    if (bx >= bw || by >= bh) { return; }

    let idx      = (by * bw + bx) * 4;
    let mask_idx =  by * bw + bx;
    if (mask_buf[mask_idx] == 0.0) { return; }   // heal_copy already wrote this pixel

    let idims = vec2<i32>(textureDimensions(input_tex));

    // Source pixel (from the reference region of input_tex)
    let delta     = vec2<i32>(bx, by) - (params.dest_center - params.bbox_origin);
    let src_full  = params.src_center + delta;
    let src_local = clamp(src_full - params.global_offset, vec2<i32>(0), idims - 1);
    let sv        = textureLoad(input_tex, src_local, 0).rgb;

    let diff   = vec3<f32>(diff_buf[idx], diff_buf[idx + 1], diff_buf[idx + 2]);
    let healed = clamp(sv + diff, vec3<f32>(0.0), vec3<f32>(1.0));

    // Dest pixel in tile-local output coords
    let dest_full  = params.bbox_origin + vec2<i32>(bx, by);
    let dest_local = dest_full - params.global_offset;
    textureStore(output_tex, dest_local, vec4<f32>(healed, 1.0));
}
