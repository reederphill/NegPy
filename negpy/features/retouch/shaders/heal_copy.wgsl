// heal_copy.wgsl — Copies the retouch output texture into the heal texture verbatim.
// Runs once per frame when manual spots are present, initialising every pixel in
// tex_heal so that heal_composite only needs to overwrite its masked pixels.

@group(0) @binding(0) var input_tex:  texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims   = vec2<i32>(textureDimensions(input_tex));
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    if (coords.x >= dims.x || coords.y >= dims.y) { return; }
    textureStore(output_tex, coords, textureLoad(input_tex, coords, 0));
}
