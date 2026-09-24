#version 420 core
// Fullscreen blit vertex for the fragment fallback path (stage 4).
// Own code. Locations are explicit to match the 420 convert/upscale
// fragment shaders.
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 0) out vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
