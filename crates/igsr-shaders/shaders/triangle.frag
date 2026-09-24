#version 330 core
// IGSR testbed debug triangle (own code).
in vec3 v_col;
out vec4 o_col;
void main() {
    o_col = vec4(v_col, 1.0);
}
