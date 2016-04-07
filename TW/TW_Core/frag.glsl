#version 330 core
uniform sampler2D tex;
in vec2 vs_tex_coord;
layout(location = 0) out vec4 color;

void main(void)
{
	color = texture(tex, vs_tex_coord);
}