#version 330 core
uniform sampler2D tex;
in vec2 vs_tex_coord;
layout(location = 0) out vec4 color;

void main(void)
{
	color = texture(tex, vs_tex_coord)*vec4(1.0f,1.0f,1.0f,0.5f);
}