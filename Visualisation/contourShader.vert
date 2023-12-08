#version 330 core

layout (location = 0) in vec2 position;
layout (location = 1) in float fieldValue; // May not be needed - try commenting out to see what happens

void main()
{
	gl_Position = vec4(position, 0.0f, 1.0f);
}