#version 330 core

in float relativeStrength;

out vec4 FragColour;

void main()
{
    FragColour = vec4(1.0, relativeStrength, 0.1, 1.0);
}