#version 330 core

in float relativeStrength;

uniform float subtrahend; // All strength values must have this subtracted from them for the range [0, max] ...
uniform float scalar; // ... and then must be multiplied by this to be in the range [0, 1] for processing

out vec4 FragColour;

void BluePurpleScale(out vec4 colourVector, in float strength) {
    colourVector = vec4(strength, 0.15625, 0.96875, 1.0);
}

void GreenRedScale(out vec4 colourVector, in float strength)
{
    // Red: 0 for strength < 0.5 then linear increase on [0.5, 0.75] then 1 for strength > 0.75
    // Green: 0 for strength < 0.25, then linear increase on [0.25, 0.5] to 1 then linear decrease on [0.5, 0.75], then 0 for strength > 0.75
    // Blue: 1 for strength < 0.25 then linear decrease on [0.25, 0.5] then 0 for strength > 0.5 (opposite of red)
    if (strength < 0.25)
    {
        colourVector = vec4(0.0, 0.0, 1.0, 1.0);
    }
    else if (strength < 0.5) // Interval [0.25, 0.5]
    {
        colourVector = vec4(0.0, (strength - 0.25) * 4.0, 1.0 - (strength - 0.25) * 4, 1.0);
    }
    else if (strength < 0.75) // Interval [0.5, 0.75]
    {
        colourVector = vec4((strength - 0.5) * 4.0, 1.0 - (strength - 0.5) * 4, 0.0, 1.0);
    }
    else
    {
        colourVector = vec4(1.0, 0.0, 0.0, 1.0);
    }
}


void main()
{
    float normalisedStrength = (relativeStrength - subtrahend) * scalar;
    BluePurpleScale(FragColour, normalisedStrength);
}