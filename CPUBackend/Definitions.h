#pragma once

// Definitions for boundary cells. For the last 5 bits, format is [self] [north] [east] [south] [west]
// Where 1 means the corresponding cell is fluid, and 0 means the corresponding cell is obstacle.
// Boundary cells are defined as obstacle cells with fluid on 1 side or 2 adjacent sides
// For fluid cells, XOR the corresponding inverse boundary with FLUID.
#define B_N   0b00001000
#define B_NE  0b00001100
#define B_E   0b00000100
#define B_SE  0b00000110
#define B_S   0b00000010
#define B_SW  0b00000011
#define B_W   0b00000001
#define B_NW  0b00001001
#define OBS   0b00000000
#define FLUID 0b00011111

typedef double REAL;


struct DoubleField
{
	REAL** x;
	REAL** y;
};

struct DoubleReal
{
	REAL x;
	REAL y;
};