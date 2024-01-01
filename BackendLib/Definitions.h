#ifndef DEFINITIONS_H
#define DEFINITIONS_H

typedef float REAL;
typedef unsigned __int8 BYTE;

// Definitions for boundary cells. For the last 5 bits, format is [self] [north] [east] [south] [west]
// Where 1 means the corresponding cell is fluid, and 0 means the corresponding cell is obstacle.
// Boundary cells are defined as obstacle cells with fluid on 1 side or 2 adjacent sides
// For fluid cells, XOR the corresponding inverse boundary with FLUID.
constexpr BYTE B_N = 0b00001000;
constexpr BYTE B_NE = 0b00001100;
constexpr BYTE B_E = 0b00000100;
constexpr BYTE B_SE = 0b00000110;
constexpr BYTE B_S = 0b00000010;
constexpr BYTE B_SW = 0b00000011;
constexpr BYTE B_W = 0b00000001;
constexpr BYTE B_NW = 0b00001001;
constexpr BYTE OBS = 0b00000000;
constexpr BYTE FLUID = 0b00011111;


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

struct SimulationParameters
{
	REAL width;
	REAL height;
	REAL timeStepSafetyFactor;
	REAL relaxationParameter;
	REAL pressureResidualTolerance;
	int pressureMinIterations;
	int pressureMaxIterations;
	REAL reynoldsNo;
	REAL inflowVelocity;
	REAL surfaceFrictionalPermissibility;
	DoubleReal bodyForces;
};

struct ThreadStatus
{
	bool running;
	bool startNextIterationRequested;
	bool stopRequested;

	ThreadStatus() : running(false), startNextIterationRequested(false), stopRequested(false) {} // Constructor just sets everything to false.
};

#endif