#ifndef DEFINITIONS_H
#define DEFINITIONS_H

typedef float REAL;
typedef unsigned __int8 BYTE;

// Definitions for boundary cells. For the last 5 bits, format is [self] [north] [east] [south] [west]
// Where 1 means the corresponding cell is fluid, and 0 means the corresponding cell is obstacle.
// Boundary cells are defined as obstacle cells with fluid on 1 side or 2 adjacent sides
// For fluid cells, XOR the corresponding inverse boundary with FLUID.
constexpr BYTE B_N   = 0b00001000;
constexpr BYTE B_NE  = 0b00001100;
constexpr BYTE B_E   = 0b00000100;
constexpr BYTE B_SE  = 0b00000110;
constexpr BYTE B_S   = 0b00000010;
constexpr BYTE B_SW  = 0b00000011;
constexpr BYTE B_W   = 0b00000001;
constexpr BYTE B_NW  = 0b00001001;
constexpr BYTE OBS   = 0b00000000;
constexpr BYTE FLUID = 0b00011111;

// Constants used for parsing of flags.
constexpr BYTE SELF  = 0b00010000; // SELF bit
constexpr BYTE NORTH = 0b00001000; // NORTH bit
constexpr BYTE EAST  = 0b00000100; // EAST bit
constexpr BYTE SOUTH = 0b00000010; // SOUTH bit
constexpr BYTE WEST  = 0b00000001; // WEST bit

constexpr BYTE SELFSHIFT  = 4; // Amount to shift for SELF bit at LSB.
constexpr BYTE NORTHSHIFT = 3; // Amount to shift for NORTH bit at LSB.
constexpr BYTE EASTSHIFT  = 2; // Amount to shift for EAST bit at LSB.
constexpr BYTE SOUTHSHIFT = 1; // Amount to shift for SOUTH bit at LSB.
constexpr BYTE WESTSHIFT  = 0; // Amount to shift for WEST bit at LSB.


struct DoubleField
{
	REAL** x;
	REAL** y;
	DoubleField(REAL** x, REAL** y) : x(x), y(y) {}
	DoubleField() : x(nullptr), y(nullptr) {}
};

struct DoubleReal
{
	REAL x;
	REAL y;
	DoubleReal(REAL x, REAL y) : x(x), y(y) {}
	DoubleReal() : x(0), y(0) {}
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
	REAL dynamicViscosity;
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