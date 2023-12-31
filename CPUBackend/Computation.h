#ifndef COMPUTATION_H
#define COMPUTATION_H

#include "Definitions.h"

constexpr BYTE SELF = 0b00010000;
constexpr BYTE NORTH = 0b00001000;
constexpr BYTE EAST = 0b00000100;
constexpr BYTE SOUTH = 0b00000010;
constexpr BYTE WEST = 0b00000001;


REAL FieldMax(REAL** field, int xLength, int yLength);

REAL ComputeGamma(DoubleField velocities, int iMax, int jMax, REAL timestep, DoubleReal stepSizes);

void ComputeFG(DoubleField velocities, DoubleField FG, BYTE** flags, int iMax, int jMax, REAL timestep, DoubleReal stepSizes, DoubleReal bodyForces, REAL gamma, REAL reynoldsNo);

void ComputeRHS(DoubleField FG, REAL** RHS, BYTE** flags, int iMax, int jMax, REAL timestep, DoubleReal stepSizes);

void ComputeTimestep(REAL& timestep, int iMax, int jMax, DoubleReal stepSizes, DoubleField velocities, REAL reynoldsNo, REAL safetyFactor);

void PoissonSubset(REAL** pressure, REAL** RHS, BYTE** flags, int xOffset, int yOffset, int iMax, int jMax, DoubleReal stepSizes, REAL omega, REAL boundaryFraction, REAL& residualNormSquare);

void ThreadLoop(REAL** pressure, REAL** RHS, BYTE** flags, int xOffset, int yOffset, int iMax, int jMax, DoubleReal stepSizes, REAL omega, REAL boundaryFraction, REAL& residualNormSquare, ThreadStatus& threadStatus);

int PoissonThreadPool(REAL** pressure, REAL** RHS, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int numFluidCells, int iMax, int jMax, DoubleReal stepSizes, REAL residualTolerance, int minIterations, int maxIterations, REAL omega, REAL& residualNorm);

int PoissonMultiThreaded(REAL** pressure, REAL** RHS, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int numFluidCells, int iMax, int jMax, DoubleReal stepSizes, REAL residualTolerance, int minIterations, int maxIterations, REAL omega, REAL& residualNorm);

int Poisson(REAL** pressure, REAL** RHS, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int numFluidCells, int iMax, int jMax, DoubleReal stepSizes, REAL residualTolerance, int minIterations, int maxIterations, REAL omega, REAL& residualNorm);

void ComputeVelocities(DoubleField velocities, DoubleField FG, REAL** pressure, BYTE** flags, int iMax, int jMax, REAL timestep, DoubleReal stepSizes);

void ComputeStream(DoubleField velocities, REAL** streamFunction, int iMax, int jMax, DoubleReal stepSizes);

#endif