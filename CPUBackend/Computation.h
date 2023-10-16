#ifndef COMPUTATION_H
#define COMPUTATION_H

#include <utility>

REAL fieldMax(REAL** field, int xLength, int yLength);

REAL ComputeGamma(DoubleField velocities, int iMax, int jMax, REAL timestep, DoubleReal stepSizes);

void ComputeFG(DoubleField velocities, DoubleField FG, BYTE** flags, int iMax, int jMax, REAL timestep, DoubleReal stepSizes, DoubleReal bodyForces, REAL gamma, REAL reynoldsNo);

void ComputeRHS(DoubleField FG, REAL** RHS, BYTE** flags, int iMax, int jMax, REAL timestep, DoubleReal stepSizes);

void ComputeTimestep(REAL& timestep, int iMax, int jMax, DoubleReal stepSizes, DoubleField velocities, REAL reynoldsNo, REAL safetyFactor);

int Poisson(REAL** currentPressure, REAL** nextPressure, REAL** RHS, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int numFluidCells, int iMax, int jMax, DoubleReal stepSizes, REAL residualTolerance, int maxIterations, REAL omega, REAL& residualNorm);

void ComputeVelocities(DoubleField velocities, DoubleField FG, REAL** pressure, BYTE** flags, int iMax, int jMax, REAL timestep, DoubleReal stepSizes);

#endif