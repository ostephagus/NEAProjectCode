#pragma once

REAL fieldMax(REAL** field, int xLength, int yLength);

void ComputeFG(DoubleField velocities, DoubleField FG, int iMax, int jMax, REAL timestep, DoubleReal stepSizes, DoubleReal bodyForces, REAL gamma, REAL reynoldsNo);

void ComputeRHS(DoubleField FG, REAL** RHS, int iMax, int jMax, REAL timestep, DoubleReal stepSizes);

void ComputeTimestep(REAL& timestep, int iMax, int jMax, DoubleReal stepSizes, DoubleField velocities, REAL reynoldsNo, REAL safetyFactor);

void CopyBoundaryPressures(REAL** pressure, int iMax, int jMax);

int Poisson(REAL** pressure, REAL** RHS, int iMax, int jMax, DoubleReal stepSizes, REAL residualTolerance, int maxIterations, REAL omega, REAL residualNorm);
