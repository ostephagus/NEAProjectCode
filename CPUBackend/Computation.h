#ifndef COMPUTATION_H
#define COMPUTATION_H

REAL fieldMax(REAL** field, int xLength, int yLength);

REAL ComputeGamma(DoubleField velocities, int iMax, int jMax, REAL timestep, DoubleReal stepSizes);

void ComputeFG(DoubleField velocities, DoubleField FG, int iMax, int jMax, REAL timestep, DoubleReal stepSizes, DoubleReal bodyForces, REAL gamma, REAL reynoldsNo);

void ComputeRHS(DoubleField FG, REAL** RHS, int iMax, int jMax, REAL timestep, DoubleReal stepSizes);

void ComputeTimestep(REAL& timestep, int iMax, int jMax, DoubleReal stepSizes, DoubleField velocities, REAL reynoldsNo, REAL safetyFactor);

int Poisson(REAL** currentPressure, REAL** RHS, int iMax, int jMax, DoubleReal stepSizes, REAL residualTolerance, int maxIterations, REAL omega, REAL &residualNorm);

void ComputeVelocities(DoubleField velocities, DoubleField FG, REAL** pressure, int iMax, int jMax, REAL timestep, DoubleReal stepSizes);

#endif