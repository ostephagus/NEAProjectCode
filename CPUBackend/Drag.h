#ifndef DRAG_H
#define DRAG_H

#include "pch.h"

/// <summary>
/// Computes the drag on the obstacle in the simulation domain.
/// </summary>
/// <returns>The overall drag on the object.</returns>
REAL ComputeObstacleDrag(DoubleField velocities, REAL** pressure, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int iMax, int jMax, DoubleReal stepSizes, REAL viscosity);

/// <summary>
/// Computes the drag coefficient for the obstacle in the simulation domain.
/// </summary>
/// <returns>The drag coefficient for the object.</returns>
REAL ComputeDragCoefficient(DoubleField velocities, REAL** pressure, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int iMax, int jMax, DoubleReal stepSizes, REAL viscosity, REAL density, REAL inflowVelocity);
#endif // !DRAG_H

