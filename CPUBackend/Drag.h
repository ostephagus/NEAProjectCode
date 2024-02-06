#ifndef DRAG_H
#define DRAG_H

#include "pch.h"

/// <summary>
/// Computes the drag on the obstacle in the simulation domain.
/// </summary>
/// <returns>The overall drag on the object.</returns>
REAL ComputeObstacleDrag(DoubleField velocities, REAL** pressure, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int iMax, int jMax, DoubleReal stepSizes, REAL viscosity);

#endif // !DRAG_H

