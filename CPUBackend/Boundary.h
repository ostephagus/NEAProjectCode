#ifndef BOUNDARY_H
#define BOUNDARY_H

#include "pch.h"

void SetBoundaryConditions(DoubleField velocities, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int iMax, int jMax, REAL inflowVelocity, REAL chi);

void CopyBoundaryPressures(REAL** pressure, std::pair<int, int>* coordinates, int numCoords, BYTE** flags, int iMax, int jMax);

std::pair<std::pair<int, int>*, int> FindBoundaryCells(BYTE** flags, int iMax, int jMax);

#endif