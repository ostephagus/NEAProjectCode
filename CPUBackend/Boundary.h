#ifndef BOUNDARY_H
#define BOUNDARY_H

#include "Definitions.h"

void SetBoundaryConditions(DoubleField velocities, int iMax, int jMax, REAL inflowVelocity);

void CopyBoundaryPressures(REAL** pressure, int iMax, int jMax);

#endif