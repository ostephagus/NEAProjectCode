#ifndef DISCRETE_DERIVATIVES_H
#define DISCRETE_DERIVATIVES_H

#include "pch.h"

REAL PuPx(REAL** hVel, int i, int j, REAL delx);

REAL PvPy(REAL** vVel, int i, int j, REAL dely);

REAL PuSquaredPx(REAL** hVel, int i, int j, REAL delx, REAL gamma);

REAL PvSquaredPy(REAL** vVel, int i, int j, REAL dely, REAL gamma);

REAL PuvPx(DoubleField velocities, int i, int j, DoubleReal stepSizes, REAL gamma);

REAL PuvPy(DoubleField velocities, int i, int j, DoubleReal stepSizes, REAL gamma);

REAL SecondPuPx(REAL** hVel, int i, int j, REAL delx);

REAL SecondPuPy(REAL** hVel, int i, int j, REAL dely);

REAL SecondPvPx(REAL** vVel, int i, int j, REAL delx);

REAL SecondPvPy(REAL** vVel, int i, int j, REAL dely);

REAL PpPx(REAL** pressure, int i, int j, REAL delx);

REAL PpPy(REAL** pressure, int i, int j, REAL dely);

REAL square(REAL operand);

#endif