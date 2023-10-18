#ifndef INIT_H
#define INIT_H

#include "Definitions.h"

REAL** MatrixMAlloc(int xLength, int yLength);

BYTE** FlagMatrixMAlloc(int xLength, int yLength);

bool** ObstacleMatrixMAlloc(int xLength, int yLength);

void FreeMatrix(REAL** matrix, int xLength);

void FreeMatrix(BYTE** matrix, int xLength);

void FreeMatrix(bool** matrix, int xLength);

void SetFlags(bool** obstacles, BYTE ** flags, int xLength, int yLength);

#endif