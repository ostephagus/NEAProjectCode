#ifndef FLAGS_H

#include "pch.h"

void SetFlags(bool** obstacles, BYTE** flags, int xLength, int yLength);

int CountFluidCells(BYTE** flags, int iMax, int jMax);

#endif // !FLAGS_H

