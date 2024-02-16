#include "pch.h"
#include "Flags.h"

void SetFlags(bool** obstacles, BYTE** flags, int xLength, int yLength) {
    for (int i = 1; i < xLength - 1; i++) {
        for (int j = 1; j < yLength - 1; j++) {
            flags[i][j] = ((BYTE)obstacles[i][j] << 4) + ((BYTE)obstacles[i][j + 1] << 3) + ((BYTE)obstacles[i + 1][j] << 2) + ((BYTE)obstacles[i][j - 1] << 1) + (BYTE)obstacles[i - 1][j]; // 5 bits in the format: self, north, east, south, west.
        }
    }
}

// Counts number of fluid cells in the region [1,iMax]x[1,jMax]
int CountFluidCells(BYTE** flags, int iMax, int jMax) {
    int count = 0;
    for (int i = 0; i <= iMax; i++) {
        for (int j = 0; j <= jMax; j++) {
            count += flags[i][j] >> 4; // This will include only the "self" bit, which is one for fluid cells and 0 for boundary and obstacle cells.
        }
    }
    return count;
}