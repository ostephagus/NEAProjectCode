#include "Definitions.h"
#include "Init.h"
REAL** MatrixMAlloc(int xLength, int yLength) {

	// Create array of pointers pointing to more arrays
	REAL** matrix = new REAL* [xLength];

	//Create the arrays inside each outer array
	for (int i = 0; i < xLength; ++i) {
		matrix[i] = new REAL[yLength]();
	}

	return matrix;
}

BYTE** FlagMatrixMAlloc(int xLength, int yLength) {

	// Create array of pointers pointing to more arrays
	BYTE** matrix = new BYTE* [xLength];

	//Create the arrays inside each outer array
	for (int i = 0; i < xLength; ++i) {
		matrix[i] = new BYTE[yLength]();
	}

	return matrix;
}

bool** ObstacleMatrixMAlloc(int xLength, int yLength) {

	// Create array of pointers pointing to more arrays
	bool** matrix = new bool* [xLength];

	//Create the arrays inside each outer array
	for (int i = 0; i < xLength; ++i) {
		matrix[i] = new bool[yLength]();
	}

	return matrix;
}

void FreeMatrix(REAL** matrix, int xLength) {
	for (int i = 0; i < xLength; ++i) {
		delete[] matrix[i];
	}
	delete[] matrix;
}

void FreeMatrix(BYTE** matrix, int xLength) {
	for (int i = 0; i < xLength; ++i) {
		delete[] matrix[i];
	}
	delete[] matrix;
}

void FreeMatrix(bool** matrix, int xLength) {
	for (int i = 0; i < xLength; ++i) {
		delete[] matrix[i];
	}
	delete[] matrix;
}

void SetFlags(bool** obstacles, BYTE** flags, int xLength, int yLength) {
	for (int i = 1; i < xLength - 1; i++) {
		for (int j = 1; j < yLength - 1; j++) {
			flags[i][j] = ((BYTE)obstacles[i][j] << 4) + ((BYTE)obstacles[i][j + 1] << 3) + ((BYTE)obstacles[i + 1][j] << 2) + ((BYTE)obstacles[i][j - 1] << 1) + (BYTE)obstacles[i - 1][j]; //5 bits in the format: self, north, east, south, west.
		}
	}
}

