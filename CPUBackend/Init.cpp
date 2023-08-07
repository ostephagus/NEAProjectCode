#include "Init.h"
#include "Definitions.h"
REAL** MatrixMAlloc(int xLength, int yLength) {

	// Create array of pointers pointing to more arrays
	REAL** matrix = new REAL* [xLength];

	//Create the arrays inside each outer array
	for (int i = 0; i < xLength; ++i) {
		matrix[i] = new REAL[yLength];
	}

	return matrix;
}

void FreeMatrix(REAL** matrix, int xLength) {
	for (int i = 0; i < xLength; ++i) {
		delete[] matrix[i];
	}
	delete[] matrix;
}

