#include "pch.h"
#include "Init.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

REAL** MatrixMAlloc(int xLength, int yLength) {

	// Create array of pointers pointing to more arrays
	REAL** matrix = new REAL* [xLength];

	// Create the arrays inside each outer array
	for (int i = 0; i < xLength; ++i) {
		matrix[i] = new REAL[yLength]();
	}

	return matrix;
}

BYTE** FlagMatrixMAlloc(int xLength, int yLength) {

	// Create array of pointers pointing to more arrays
	BYTE** matrix = new BYTE * [xLength];

	// Create the arrays inside each outer array
	for (int i = 0; i < xLength; ++i) {
		matrix[i] = new BYTE[yLength]();
	}

	return matrix;
}

bool** ObstacleMatrixMAlloc(int xLength, int yLength) {
	// Create array of pointers pointing to more arrays
	bool** matrix = new bool* [xLength];

	// Create the arrays inside each outer array
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

