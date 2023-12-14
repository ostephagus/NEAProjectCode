#include "Definitions.h"
#include "Init.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

bool** ReadObstaclesFromFile(std::string filename) {
	std::ifstream binFile(filename);
	std::vector<BYTE> buffer(std::istreambuf_iterator<char>(binFile), {}); // Copy the contents of the file to a buffer
	int xLength = *reinterpret_cast<int*>(&buffer[0]); // Use the fact that the bits are stored concurrently in memory to get the xLength and yLength from the first and second 4 bits.
	int yLength = *reinterpret_cast<int*>(&buffer[4]);
	bool** obstacles = new bool* [xLength];
	for (int i = 0; i < xLength; i++) {
		obstacles[i] = new bool[yLength]();
		for (int j = 0; j < yLength; j++) {
			obstacles[i][j] = *reinterpret_cast<bool*>(&buffer[yLength * i + j + 8]);
		}
	}
	return obstacles;
}

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
	std::cout << "Started to allocate obstacle memory" << std::endl;
	// Create array of pointers pointing to more arrays
	bool** matrix = new bool* [xLength];
	std::cout << "Array of pointers allocated" << std::endl;
	//Create the arrays inside each outer array
	for (int i = 0; i < xLength; ++i) {
		matrix[i] = new bool[yLength]();
		std::cout << i << " ";
	}

	std::cout << "Finished allocating obstacle memory" << std::endl;
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
		std::cout << i << " ";
	}
	std::cout << std::endl;
}

