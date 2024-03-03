#include "pch.h"
#include "Solver.h"

Solver::Solver(SimulationParameters parameters, int iMax, int jMax) : iMax(iMax), jMax(jMax), parameters(parameters) {}

Solver::~Solver() {

}

template<typename T>
void Solver::UnflattenArray(T** pointerArray, int paDownOffset, int paLeftOffset, T* flattenedArray, int faDownOffset, int faUpOffset, int faLeftOffset, int xLength, int yLength) {
	int faTotalYLength = faDownOffset + yLength + faUpOffset;
	for (int i = 0; i < xLength; i++) { // Copy one row at a time
		memcpy(
			pointerArray[i + paLeftOffset] + paDownOffset,                       // Destination address - ptr to row (i + paLeftoffset) and column starts at paDownOffset
			flattenedArray + (i + faLeftOffset) * faTotalYLength + faDownOffset, // Source start address - (i + faLeftOffset) * size of a column including offsets, and add down offset for start of copy address
			yLength * sizeof(T)                                                  // Number of bytes to copy, size of a column, excluding offsets
		);
	}
}
template void Solver::UnflattenArray(bool** pointerArray, int paDownOffset, int paLeftOffset, bool* flattenedArray, int faDownOffset, int faUpOffset, int faLeftOffset, int xLength, int yLength); // Templates for the types I may plan to use
template void Solver::UnflattenArray(BYTE** pointerArray, int paDownOffset, int paLeftOffset, BYTE* flattenedArray, int faDownOffset, int faUpOffset, int faLeftOffset, int xLength, int yLength);
template void Solver::UnflattenArray(REAL** pointerArray, int paDownOffset, int paLeftOffset, REAL* flattenedArray, int faDownOffset, int faUpOffset, int faLeftOffset, int xLength, int yLength);


template<typename T>
void Solver::FlattenArray(T** pointerArray, int paDownOffset, int paLeftOffset, T* flattenedArray, int faDownOffset, int faUpOffset, int faLeftOffset, int xLength, int yLength) {
	int faTotalYLength = faDownOffset + yLength + faUpOffset;
	for (int i = 0; i < xLength; i++) { // Copy one row at a time (rows are not guaranteed to be contiguously stored)
		memcpy(
			flattenedArray + (i + faLeftOffset) * faTotalYLength + faDownOffset,  // Destination address - (i + faLeftOffset) * size of a column including offsets, and add down offset for start of copy address
			pointerArray[i + paLeftOffset] + paDownOffset,                        // Source start address - ptr to row (i + paLeftoffset) and column starts at paDownOffset
			yLength * sizeof(T)                                                   // Number of bytes to copy, size of a column, excluding offsets.
		);
	}
}
template void Solver::FlattenArray(bool** pointerArray, int paDownOffset, int paLeftOffset, bool* flattenedArray, int faDownOffset, int faUpOffset, int faLeftOffset, int xLength, int yLength);
template void Solver::FlattenArray(BYTE** pointerArray, int paDownOffset, int paLeftOffset, BYTE* flattenedArray, int faDownOffset, int faUpOffset, int faLeftOffset, int xLength, int yLength);
template void Solver::FlattenArray(REAL** pointerArray, int paDownOffset, int paLeftOffset, REAL* flattenedArray, int faDownOffset, int faUpOffset, int faLeftOffset, int xLength, int yLength);

SimulationParameters Solver::GetParameters() const {
    return parameters;
}

void Solver::SetParameters(SimulationParameters parameters) {
    this->parameters = parameters;
}

int Solver::GetIMax() const {
    return iMax;
}
int Solver::GetJMax() const {
    return jMax;
}

void Solver::SetIMax(int iMax) {
	this->iMax = iMax;
}
void Solver::SetJMax(int jMax) {
	this->jMax = jMax;
}
