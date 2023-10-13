#include "Definitions.h"
#include "Computation.h"
#include "DiscreteDerivatives.h"
#include "Init.h"
#include "Boundary.h"
#include <iostream>

constexpr BYTE SELF = 0b00010000;
constexpr BYTE NRTH = 0b00001000;
constexpr BYTE EAST = 0b00000100;
constexpr BYTE SOUT = 0b00000010;
constexpr BYTE WEST = 0b00000001;

REAL fieldMax(REAL** field, int xLength, int yLength) {
	REAL max = 0;
	for (int i = 0; i < xLength; ++i) {
		for (int j = 0; j < yLength; ++j) {
			if (field[i][j] > max) {
				max = field[i][j];
			}
		}
	}
	return max;
}

REAL ComputeGamma(DoubleField velocities, int iMax, int jMax, REAL timestep, DoubleReal stepSizes) {
	REAL horizontalComponent = fieldMax(velocities.x, iMax+2, jMax+2) * (timestep / stepSizes.x);
	REAL verticalComponent = fieldMax(velocities.y, iMax+2, jMax+2) * (timestep / stepSizes.y);

	if (horizontalComponent > verticalComponent) {
		return horizontalComponent;
	}
	return verticalComponent;
}

void ComputeFG(DoubleField velocities, DoubleField FG, BYTE** flags, int iMax, int jMax, REAL timestep, DoubleReal stepSizes, DoubleReal bodyForces, REAL gamma, REAL reynoldsNo) {
	// F or G must be set to the corresponding velocity when this references a velocity crossing a boundary
	// F must be set to u when the self bit and the east bit are different (eastern boundary cells and fluid cells to the west of a boundary)
	// G must be set to v when the self bit and the north bit are different (northern boundary cells and fluid cells to the south of a boundary)
	for (int i = 0; i <= iMax; ++i) {
		for (int j = 0; j <= jMax; ++j) {
			bool skipF = false, skipG = false; //Some values are not evaluated for F, G, or both
			if (i == 0 && j == 0) { //Values equal to 0 are boundary cells and are separate with flag 0.
				continue;
			}
			if (i == 0) { // Setting F equal to u and G equal to v at the boundaries
				FG.x[i][j] = velocities.x[i][j];
				continue;
			}
			if (j == 0) {
				FG.y[i][j] = velocities.y[i][j];
				continue;
			}

			if (i == iMax) { //Flag of these will be 00010xxx
				FG.x[i][j] = velocities.x[i][j];
			}
			if (j == jMax) { //Flag of these will be 0001x0xx
				FG.y[i][j] = velocities.y[i][j];
			}

			if ((flags[i][j] & SELF) && (flags[i][j] & EAST)) { // If self bit and east bit are both 1 - fluid cell not near a boundary
				FG.x[i][j] = velocities.x[i][j] + timestep * (1 / reynoldsNo * (SecondPuPx(velocities.x, i, j, stepSizes.x) + SecondPuPy(velocities.x, i, j, stepSizes.y)) - PuSquaredPx(velocities.x, i, j, stepSizes.x, gamma) - PuvPy(velocities, i, j, stepSizes, gamma) + bodyForces.x);
			}
			else if (!(flags[i][j] & SELF) && !(flags[i][j] & EAST)) { // If self bit and east bit are both 0 - inside an obstacle
				FG.x[i][j] = 0;
			}
			else {
				FG.x[i][j] = velocities.x[i][j];
			}

			if ((flags[i][j] & SELF) && (flags[i][j] & NRTH)) { // If self bit and east bit are both 1 - fluid cell not near a boundary
				FG.y[i][j] = velocities.y[i][j] + timestep * (1 / reynoldsNo * (SecondPvPx(velocities.y, i, j, stepSizes.x) + SecondPvPy(velocities.y, i, j, stepSizes.y)) - PuvPx(velocities, i, j, stepSizes, gamma) - PvSquaredPy(velocities.y, i, j, stepSizes.y, gamma) + bodyForces.y);
			}
			else if (!(flags[i][j] & SELF) && !(flags[i][j] & NRTH)) { // If self bit and east bit are both 0 - inside an obstacle
				FG.y[i][j] = 0;
			}
			else {
				FG.y[i][j] = velocities.y[i][j];
			}
		}
	}
}

void ComputeRHS(DoubleField FG, REAL** RHS, int iMax, int jMax, REAL timestep, DoubleReal stepSizes) {
	for (int i = 1; i <= iMax; ++i) {
		for (int j = 1; j <= jMax; ++j) {
			RHS[i][j] = (1 / timestep) * (((FG.x[i][j] - FG.x[i - 1][j]) / stepSizes.x) + ((FG.y[i][j] - FG.y[i][j - 1]) / stepSizes.y));
		}
	}
}

void ComputeTimestep(REAL& timestep, int iMax, int jMax, DoubleReal stepSizes, DoubleField velocities, REAL reynoldsNo, REAL safetyFactor) {
	REAL inverseSquareRestriction = 0.5 * reynoldsNo * (1 / (stepSizes.x * stepSizes.x) + 1 / (stepSizes.y * stepSizes.y));
	REAL xTravelRestriction = stepSizes.x / fieldMax(velocities.x, iMax, jMax);
	REAL yTravelRestriction = stepSizes.y / fieldMax(velocities.y, iMax, jMax);

	REAL smallestRestriction = inverseSquareRestriction; // Choose the smallest restriction
	if (xTravelRestriction < smallestRestriction) {
		smallestRestriction = xTravelRestriction;
	}
	if (yTravelRestriction < smallestRestriction) {
		smallestRestriction = yTravelRestriction;
	}
	timestep = safetyFactor * smallestRestriction;
}

int Poisson(REAL** currentPressure, REAL** RHS, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int numFluidCells, int iMax, int jMax, DoubleReal stepSizes, REAL residualTolerance, int maxIterations, REAL omega, REAL &residualNorm) {
	int currentIteration = 0;
	REAL** nextPressure = MatrixMAlloc(iMax + 2, jMax + 2);
	do {
		residualNorm = 0;
		if (currentIteration % 100 == 0)
		{
			std::cout << "Iteration " << currentIteration << std::endl; //DEBUGGING
		}
		for (int i = 1; i <= iMax; i++) {
			for (int j = 1; j <= jMax; j++) {
				REAL relaxedPressure = (1 - omega) * currentPressure[i][j];
				REAL boundaryFraction = omega / ((2 / square(stepSizes.x)) + (2 / square(stepSizes.y)));
				REAL pressureAverages = ((currentPressure[i + 1][j] + currentPressure[i - 1][j]) / square(stepSizes.x)) + ((currentPressure[i][j + 1] + currentPressure[i][j - 1]) / square(stepSizes.y)) - RHS[i][j];

				nextPressure[i][j] = relaxedPressure + boundaryFraction * pressureAverages;
				//std::cout << nextPressure[i][j];
				REAL currentResidual = pressureAverages - (2 * currentPressure[i][j]) / square(stepSizes.x) - (2 * currentPressure[i][j]) / square(stepSizes.y);
				residualNorm += square(currentResidual);
			}
		}
		
		CopyBoundaryPressures(nextPressure, coordinates, coordinatesLength, flags, iMax, jMax);
		std::swap(currentPressure, nextPressure);
		residualNorm = sqrt(residualNorm) / (numFluidCells);
		if (currentIteration % 100 == 0)
		{
			std::cout << "Residual norm " << residualNorm << std::endl; //DEBUGGING
		}
		currentIteration++;
	} while (currentIteration < maxIterations && residualNorm > residualTolerance);

	FreeMatrix(nextPressure, iMax + 2);
	return currentIteration;
}

void ComputeVelocities(DoubleField velocities, DoubleField FG, REAL** pressure, int iMax, int jMax, REAL timestep, DoubleReal stepSizes) {
	for (int i = 1; i <= iMax; i++) {
		for (int j = 1; j <= jMax; j++) {
			if (i != iMax)
			{
				velocities.x[i][j] = FG.x[i][j] - (timestep / stepSizes.x) * (pressure[i + 1][j] - pressure[i][j]);
			}
			if (j != jMax)
			{
				velocities.y[i][j] = FG.y[i][j] - (timestep / stepSizes.y) * (pressure[i][j + 1] - pressure[i][j]);
			}
		}
	}
}
