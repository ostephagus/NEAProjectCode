#include "Definitions.h"
#include "Computation.h"
#include "DiscreteDerivatives.h"
#include "Init.h"

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

void ComputeFG(DoubleField velocities, DoubleField FG, int iMax, int jMax, REAL timestep, DoubleReal stepSizes, DoubleReal bodyForces, REAL gamma, REAL reynoldsNo) {
	for (int i = 0; i <= iMax; ++i) {
		for (int j = 0; j <= jMax; ++j) {
			bool skipF = false, skipG = false; //Some values are not evaluated for F, G, or both

			if (i == 0) { // Setting F equal to u and G equal to v at the boundaries
				FG.x[i][j] = velocities.x[i][j];
				continue;
			}
			if (j == 0) {
				FG.y[i][j] = velocities.y[i][j];
				continue;
			}

			if (i == iMax) {
				FG.x[i][j] = velocities.x[i][j];
				skipF = true;
				skipG = false;
			}
			if (j == jMax) {
				FG.y[i][j] = velocities.y[i][j];
				skipF = false;
				skipG = true;
			}
			if (!skipF) {
				FG.x[i][j] = velocities.x[i][j] + timestep * (1 / reynoldsNo * (SecondPuPx(velocities.x, i, j, stepSizes.x) + SecondPuPy(velocities.x, i, j, stepSizes.y)) - PuSquaredPx(velocities.x, i, j, stepSizes.x, gamma) - PuvPy(velocities, i, j, stepSizes, gamma) + bodyForces.x);
			}

			if (!skipG) {
				FG.y[i][j] = velocities.y[i][j] + timestep * (1 / reynoldsNo * (SecondPvPx(velocities.y, i, j, stepSizes.x) + SecondPvPy(velocities.y, i, j, stepSizes.y)) - PuvPx(velocities, i, j, stepSizes, gamma) - PvSquaredPy(velocities.y, i, j, stepSizes.y, gamma) + bodyForces.y);
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

void CopyBoundaryPressures(REAL** pressure, int iMax, int jMax) {
	for (int i = 1; i <= iMax; i++) {
		pressure[i][0] = pressure[i][1];
		pressure[i][jMax + 1] = pressure[i][jMax];
	}
	for (int j = 1; j <= jMax; j++) {
		pressure[0][j] = pressure[1][j];
		pressure[iMax + 1][j] = pressure[iMax][j];
	}
}

int Poisson(REAL** pressure, REAL** RHS, int iMax, int jMax, DoubleReal stepSizes, REAL residualTolerance, int maxIterations, REAL omega, REAL residualNorm) { 
	int currentIteration = 0;
	REAL** residualField = MatrixMAlloc(iMax + 2, jMax + 2);
	do {
		CopyBoundaryPressures(pressure, iMax, jMax);
		//IDEAS: Might need 2 different fields for the "previous" iteration and the "current" iteration. These could easily be 2 different memory allocations that are overwritten in turn each iteration. There would need to be some management here about which memory allocation to use, and then at the end some sort of copying to the actual pressure field that is passed as a parameter. This could be done with an array of pointers, 
		currentIteration++;
	} while (currentIteration < maxIterations && residualNorm > residualTolerance);

	FreeMatrix(residualField, iMax + 1);
}
