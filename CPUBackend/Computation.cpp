#include "Definitions.h"
#include "Computation.h"
#include "DiscreteDerivatives.h"
#include "Init.h"
#include <algorithm>
#include <iostream>


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

void ComputeFG(DoubleField velocities, DoubleField FG, int iMax, int jMax, REAL timestep, DoubleReal stepSizes, DoubleReal bodyForces, REAL gamma, REAL reynoldsNo) {
	for (int i = 0; i <= iMax; ++i) {
		for (int j = 0; j <= jMax; ++j) {
			bool skipF = false, skipG = false; //Some values are not evaluated for F, G, or both
			if (i == 0 && j == 0) {
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

			if (i == iMax) {
				FG.x[i][j] = velocities.x[i][j];
				skipF = true;
			}
			if (j == jMax) {
				FG.y[i][j] = velocities.y[i][j];
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

int Poisson(REAL** currentPressure, REAL** RHS, int iMax, int jMax, DoubleReal stepSizes, REAL residualTolerance, int maxIterations, REAL omega, REAL &residualNorm) {
	int currentIteration = 0;
	REAL** nextPressure = MatrixMAlloc(iMax + 2, jMax + 2);
	//REAL** residualField = MatrixMAlloc(iMax + 1, jMax + 1);
	do {
		std::cout << "Iteration " << currentIteration << std::endl; //DEBUGGING
		residualNorm = 0;
		for (int i = 1; i <= iMax; i++) {
			for (int j = 1; j <= jMax; j++) {
				REAL relaxedPressure = (1 - omega) * currentPressure[i][j];
				REAL boundaryFraction = omega / ((2 / square(stepSizes.x)) + (2 / square(stepSizes.y)));
				REAL pressureAverages = ((currentPressure[i + 1][j] + currentPressure[i - 1][j]) / square(stepSizes.x)) + ((currentPressure[i][j + 1] + currentPressure[i][j - 1]) / square(stepSizes.y)) - RHS[i][j];

				nextPressure[i][j] = relaxedPressure + boundaryFraction * pressureAverages;
				std::cout << nextPressure[i][j];
				REAL currentResidual = pressureAverages - (2 * currentPressure[i][j]) / square(stepSizes.x) - (2 * currentPressure[i][j]) / square(stepSizes.y);
				residualNorm += square(currentResidual);
				std::cout << ' ';
			}
			std::cout << std::endl;
		}
		
		CopyBoundaryPressures(nextPressure, iMax, jMax);
		std::swap(currentPressure, nextPressure);
		residualNorm = sqrt(residualNorm);
		currentIteration++;
	} while (currentIteration < maxIterations && residualNorm > residualTolerance);

	FreeMatrix(nextPressure, iMax + 1);
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
