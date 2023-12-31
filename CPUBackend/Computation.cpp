#include "Computation.h"
#include "DiscreteDerivatives.h"
#include "Init.h"
#include "Boundary.h"
#include <iostream>
#include <thread>
#include <cmath>
#include <chrono>
//#define DEBUGOUT

REAL ArraySum(REAL* array, int arrayLength) {
	if (arrayLength == 0) return 0;
	if (arrayLength == 1) return array[0];
	int midPoint = arrayLength / 2;
	return ArraySum(array, midPoint) + ArraySum((array + midPoint), arrayLength - midPoint);
}

REAL FieldMax(REAL** field, int xLength, int yLength) {
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
	REAL horizontalComponent = FieldMax(velocities.x, iMax+2, jMax+2) * (timestep / stepSizes.x);
	REAL verticalComponent = FieldMax(velocities.y, iMax+2, jMax+2) * (timestep / stepSizes.y);

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
			if (i == 0 && j == 0) { // Values equal to 0 are boundary cells and are separate with flag 0.
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

			if (i == iMax) { // Flag of these will be 00010xxx
				FG.x[i][j] = velocities.x[i][j];
			}
			if (j == jMax) { // Flag of these will be 0001x0xx
				FG.y[i][j] = velocities.y[i][j];
			}

			if (flags[i][j] & SELF && flags[i][j] & EAST) { // If self bit and east bit are both 1 - fluid cell not near a boundary
				FG.x[i][j] = velocities.x[i][j] + timestep * (1 / reynoldsNo * (SecondPuPx(velocities.x, i, j, stepSizes.x) + SecondPuPy(velocities.x, i, j, stepSizes.y)) - PuSquaredPx(velocities.x, i, j, stepSizes.x, gamma) - PuvPy(velocities, i, j, stepSizes, gamma) + bodyForces.x);
			}
			else if (!(flags[i][j] & SELF) && !(flags[i][j] & EAST)) { // If self bit and east bit are both 0 - inside an obstacle
				FG.x[i][j] = 0;
			}
			else { // The variable's position lies on a boundary (though the cell may not - a side-effect of the staggered-grid discretisation.
				FG.x[i][j] = velocities.x[i][j];
			}

			if (flags[i][j] & SELF && flags[i][j] & NORTH) { // Same as for G, but the relevant bits are self and north
				FG.y[i][j] = velocities.y[i][j] + timestep * (1 / reynoldsNo * (SecondPvPx(velocities.y, i, j, stepSizes.x) + SecondPvPy(velocities.y, i, j, stepSizes.y)) - PuvPx(velocities, i, j, stepSizes, gamma) - PvSquaredPy(velocities.y, i, j, stepSizes.y, gamma) + bodyForces.y);
			}
			else if (!(flags[i][j] & SELF) && !(flags[i][j] & NORTH)) {
				FG.y[i][j] = 0;
			}
			else {
				FG.y[i][j] = velocities.y[i][j];
			}
		}
	}
}

void ComputeRHS(DoubleField FG, REAL** RHS, BYTE** flags, int iMax, int jMax, REAL timestep, DoubleReal stepSizes) {
	for (int i = 1; i <= iMax; ++i) {
		for (int j = 1; j <= jMax; ++j) {
			if (!(flags[i][j] & SELF)) { // RHS is defined in the middle of cells, so only check the SELF bit
				continue; // Skip if the cell is not a fluid cell
			}
			RHS[i][j] = (1 / timestep) * (((FG.x[i][j] - FG.x[i - 1][j]) / stepSizes.x) + ((FG.y[i][j] - FG.y[i][j - 1]) / stepSizes.y));
		}
	}
}

void ComputeTimestep(REAL& timestep, int iMax, int jMax, DoubleReal stepSizes, DoubleField velocities, REAL reynoldsNo, REAL safetyFactor) {
	REAL inverseSquareRestriction = (REAL)0.5 * reynoldsNo * (1 / (stepSizes.x * stepSizes.x) + 1 / (stepSizes.y * stepSizes.y));
	REAL xTravelRestriction = stepSizes.x / FieldMax(velocities.x, iMax, jMax);
	REAL yTravelRestriction = stepSizes.y / FieldMax(velocities.y, iMax, jMax);

	REAL smallestRestriction = inverseSquareRestriction; // Choose the smallest restriction
	if (xTravelRestriction < smallestRestriction) {
		smallestRestriction = xTravelRestriction;
	}
	if (yTravelRestriction < smallestRestriction) {
		smallestRestriction = yTravelRestriction;
	}
	timestep = safetyFactor * smallestRestriction;
}

void PoissonSubset(REAL** pressure, REAL** RHS, BYTE** flags, int xOffset, int yOffset, int iMax, int jMax, DoubleReal stepSizes, REAL omega, REAL boundaryFraction, REAL& residualNormSquare) {
	for (int i = xOffset + 1; i <= iMax; i++) {
		for (int j = yOffset + 1; j <= jMax; j++) {
			if (!(flags[i][j] & SELF)) { // Pressure is defined in the middle of cells, so only check the SELF bit
				continue; // Skip if the cell is not a fluid cell
			}
			REAL relaxedPressure = (1 - omega) * pressure[i][j];
			REAL pressureAverages = ((pressure[i + 1][j] + pressure[i - 1][j]) / square(stepSizes.x)) + ((pressure[i][j + 1] + pressure[i][j - 1]) / square(stepSizes.y)) - RHS[i][j];

			pressure[i][j] = relaxedPressure + boundaryFraction * pressureAverages;
			residualNormSquare += square(pressureAverages - (2 * pressure[i][j]) / square(stepSizes.x) - (2 * pressure[i][j]) / square(stepSizes.y));
		}
	}
}

void ThreadLoop(REAL** pressure, REAL** RHS, BYTE** flags, int xOffset, int yOffset, int iMax, int jMax, DoubleReal stepSizes, REAL omega, REAL boundaryFraction, REAL& residualNormSquare, ThreadStatus& threadStatus) {
	while (!threadStatus.stopRequested) { // Condition to stop the thread entirely
		std::cout << "Thread waiting" << std::endl;
		while (!threadStatus.startNextIterationRequested) { // Wait until the next iteration is requested
			if (threadStatus.stopRequested) { // If a request to stop occurs in this loop, do not complete another iteration.
				return;
			}
		}
		std::cout << "Thread running" << std::endl;
		threadStatus.running = true;
		threadStatus.startNextIterationRequested = false; // Set it to false so that only 1 iteration occurs if there is no input from thread owner
		for (int i = xOffset + 1; i <= iMax; i++) {
			for (int j = yOffset + 1; j <= jMax; j++) {
				if (!(flags[i][j] & SELF)) { // Pressure is defined in the middle of cells, so only check the SELF bit
					continue; // Skip if the cell is not a fluid cell
				}
				REAL relaxedPressure = (1 - omega) * pressure[i][j];
				REAL pressureAverages = ((pressure[i + 1][j] + pressure[i - 1][j]) / square(stepSizes.x)) + ((pressure[i][j + 1] + pressure[i][j - 1]) / square(stepSizes.y)) - RHS[i][j];

				pressure[i][j] = relaxedPressure + boundaryFraction * pressureAverages;
				residualNormSquare = square(pressureAverages - (2 * pressure[i][j]) / square(stepSizes.x) - (2 * pressure[i][j]) / square(stepSizes.y));
			}
		}
		threadStatus.running = false;
	}
}

int PoissonThreadPool(REAL** pressure, REAL** RHS, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int numFluidCells, int iMax, int jMax, DoubleReal stepSizes, REAL residualTolerance, int minIterations, int maxIterations, REAL omega, REAL& residualNorm) {
	int currentIteration = 0;
	REAL boundaryFraction = omega / ((2 / square(stepSizes.x)) + (2 / square(stepSizes.y)));

	int totalThreads = std::thread::hardware_concurrency(); // Number of threads returned by hardware, may not be reliable and may be 0 in error case
	int xBlocks, yBlocks; // Number of blocks in the x and y direction

	if (totalThreads % 4 == 0 && totalThreads > 4) { // Encompasses most multi-threaded CPUs (even number of cores, 2 threads per core)
		yBlocks = 4;
		xBlocks = totalThreads / 4;
	}
	else if (totalThreads % 2 == 0 && totalThreads > 2) { // Hopefully a catch-all case given all modern CPUs have even numbers of cores
		yBlocks = 2;
		xBlocks = totalThreads / 2;
	}
	else { // threadHint is odd or 0
		totalThreads = 1;
		yBlocks = 1;
		xBlocks = 1;
	}

	// Initialise the threads to use, which at this point will be sitting in a loop waiting for the next iteration request
	REAL* residualNorms = new REAL[totalThreads]();
	std::thread* threads = new std::thread[totalThreads]; // Array of all running threads, heap allocated because size is runtime-determined
	ThreadStatus* threadStatuses = new ThreadStatus[totalThreads]();
	int threadNum = 0;
	for (int xBlock = 0; xBlock < xBlocks; xBlock++) {
		for (int yBlock = 0; yBlock < yBlocks; yBlock++) {
			threads[threadNum] = std::thread(ThreadLoop, pressure, RHS, flags, (iMax * xBlock) / xBlocks, (jMax * yBlock) / yBlocks, (iMax * (xBlock + 1)) / xBlocks, (jMax * (yBlock + 1)) / yBlocks, stepSizes, omega, boundaryFraction, std::ref(residualNorms[threadNum]), std::ref(threadStatuses[threadNum]));
			threadNum++;
		}
	}
	do {
		residualNorm = 0;

		// Dispach threads and perform computation
		for (int threadNum = 0; threadNum < totalThreads; threadNum++) {
			threadStatuses[threadNum].startNextIterationRequested = true; // Loop through the threads and start the iteration
			threadStatuses[threadNum].running = true; // TESTING
		}


		// Wait for threads to finish exection
		for (int threadNum = 0; threadNum < totalThreads; threadNum++) {
			while (threadStatuses[threadNum].running) {} // Wait until the current thread stops running
			residualNorm += residualNorms[threadNum];
		}


		CopyBoundaryPressures(pressure, coordinates, coordinatesLength, flags, iMax, jMax);
		residualNorm = sqrt(residualNorm) / (numFluidCells);
		currentIteration++;
	} while ((currentIteration < maxIterations && residualNorm > residualTolerance) || currentIteration < minIterations);

	// Stop and join the threads
	for (int threadNum = 0; threadNum < totalThreads; threadNum++) {
		threadStatuses[threadNum].stopRequested = true; // Request for stop
		threads[threadNum].join(); // And wait for it to actually stop
	}

	delete[] threadStatuses;
	delete[] threads;
	delete[] residualNorms;
	return currentIteration;
}

int PoissonMultiThreaded(REAL** pressure, REAL** RHS, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int numFluidCells, int iMax, int jMax, DoubleReal stepSizes, REAL residualTolerance, int minIterations, int maxIterations, REAL omega, REAL& residualNorm) {
	int currentIteration = 0;
	REAL boundaryFraction = omega / ((2 / square(stepSizes.x)) + (2 / square(stepSizes.y)));
	
	int threadHint = std::thread::hardware_concurrency(); // Number of threads returned by hardware, may not be reliable and may be 0 in error case
	int xBlocks, yBlocks; // Number of blocks in the x and y direction

	if (threadHint % 4 == 0 && threadHint > 4) { // Encompasses most multi-threaded CPUs (even number of cores, 2 threads per core)
		yBlocks = 4;
		xBlocks = threadHint / 4;
	}
	else if (threadHint % 2 == 0 && threadHint > 2) { // Hopefully a catch-all case given all modern CPUs have even numbers of cores
		yBlocks = 2;
		xBlocks = threadHint / 2;
	}
	else { // threadHint is odd or 0
		if (threadHint == 0) threadHint = 1;
		yBlocks = 1;
		xBlocks = 1;
	}
	REAL* residualNorms = new REAL[xBlocks * yBlocks]();

	do {
		CopyBoundaryPressures(pressure, coordinates, coordinatesLength, flags, iMax, jMax);

		residualNorm = 0;
#ifdef DEBUGOUT
		if (currentIteration % 100 == 0)
		{
			std::cout << "Pressure iteration " << currentIteration << std::endl; // DEBUGGING
		}
#endif // DEBUGOUT
		// Dispach threads and perform computation
		std::thread* threads = new std::thread[xBlocks * yBlocks]; // Array of all running threads, heap allocated because size is runtime-determined
		int threadNum = 0;
		for (int xBlock = 0; xBlock < xBlocks; xBlock++) {
			for (int yBlock = 0; yBlock < yBlocks; yBlock++) {
				threads[threadNum] = std::thread(PoissonSubset, pressure, RHS, flags, (iMax * xBlock) / xBlocks, (jMax * yBlock) / yBlocks, (iMax * (xBlock + 1)) / xBlocks, (jMax * (yBlock + 1)) / yBlocks, stepSizes, omega, boundaryFraction, std::ref(residualNorms[threadNum]));
				threadNum++;
			}
		}
		

		// Wait for threads to finish exection
		for (int threadNum = 0; threadNum < xBlocks * yBlocks; threadNum++) {
			threads[threadNum].join();
		}

		residualNorm = ArraySum(residualNorms, xBlocks * yBlocks);

		delete[] threads;

		residualNorm = sqrt(residualNorm) / (numFluidCells);
#ifdef DEBUGOUT
		if (currentIteration % 100 == 0)
		{
			std::cout << "Residual norm " << residualNorm << std::endl; // DEBUGGING
		}
#endif // DEBUGOUT
		currentIteration++;
	} while ((currentIteration < maxIterations && residualNorm > residualTolerance) || currentIteration < minIterations);
	delete[] residualNorms;
	return currentIteration;
}

int Poisson(REAL** pressure, REAL** RHS, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int numFluidCells, int iMax, int jMax, DoubleReal stepSizes, REAL residualTolerance, int minIterations, int maxIterations, REAL omega, REAL &residualNorm) {
	int currentIteration = 0;
	REAL boundaryFraction = omega / ((2 / square(stepSizes.x)) + (2 / square(stepSizes.y)));
	do {

		residualNorm = 0;
#ifdef DEBUGOUT
		if (currentIteration % 100 == 0)
		{
			std::cout << "Pressure iteration " << currentIteration << std::endl; // DEBUGGING
		}
#endif // DEBUGOUT
		for (int i = 1; i <= iMax; i++) {
			if (i == 224) {
				int bp = 1;
			}
			for (int j = 1; j <= jMax; j++) {
				if (j == 250) {
					int bp2 = 1;
				}
				if (!(flags[i][j] & SELF)) { // Pressure is defined in the middle of cells, so only check the SELF bit
					continue; // Skip if the cell is not a fluid cell
				}
				REAL relaxedPressure = (1 - omega) * pressure[i][j];
				REAL pressureAverages = ((pressure[i + 1][j] + pressure[i - 1][j]) / square(stepSizes.x)) + ((pressure[i][j + 1] + pressure[i][j - 1]) / square(stepSizes.y)) - RHS[i][j];

				pressure[i][j] = relaxedPressure + boundaryFraction * pressureAverages;
				REAL currentResidual = pressureAverages - (2 * pressure[i][j]) / square(stepSizes.x) - (2 * pressure[i][j]) / square(stepSizes.y);
				residualNorm += square(currentResidual);
			}
		}
		
		residualNorm = sqrt(residualNorm / numFluidCells);
		CopyBoundaryPressures(pressure, coordinates, coordinatesLength, flags, iMax, jMax);
#ifdef DEBUGOUT
		if (currentIteration % 100 == 0)
		{
			std::cout << "Residual norm " << residualNorm << std::endl; // DEBUGGING
		}
#endif // DEBUGOUT
		currentIteration++;
	} while ((currentIteration < maxIterations && residualNorm > residualTolerance) || currentIteration < minIterations);
	return currentIteration;
}

void ComputeVelocities(DoubleField velocities, DoubleField FG, REAL** pressure, BYTE** flags, int iMax, int jMax, REAL timestep, DoubleReal stepSizes) {
	for (int i = 1; i <= iMax; i++) {
		for (int j = 1; j <= jMax; j++) {
			if (!(flags[i][j] & SELF)) { // If the cell is not a fluid cell, skip it
				continue;
			}
			if (flags[i][j] & EAST) // If the edge the velocity is defined on is a boundary edge, skip the calculation (this is when the cell to the east is not fluid)
			{
				velocities.x[i][j] = FG.x[i][j] - (timestep / stepSizes.x) * (pressure[i + 1][j] - pressure[i][j]);
			}
			if (flags[i][j] & NORTH) // Same, but in this case for north boundary
			{
				velocities.y[i][j] = FG.y[i][j] - (timestep / stepSizes.y) * (pressure[i][j + 1] - pressure[i][j]);
			}
		}
	}
}

void ComputeStream(DoubleField velocities, REAL** streamFunction, int iMax, int jMax, DoubleReal stepSizes) {
	for (int i = 0; i <= iMax; i++) {
		streamFunction[i][0] = 0; // Stream function boundary condition
		for (int j = 1; j <= jMax; j++) {
			streamFunction[i][j] = streamFunction[i][j - 1] + velocities.x[i][j] * stepSizes.y; // Obstacle boundary conditions are taken care of by the fact that u = 0 inside obstacle cells.
		}
	}
}

