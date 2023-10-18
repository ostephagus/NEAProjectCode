#include "DiscreteDerivatives.h"
#include <cmath>

/*
A note on terminology:
Below are functions to represent the calculations of different derivatives used in the Navier-Stokes equations. They have been discretised.
Average: the sum of 2 quantities, then divided by 2. Taking the mean of the 2 quantities.
Difference: The same as above, but with subtraction.
Forward: applying an average or difference between the current cell (i,j) and the next cell along (i+1,j) or (i,j+1)
Backward: the same as above, but applied to the cell behind - (i-1,j) or (i,j-1).
Downshift: Any of the above with respect to the cell below the current one, (i, j-1).
Second derivative: the double application of a derivative.
Donor and non-donor: There are 2 different discretisation methods here, one of which is donor-cell discretisation. The 2 parts of each discretisation formula are named as such.
*/

REAL PuPx(REAL** hVel, int i, int j, REAL delx) { //NOTE: P here is used to represent the partial operator, so PuPx should be read "partial u by partial x"
	return (hVel[i][j] - hVel[i - 1][j]) / delx;
}

REAL PvPy(REAL** vVel, int i, int j, REAL dely) {
	return (vVel[i][j] - vVel[i][j - 1]) / dely;
}

REAL PuSquaredPx
(REAL** hVel, int i, int j, REAL delx, REAL gamma) {
	REAL forwardAverage = (hVel[i][j] + hVel[i + 1][j]) / 2;
	REAL backwardAverage = (hVel[i - 1][j] + hVel[i][j]) / 2;

	REAL forwardDifference = (hVel[i][j] - hVel[i + 1][j]) / 2;
	REAL backwardDifference = (hVel[i - 1][j] - hVel[i][j]) / 2;

	REAL nonDonorTerm = (1 / delx) * (square(forwardAverage) - square(backwardAverage));
	REAL donorTerm = (gamma / delx) * ((abs(forwardAverage) * forwardDifference) - (abs(backwardAverage) * backwardDifference));

	return nonDonorTerm + donorTerm;
}

REAL PvSquaredPy(REAL** vVel, int i, int j, REAL dely, REAL gamma) {
	REAL forwardAverage = (vVel[i][j] + vVel[i][j + 1]) / 2;
	REAL backwardAverage = (vVel[i][j - 1] + vVel[i][j]) / 2;

	REAL forwardDifference = (vVel[i][j] - vVel[i][j + 1]) / 2;
	REAL backwardDifference = (vVel[i][j - 1] - vVel[i][j]) / 2;

	REAL nonDonorTerm = (1 / dely) * (square(forwardAverage) - square(backwardAverage));
	REAL donorTerm = (gamma / dely) * ((abs(forwardAverage) * forwardDifference) - (abs(backwardAverage) * backwardDifference));
	return nonDonorTerm + donorTerm;
}

REAL PuvPx(DoubleField velocities, int i, int j, DoubleReal stepSizes, REAL gamma) {
	REAL jForwardAverageU = (velocities.x[i][j] + velocities.x[i][j + 1]) / 2;
	REAL iForwardAverageV = (velocities.y[i][j] + velocities.y[i + 1][j]) / 2;
	REAL iBackwardAverageV = (velocities.y[i - 1][j] + velocities.y[i][j]) / 2;

	REAL jForwardAverageUDownshift = (velocities.x[i - 1][j] + velocities.x[i - 1][j + 1]) / 2;

	REAL iForwardDifferenceV = (velocities.y[i][j] - velocities.y[i + 1][j]) / 2;
	REAL iBackwardDifferenceV = (velocities.y[i - 1][j] - velocities.y[i][j]) / 2;

	REAL nonDonorTerm = (1 / stepSizes.x) * ((jForwardAverageU * iForwardAverageV) - (jForwardAverageUDownshift * iBackwardAverageV));
	REAL donorTerm = (gamma / stepSizes.x) * ((abs(jForwardAverageU) * iForwardDifferenceV) - (abs(jForwardAverageUDownshift) * iBackwardDifferenceV));
	return nonDonorTerm + donorTerm;
}

REAL PuvPy(DoubleField velocities, int i, int j, DoubleReal stepSizes, REAL gamma) {
	REAL iForwardAverageV = (velocities.y[i][j] + velocities.y[i + 1][j]) / 2;
	REAL jForwardAverageU = (velocities.x[i][j] + velocities.x[i][j + 1]) / 2;
	REAL jBackwardAverageU = (velocities.x[i][j - 1] + velocities.x[i][j]) / 2;

	REAL iForwardAverageVDownshift = (velocities.y[i][j - 1] + velocities.y[i + 1][j - 1]) / 2;

	REAL jForwardDifferenceU = (velocities.x[i][j] - velocities.x[i][j + 1]) / 2;
	REAL jBackwardDifferenceU = (velocities.x[i][j - 1] - velocities.x[i][j]) / 2;

	REAL nonDonorTerm = (1 / stepSizes.y) * ((iForwardAverageV * jForwardAverageU) - (iForwardAverageVDownshift * jBackwardAverageU));
	REAL donorTerm = (gamma / stepSizes.y) * ((abs(iForwardAverageV) * jForwardDifferenceU) - (abs(iForwardAverageVDownshift) * jBackwardDifferenceU));

	return nonDonorTerm + donorTerm;
}

REAL SecondPuPx(REAL** hVel, int i, int j, REAL delx) {
	return (hVel[i + 1][j] - 2 * hVel[i][j] + hVel[i - 1][j]) / square(delx);
}

REAL SecondPuPy(REAL** hVel, int i, int j, REAL dely) {
	return (hVel[i][j + 1] - 2 * hVel[i][j] + hVel[i][j - 1]) / square(dely);
}

REAL SecondPvPx(REAL** vVel, int i, int j, REAL delx) {
	return (vVel[i + 1][j] - 2 * vVel[i][j] + vVel[i - 1][j]) / square(delx);
}

REAL SecondPvPy(REAL** vVel, int i, int j, REAL dely) {
	return (vVel[i][j + 1] - 2 * vVel[i][j] + vVel[i][j - 1]) / square(dely);
}

REAL PpPx(REAL** pressure, int i, int j, REAL delx) {
	return (pressure[i + 1][j] - pressure[i][j]) / delx;
}

REAL PpPy(REAL** pressure, int i, int j, REAL dely) {
	return (pressure[i][j + 1] - pressure[i][j]) / dely;
}

REAL square(REAL operand) {
	return pow(operand, (REAL)2);
}