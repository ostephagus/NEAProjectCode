#include "DiscreteDerivatives.cuh"
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

__device__ REAL PuPx(PointerWithPitch<REAL> hVel, int i, int j, REAL delx) { // NOTE: P here is used to represent the partial operator, so PuPx should be read "partial u by partial x"
	return (*F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j) - *F_PITCHACCESS(hVel.ptr, hVel.pitch, i - 1, j)) / delx;
}

__device__ REAL PvPy(PointerWithPitch<REAL> vVel, int i, int j, REAL dely) {
	return (*F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j) - *F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j - 1)) / dely;
}

__device__ REAL PuSquaredPx(PointerWithPitch<REAL> hVel, int i, int j, REAL delx, REAL gamma) {
	REAL forwardAverage = (*F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j) + *F_PITCHACCESS(hVel.ptr, hVel.pitch, i + 1, j)) / 2;
	REAL backwardAverage = (*F_PITCHACCESS(hVel.ptr, hVel.pitch, i - 1, j) + *F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j)) / 2;

	REAL forwardDifference = (*F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j) - *F_PITCHACCESS(hVel.ptr, hVel.pitch, i + 1, j)) / 2;
	REAL backwardDifference = (*F_PITCHACCESS(hVel.ptr, hVel.pitch, i - 1, j) - *F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j)) / 2;

	REAL nonDonorTerm = (1 / delx) * (square(forwardAverage) - square(backwardAverage));
	REAL donorTerm = (gamma / delx) * ((abs(forwardAverage) * forwardDifference) - (abs(backwardAverage) * backwardDifference));

	return nonDonorTerm + donorTerm;
}

__device__ REAL PvSquaredPy(PointerWithPitch<REAL> vVel, int i, int j, REAL dely, REAL gamma) {
	REAL forwardAverage = (*F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j) + *F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j + 1)) / 2;
	REAL backwardAverage = (*F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j - 1) + *F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j)) / 2;

	REAL forwardDifference = (*F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j) - *F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j + 1)) / 2;
	REAL backwardDifference = (*F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j - 1) - *F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j)) / 2;

	REAL nonDonorTerm = (1 / dely) * (square(forwardAverage) - square(backwardAverage));
	REAL donorTerm = (gamma / dely) * ((abs(forwardAverage) * forwardDifference) - (abs(backwardAverage) * backwardDifference));
	return nonDonorTerm + donorTerm;
}

__device__ REAL PuvPx(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int i, int j, REAL delX, REAL delY, REAL gamma) {
	REAL jForwardAverageU = (*F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j) + *F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j + 1)) / 2;
	REAL iForwardAverageV = (*F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j) + *F_PITCHACCESS(vVel.ptr, vVel.pitch, i + 1, j)) / 2;
	REAL iBackwardAverageV = (*F_PITCHACCESS(vVel.ptr, vVel.pitch, i - 1, j) + *F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j)) / 2;

	REAL jForwardAverageUDownshift = (*F_PITCHACCESS(hVel.ptr, hVel.pitch, i - 1, j) + *F_PITCHACCESS(hVel.ptr, hVel.pitch, i - 1, j + 1)) / 2;

	REAL iForwardDifferenceV = (*F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j) - *F_PITCHACCESS(vVel.ptr, vVel.pitch, i + 1, j)) / 2;
	REAL iBackwardDifferenceV = (*F_PITCHACCESS(vVel.ptr, vVel.pitch, i - 1, j) - *F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j)) / 2;

	REAL nonDonorTerm = (1 / delX) * ((jForwardAverageU * iForwardAverageV) - (jForwardAverageUDownshift * iBackwardAverageV));
	REAL donorTerm = (gamma / delX) * ((abs(jForwardAverageU) * iForwardDifferenceV) - (abs(jForwardAverageUDownshift) * iBackwardDifferenceV));
	return nonDonorTerm + donorTerm;
}

__device__ REAL PuvPy(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int i, int j, REAL delX, REAL delY, REAL gamma) {
	REAL iForwardAverageV = (*F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j) + *F_PITCHACCESS(vVel.ptr, vVel.pitch, i + 1, j)) / 2;
	REAL jForwardAverageU = (*F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j) + *F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j + 1)) / 2;
	REAL jBackwardAverageU = (*F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j - 1) + *F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j)) / 2;

	REAL iForwardAverageVDownshift = (*F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j - 1) + *F_PITCHACCESS(vVel.ptr, vVel.pitch, i + 1, j - 1)) / 2;

	REAL jForwardDifferenceU = (*F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j) - *F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j + 1)) / 2;
	REAL jBackwardDifferenceU = (*F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j - 1) - *F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j)) / 2;

	REAL nonDonorTerm = (1 / delY) * ((iForwardAverageV * jForwardAverageU) - (iForwardAverageVDownshift * jBackwardAverageU));
	REAL donorTerm = (gamma / delY) * ((abs(iForwardAverageV) * jForwardDifferenceU) - (abs(iForwardAverageVDownshift) * jBackwardDifferenceU));

	return nonDonorTerm + donorTerm;
}

__device__ REAL SecondPuPx(PointerWithPitch<REAL> hVel, int i, int j, REAL delx) {
	return (*F_PITCHACCESS(hVel.ptr, hVel.pitch, i + 1, j) - 2 * *F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j) + *F_PITCHACCESS(hVel.ptr, hVel.pitch, i - 1, j)) / square(delx);
}

__device__ REAL SecondPuPy(PointerWithPitch<REAL> hVel, int i, int j, REAL dely) {
	return (*F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j + 1) - 2 * *F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j) + *F_PITCHACCESS(hVel.ptr, hVel.pitch, i, j - 1)) / square(dely);
}

__device__ REAL SecondPvPx(PointerWithPitch<REAL> vVel, int i, int j, REAL delx) {
	return (*F_PITCHACCESS(vVel.ptr, vVel.pitch, i + 1, j) - 2 * *F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j) + *F_PITCHACCESS(vVel.ptr, vVel.pitch, i - 1, j)) / square(delx);
}

__device__ REAL SecondPvPy(PointerWithPitch<REAL> vVel, int i, int j, REAL dely) {
	return (*F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j + 1) - 2 * *F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j) + *F_PITCHACCESS(vVel.ptr, vVel.pitch, i, j - 1)) / square(dely);
}

__device__ REAL PpPx(PointerWithPitch<REAL> pressure, int i, int j, REAL delx) {
	return (*F_PITCHACCESS(pressure.ptr, pressure.pitch, i + 1, j) - *F_PITCHACCESS(pressure.ptr, pressure.pitch, i, j)) / delx;
}

__device__ REAL PpPy(PointerWithPitch<REAL> pressure, int i, int j, REAL dely) {
	return (*F_PITCHACCESS(pressure.ptr, pressure.pitch, i, j + 1) - *F_PITCHACCESS(pressure.ptr, pressure.pitch, i, j)) / dely;
}

__device__ REAL square(REAL operand) {
	return operand * operand;
}