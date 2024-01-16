#ifndef PRESSURE_COMPUTATION_CUH

#include "Definitions.cuh"

/// <summary>
/// Performs SOR iterations to solve the pressure poisson equation. Handles kernel launching internally. Requires 4 streams.
/// </summary>
/// <param name="coordinates">The coordinates of the boundary cells.</param>
/// <param name="coordinatesLength">The length of the coordinates array.</param>
/// <param name="omega">Relaxation between 0 and 2.</param>
/// <param name="residualNorm">The residual norm of the final iteration. This is an output variable, and does not need to be allocated.</param>
int Poisson(cudaStream_t* streams, dim3 threadsPerBlock, PointerWithPitch<REAL> pressure, PointerWithPitch<REAL> RHS, PointerWithPitch<BYTE> flags, uint2* coordinates, int coordinatesLength, int numFluidCells, int iMax, int jMax, int numColours, REAL delX, REAL delY, REAL residualTolerance, int minIterations, int maxIterations, REAL omega, REAL* residualNorm);

#endif // !PRESSURE_COMPUTATION_CUH
