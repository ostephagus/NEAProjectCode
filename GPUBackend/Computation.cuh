#ifndef COMPUTATION_CUH
#define COMPUTATION_CUH

#include "Definitions.cuh"

/// <summary>
/// Requires iMax x jMax threads.
/// </summary>
__global__ void ComputeRHS(pointerWithPitch<REAL> F, pointerWithPitch<REAL> G, pointerWithPitch<REAL> RHS, pointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY);


#endif // !COMPUTATION_CUH