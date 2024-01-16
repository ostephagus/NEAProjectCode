#ifndef COMPUTATION_CUH
#define COMPUTATION_CUH

#include "Definitions.cuh"

/// <summary>
/// Computes gamma using a reduction kernel. Handles kernel launching internally. Requires 2 streams.
/// </summary>
/// <param name="gamma">A pointer to the location to output the calculated gamma.</param>
/// <param name="streams">A pointer to an array of at least 2 streams.</param>
/// <param name="threadsPerBlock">The maximum number of threads per thread block.</param>
cudaError_t ComputeGamma(REAL* gamma, cudaStream_t* streams, int threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY);

cudaError_t ComputeTimestep(REAL* timestep, cudaStream_t* streams, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int iMax, int jMax, REAL delX, REAL delY, REAL reynoldsNo, REAL safetyFactor);

/// <summary>
/// Computes F and G. Handles kernel launching internally. Requires 4 threads.
/// </summary>
cudaError_t ComputeFG(cudaStream_t* streams, dim3 threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> F, PointerWithPitch<REAL> G, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY, REAL xForce, REAL yForce, REAL* gamma, REAL reynoldsNum);

/// <summary>
/// Computes pressure RHS. Requires iMax x jMax threads.
/// </summary>
__global__ void ComputeRHS(PointerWithPitch<REAL> F, PointerWithPitch<REAL> G, PointerWithPitch<REAL> RHS, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY);

/// <summary>
/// Computes both vertical and horizontal velocities. Handles kernel launching internally.
/// </summary>
cudaError_t ComputeVelocities(cudaStream_t* streams, dim3 threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> F, PointerWithPitch<REAL> G, PointerWithPitch<REAL> pressure, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY);

/// <summary>
/// Computes stream function in the y direction. Requires (iMax + 1) threads.
/// </summary>
__global__ void ComputeStream(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> streamFunction, int iMax, int jMax, REAL delY);

#endif // !COMPUTATION_CUH