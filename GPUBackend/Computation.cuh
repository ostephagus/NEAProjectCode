#ifndef COMPUTATION_CUH
#define COMPUTATION_CUH

#include "Definitions.cuh"

#ifdef __INTELLISENSE__ // Allow intellisense to recognise cooperative groups
#define __CUDACC__
#endif // __INTELLISENSE__
#include <cooperative_groups.h>
#ifdef __INTELLISENSE__
#undef __CUDACC__
#endif // __INTELLISENSE__

namespace cg = cooperative_groups;
/// <summary>
/// Computes the max of the elements in <paramref name="sharedArray" />. Processes the number of elements equal to <paramref name="group" />'s size.
/// </summary>
/// <param name="group">The thread group of which the calling thread is a member.</param>
/// <param name="sharedArray">The array, in shared memory, to find the maximum of.</param>
__device__ void GroupMax(cg::thread_group group, volatile REAL* sharedArray);

/// <summary>
/// Computes the maximum of each column of a field. Requires xLength blocks, each of <c>field.pitch / sizeof(REAL)</c> threads, and 1 REAL's worth of shared memory per thread.
/// </summary>
/// <param name="partialMaxes">An array of length equal to the number of rows, for outputting the maxes of each column.</param>
/// <param name="field">The input field.</param>
/// <param name="yLength">The length of a column.</param>
__global__ void ComputePartialMaxes(REAL* partialMaxes, PointerWithPitch<REAL> field, int yLength);

/// <summary>
/// Computes the final max from a given array of partial maxes. Requires 1 block of <paramref name="xLength" /> threads, and 1 REAL's worth of shared memory per thread.
/// </summary>
/// <param name="max">The location to place the output.</param>
/// <param name="partialMaxes">An array of partial maxes, of size <paramref name="xLength" />.</param>
__global__ void ComputeFinalMax(REAL* max, REAL* partialMaxes, int xLength);

/// <summary>
/// Computes the max of a given field.The field's width and height must each be no larger than the max number of threads per block.
/// </summary>
/// <param name="max">The location to place the output</param>
/// <returns>An error code, or <c>cudaSuccess</c>.</returns>
cudaError_t FieldMax(REAL* max, cudaStream_t streamToUse, PointerWithPitch<REAL> field, int xLength, int yLength);

/// <summary>
/// Performs the unparallelisable part of ComputeGamma on the GPU to avoid having to copy memory to the CPU. Requires 1 thread.
/// </summary>
__global__ void FinishComputeGamma(REAL* gamma, REAL* hVelMax, REAL* vVelMax, REAL* timestep, REAL delX, REAL delY);

/// <summary>
/// Computes gamma using a reduction kernel. Handles kernel launching internally. Requires 2 streams.
/// </summary>
/// <param name="gamma">A pointer to the location to output the calculated gamma.</param>
/// <param name="streams">A pointer to an array of at least 2 streams.</param>
/// <param name="threadsPerBlock">The maximum number of threads per thread block.</param>
cudaError_t ComputeGamma(REAL* gamma, cudaStream_t* streams, int threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY);

/// <summary>
/// Performs the unparallelisable part of ComputeTimestep on the GPU to avoid having to copy memory to the CPU. Requires 1 thread.
/// </summary>
__global__ void FinishComputeTimestep(REAL* timestep, REAL* hVelMax, REAL* vVelMax, REAL delX, REAL delY, REAL reynoldsNo, REAL safetyFactor);

cudaError_t ComputeTimestep(REAL* timestep, cudaStream_t* streams, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int iMax, int jMax, REAL delX, REAL delY, REAL reynoldsNo, REAL safetyFactor);

/// <summary>
/// Computes F on the top and bottom of the simulation domain. Requires jMax threads.
/// </summary>
__global__ void ComputeFBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> F, int iMax, int jMax);

/// <summary>
/// Computes G on the left and right of the simulation domain. Requires iMax threads.
/// </summary>
__global__ void ComputeGBoundary(PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> G, int iMax, int jMax);

/// <summary>
/// Computes quantity F. Requires (iMax - 1) x (jMax) threads.
/// </summary>
__global__ void ComputeF(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> F, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY, REAL xForce, REAL* gamma, REAL reynoldsNum);

/// <summary>
/// Computes quantity G. Requires (iMax) x (jMax - 1) threads.
/// </summary>
__global__ void ComputeG(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> G, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY, REAL yForce, REAL* gamma, REAL reynoldsNum);

/// <summary>
/// Computes F and G. Handles kernel launching internally
/// </summary>
cudaError_t ComputeFG(cudaStream_t* streams, dim3 threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> F, PointerWithPitch<REAL> G, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY, REAL xForce, REAL yForce, REAL* gamma, REAL reynoldsNum);

/// <summary>
/// Computes pressure RHS. Requires iMax x jMax threads.
/// </summary>
__global__ void ComputeRHS(PointerWithPitch<REAL> F, PointerWithPitch<REAL> G, PointerWithPitch<REAL> RHS, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY);

/// <summary>
/// Computes horizontal velocity. Requires iMax x jMax threads, called by ComputeVelocities.
/// </summary>
__global__ void ComputeHVel(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> F, PointerWithPitch<REAL> pressure, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX);

/// <summary>
/// Computes vertical velocity. Requires iMax x jMax threads, called by ComputeVelocities.
/// </summary>
__global__ void ComputeVVel(PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> G, PointerWithPitch<REAL> pressure, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delY);

/// <summary>
/// Computes both vertical and horizontal velocities. Handles kernel launching internally.
/// </summary>
cudaError_t ComputeVelocities(cudaStream_t* streams, dim3 threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> F, PointerWithPitch<REAL> G, PointerWithPitch<REAL> pressure, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY);

/// <summary>
/// Computes stream function in the y direction. Requires (iMax + 1) threads.
/// </summary>
__global__ void ComputeStream(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> streamFunction, int iMax, int jMax, REAL delY);

#endif // !COMPUTATION_CUH