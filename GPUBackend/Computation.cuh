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

__device__ void GroupMax(cg::thread_group group, volatile float* sharedArray, int arrayLength);

__global__ void ComputeMaxesSingleBlock(float* max, float* globalArray, int arrayLength);

__global__ void ComputeMaxesMultiBlock(float* maxesArray, float* globalArray, int arrayLength);

cudaError_t ArrayMax(float* max, int numThreads, float* values, int arrayLength);

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