#ifndef BOUNDARY_CUH
#define BOUNDARY_CUH

#include "Definitions.cuh"

/// <summary>
/// Applies top boundary conditions. Requires iMax threads.
/// </summary>
/// <param name="hVel">Pointer with pitch for horizontal velocity.</param>
/// <param name="vVel">Pointer with pitch for vertical velocity.</param>
/// <param name="jMax">The number of fluid cells in the y direction.</param>
__global__ void TopBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int jMax);

/// <summary>
/// Applies bottom boundary conditions. Requires iMax threads.
/// </summary>
/// <param name="hVel">Pointer with pitch for horizontal velocity.</param>
/// <param name="vVel">Pointer with pitch for vertical velocity.</param>
__global__ void BottomBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel);

/// <summary>
/// Applies left boundary conditions. Requires jMax threads.
/// </summary>
/// <param name="hVel">Pointer with pitch for horizontal velocity.</param>
/// <param name="vVel">Pointer with pitch for vertical velocity.</param>
/// <param name="inflowVelocity">The velocity of fluid on the left boundary</param>
__global__ void LeftBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, REAL inflowVelocity);

/// <summary>
/// Applies right boundary conditions. Requires jMax threads.
/// </summary>
/// <param name="hVel">Pointer with pitch for horizontal velocity.</param>
/// <param name="vVel">Pointer with pitch for vertical velocity.</param>
/// <param name="iMax">The number of fluid cells in the x direction.</param>
__global__ void RightBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int iMax);

// TODO: handle obstacle boundary conditions.
/// <summary>
/// Sets outer boundary conditions. Handles kernel launching internally.
/// </summary>
cudaError_t SetBoundaryConditions(cudaStream_t* streams, int threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<BYTE> flags, uint2* coordinates, int coordinatesLength, int iMax, int jMax, REAL inflowVelocity, REAL chi);

#endif // !BOUNDARY_CUH