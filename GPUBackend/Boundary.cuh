#ifndef BOUNDARY_CUH
#define BOUNDARY_CUH

#include "Definitions.cuh"

/// <summary>
/// Sets the flags for each cell based on the value of surrounding cells. Requires iMax x jMax threads.
/// </summary>
/// <param name="obstacles">A boolean array indicating whether each cell is obstacle or fluid.</param>
/// <param name="flags">A BYTE array to hold the flags.</param>
__global__ void SetFlags(PointerWithPitch<bool> obstacles, PointerWithPitch<BYTE> flags, int iMax, int jMax);

/// <summary>
/// Applies top boundary conditions. Requires iMax threads.
/// </summary>
/// <param name="hVel">Pointer with pitch for horizontal velocity.</param>
/// <param name="vVel">Pointer with pitch for vertical velocity.</param>
/// <param name="jMax">The number of fluid cells in the y direction.</param>
__global__ void TopBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int iMax, int jMax);

/// <summary>
/// Applies bottom boundary conditions. Requires iMax threads.
/// </summary>
/// <param name="hVel">Pointer with pitch for horizontal velocity.</param>
/// <param name="vVel">Pointer with pitch for vertical velocity.</param>
__global__ void BottomBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int iMax, int jMax);

/// <summary>
/// Applies left boundary conditions. Requires jMax threads.
/// </summary>
/// <param name="hVel">Pointer with pitch for horizontal velocity.</param>
/// <param name="vVel">Pointer with pitch for vertical velocity.</param>
/// <param name="inflowVelocity">The velocity of fluid on the left boundary</param>
__global__ void LeftBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int iMax, int jMax, REAL inflowVelocity);

/// <summary>
/// Applies right boundary conditions. Requires jMax threads.
/// </summary>
/// <param name="hVel">Pointer with pitch for horizontal velocity.</param>
/// <param name="vVel">Pointer with pitch for vertical velocity.</param>
/// <param name="iMax">The number of fluid cells in the x direction.</param>
__global__ void RightBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int iMax, int jMax);

/// <summary>
/// Applies boundary conditions on obstacles. Requires <paramref name="coordinatesLength" /> threads.
/// </summary>
__global__ void ObstacleBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<BYTE> flags, uint2* coordinates, int coordinatesLength, REAL chi);

/// <summary>
/// Sets boundary conditions. Handles kernel launching internally. Requires 4 streams.
/// </summary>
cudaError_t SetBoundaryConditions(cudaStream_t* streams, int threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<BYTE> flags, uint2* coordinates, int coordinatesLength, int iMax, int jMax, REAL inflowVelocity, REAL chi);

void FindBoundaryCells(BYTE** flags, uint2*& coordinates, int& coordinatesLength, int iMax, int jMax);

#endif // !BOUNDARY_CUH