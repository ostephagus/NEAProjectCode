#ifndef DRAG_CUH
#define DRAG_CUH

#include "Definitions.cuh"

__device__ REAL Magnitude(REAL x, REAL y);

__device__ REAL Dot(REAL2 left, REAL2 right);

__device__ REAL2 GetUnitVector(REAL2 vector);

__device__ REAL PVPd(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, REAL distance, int iStart, int jStart, int iExtended, int jExtended);

__device__ REAL ComputeWallShear(REAL2* shearUnitVector, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, REAL2 unitNormal, int i, int j, REAL delX, REAL delY, REAL viscosity);

__global__ void ComputeViscousDrag(REAL* integrandArray, DragCoordinate* viscosityCoordinates, int viscosityCoordinatesLength, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int iMax, int jMax, REAL2 fluidVector, REAL delX, REAL delY, REAL viscosity);

__global__ void ComputeBaselinePressure(REAL* baselinePressure, PointerWithPitch<REAL> pressure, int iMax, int jMax);

__device__ REAL PressureIntegrand(REAL pressure, REAL baselinePressure, REAL2 unitNormal, REAL2 fluidVector);

__global__ void ComputePressureDrag(REAL* integrandArray, DragCoordinate* pressureCoordinates, int pressureCoordinatesLength, PointerWithPitch<REAL> pressure, int iMax, int jMax, REAL delX, REAL delY, REAL2 fluidVector, REAL* baselinePressure);
#endif // !DRAG_CUH