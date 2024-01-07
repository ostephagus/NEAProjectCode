#ifndef DISCRETE_DERIVATIVES_CUH
#define DISCRETE_DERIVATIVES_CUH

#include "Definitions.cuh"

__device__ REAL PuPx(PointerWithPitch<REAL> hVel, int i, int j, REAL delx);

__device__ REAL PvPy(PointerWithPitch<REAL> vVel, int i, int j, REAL dely);

__device__ REAL PuSquaredPx(PointerWithPitch<REAL> hVel, int i, int j, REAL delx, REAL gamma);

__device__ REAL PvSquaredPy(PointerWithPitch<REAL> vVel, int i, int j, REAL dely, REAL gamma);

__device__ REAL PuvPx(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int i, int j, REAL delX, REAL delY, REAL gamma);

__device__ REAL PuvPy(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int i, int j, REAL delX, REAL delY, REAL gamma);

__device__ REAL SecondPuPx(PointerWithPitch<REAL> hVel, int i, int j, REAL delx);

__device__ REAL SecondPuPy(PointerWithPitch<REAL> hVel, int i, int j, REAL dely);

__device__ REAL SecondPvPx(PointerWithPitch<REAL> vVel, int i, int j, REAL delx);

__device__ REAL SecondPvPy(PointerWithPitch<REAL> vVel, int i, int j, REAL dely);

__device__ REAL PpPx(PointerWithPitch<REAL> pressure, int i, int j, REAL delx);

__device__ REAL PpPy(PointerWithPitch<REAL> pressure, int i, int j, REAL dely);

__device__ REAL square(REAL operand);

#endif // !DISCRETE_DERIVATIVES_CUH