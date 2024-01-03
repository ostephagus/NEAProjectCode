#include "Boundary.cuh"
#include "math.h"
void SetBoundaryConditions(cudaStream_t* streams, int threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<BYTE> flags, uint2* coordinates, int coordinatesLength, int iMax, int jMax, REAL inflowVelocity, REAL chi) {
    int numBlocksTopBottom = (int)ceilf((float)iMax / threadsPerBlock);
    int numBlocksLeftRight = (int)ceilf((float)jMax / threadsPerBlock);
    
    TopBoundary<<< numBlocksTopBottom, threadsPerBlock, 0, streams[0] >>>(hVel, vVel, jMax);
    BottomBoundary<<< numBlocksTopBottom, threadsPerBlock, 0, streams[1] >>>(hVel, vVel);
    LeftBoundary<<< numBlocksLeftRight, threadsPerBlock, 0, streams[2] >>>(hVel, vVel, inflowVelocity);
    RightBoundary<<< numBlocksLeftRight, threadsPerBlock, 0, streams[3] >>>(hVel, vVel, iMax);
    cudaDeviceSynchronize();
}

__global__ void TopBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int jMax)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;

    *F_PITCHACCESS(hVel.ptr, hVel.pitch, index, jMax + 1) = *F_PITCHACCESS(hVel.ptr, hVel.pitch, index, jMax); // Copy hVel from the cell below
    *F_PITCHACCESS(vVel.ptr, vVel.pitch, index, jMax) = 0; // Set vVel along the top to 0
}

__global__ void BottomBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;

    *F_PITCHACCESS(hVel.ptr, hVel.pitch, index, 0) = *F_PITCHACCESS(hVel.ptr, hVel.pitch, index, 1); // Copy hVel from the cell above
    *F_PITCHACCESS(vVel.ptr, vVel.pitch, index, 0) = 0; // Set vVel along the bottom to 0
}

__global__ void LeftBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, REAL inflowVelocity)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;

    *F_PITCHACCESS(hVel.ptr, hVel.pitch, 0, index) = inflowVelocity; // Set hVel to inflow velocity on left boundary
    *F_PITCHACCESS(vVel.ptr, vVel.pitch, 0, index) = 0; // Set vVel to 0
}

__global__ void RightBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int iMax)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;

    *F_PITCHACCESS(hVel.ptr, hVel.pitch, iMax, index) = *F_PITCHACCESS(hVel.ptr, hVel.pitch, iMax - 1, index); // Copy the velocity values from the previous cell (mass flows out at the boundary)
    *F_PITCHACCESS(vVel.ptr, vVel.pitch, iMax + 1, index) = *F_PITCHACCESS(vVel.ptr, vVel.pitch, iMax, index);
}
