#include "Boundary.cuh"
#include "math.h"
void SetBoundaryConditions(cudaStream_t** streams, int threadsPerBlock, pointerWithPitch<REAL> hVel, pointerWithPitch<REAL> vVel, pointerWithPitch<BYTE> flags, uint2* coordinates, int coordinatesLength, int iMax, int jMax, REAL inflowVelocity, REAL chi) {
    int numBlocksTopBottom = (int)ceilf((float)iMax / threadsPerBlock);
    int numBlocksLeftRight = (int)ceilf((float)jMax / threadsPerBlock);
    if (streams == nullptr) { // Case where the device only supports sequential kernel execution
        TopBoundary<<< numBlocksTopBottom, threadsPerBlock >>>(hVel, vVel, jMax);
        BottomBoundary<<< numBlocksTopBottom, threadsPerBlock >>>(hVel, vVel);
        LeftBoundary<<< numBlocksLeftRight, threadsPerBlock >>>(hVel, vVel, inflowVelocity);
        RightBoundary<<< numBlocksLeftRight, threadsPerBlock >>>(hVel, vVel, iMax);
    }
    else { // Case where device supports at least some streams
        // Streams is an array of pointers to streams. This allows for, for example, streams[2] to reference the same stream as streams[0], in the case the device supports only 2 streams (the same would occur with streams[3] and streams[1].
        TopBoundary<<< numBlocksTopBottom, threadsPerBlock, 0, *streams[0] >>>(hVel, vVel, jMax);
        BottomBoundary<<< numBlocksTopBottom, threadsPerBlock, 0, *streams[1] >>>(hVel, vVel);
        LeftBoundary<<< numBlocksLeftRight, threadsPerBlock, 0, *streams[2] >>>(hVel, vVel, inflowVelocity);
        RightBoundary<<< numBlocksLeftRight, threadsPerBlock, 0, *streams[2] >>>(hVel, vVel, iMax);
    }
}

__global__ void TopBoundary(pointerWithPitch<REAL> hVel, pointerWithPitch<REAL> vVel, int jMax)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;

    *F_PITCHACCESS(hVel.ptr, hVel.pitch, index, jMax + 1) = *F_PITCHACCESS(hVel.ptr, hVel.pitch, index, jMax); // Copy hVel from the cell below
    *F_PITCHACCESS(vVel.ptr, vVel.pitch, index, jMax) = 0; // Set vVel along the top to 0
}

__global__ void BottomBoundary(pointerWithPitch<REAL> hVel, pointerWithPitch<REAL> vVel)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;

    *F_PITCHACCESS(hVel.ptr, hVel.pitch, index, 0) = *F_PITCHACCESS(hVel.ptr, hVel.pitch, index, 1); // Copy hVel from the cell above
    *F_PITCHACCESS(vVel.ptr, vVel.pitch, index, 0) = 0; // Set vVel along the bottom to 0
}

__global__ void LeftBoundary(pointerWithPitch<REAL> hVel, pointerWithPitch<REAL> vVel, REAL inflowVelocity)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;

    *F_PITCHACCESS(hVel.ptr, hVel.pitch, 0, index) = inflowVelocity; // Set hVel to inflow velocity on left boundary
    *F_PITCHACCESS(vVel.ptr, vVel.pitch, 0, index) = 0; // Set vVel to 0
}

__global__ void RightBoundary(pointerWithPitch<REAL> hVel, pointerWithPitch<REAL> vVel, int iMax)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;

    *F_PITCHACCESS(hVel.ptr, hVel.pitch, iMax, index) = *F_PITCHACCESS(hVel.ptr, hVel.pitch, iMax - 1, index); // Copy the velocity values from the previous cell (mass flows out at the boundary)
    *F_PITCHACCESS(vVel.ptr, vVel.pitch, iMax + 1, index) = *F_PITCHACCESS(vVel.ptr, vVel.pitch, iMax, index);
}
