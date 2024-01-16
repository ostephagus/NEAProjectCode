#include "Computation.cuh"
#include "DiscreteDerivatives.cuh"
#include "ReductionKernels.cuh"
#include <cmath>

/// <summary>
/// Performs the unparallelisable part of ComputeGamma on the GPU to avoid having to copy memory to the CPU. Requires 1 thread.
/// </summary>
__global__ void FinishComputeGamma(REAL* gamma, REAL* hVelMax, REAL* vVelMax, REAL* timestep, REAL delX, REAL delY) {
    REAL horizontalComponent = *hVelMax * (*timestep / delX);
    REAL verticalComponent = *vVelMax * (*timestep / delY);

    if (horizontalComponent > verticalComponent) {
        *gamma = horizontalComponent;
    }
    else {
        *gamma = verticalComponent;
    }
}

cudaError_t ComputeGamma(REAL* gamma, cudaStream_t* streams, int threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY) {
    cudaError_t retVal;
    REAL* hVelMax;
    retVal = cudaMalloc(&hVelMax, sizeof(REAL));
    if (retVal != cudaSuccess) goto free;

    REAL* vVelMax;
    retVal = cudaMalloc(&vVelMax, sizeof(REAL));
    if (retVal != cudaSuccess) goto free;

    FieldMax(hVelMax, streams[0], hVel, iMax + 2, jMax + 2);

    retVal = cudaStreamSynchronize(streams[0]);
    if (retVal != cudaSuccess) goto free;

    FieldMax(vVelMax, streams[1], vVel, iMax + 2, jMax + 2);

    retVal = cudaStreamSynchronize(streams[1]);
    if (retVal != cudaSuccess) goto free;

    FinishComputeGamma KERNEL_ARGS(1, 1, 0, streams[0]) (gamma, hVelMax, vVelMax, timestep, delX, delY);

    free:
    cudaFree(hVelMax);
    cudaFree(vVelMax);
    return retVal;
}

/// <summary>
/// Performs the unparallelisable part of ComputeTimestep on the GPU to avoid having to copy memory to the CPU. Requires 1 thread.
/// </summary>
__global__ void FinishComputeTimestep(REAL* timestep, REAL* hVelMax, REAL* vVelMax, REAL delX, REAL delY, REAL reynoldsNo, REAL safetyFactor)
{
    REAL inverseSquareRestriction = (REAL)0.5 * reynoldsNo * (1 / square(delX) + 1 / square(delY));
    REAL xTravelRestriction = delX / *hVelMax;
    REAL yTravelRestriction = delY / *vVelMax;

    REAL smallestRestriction = inverseSquareRestriction; // Choose the smallest restriction
    if (xTravelRestriction < smallestRestriction) {
        smallestRestriction = xTravelRestriction;
    }
    if (yTravelRestriction < smallestRestriction) {
        smallestRestriction = yTravelRestriction;
    }
    *timestep = safetyFactor * smallestRestriction;
}

cudaError_t ComputeTimestep(REAL* timestep, cudaStream_t* streams, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int iMax, int jMax, REAL delX, REAL delY, REAL reynoldsNo, REAL safetyFactor)
{
    cudaError_t retVal;
    REAL* hVelMax;
    retVal = cudaMalloc(&hVelMax, sizeof(REAL));
    if (retVal != cudaSuccess) goto free;

    REAL* vVelMax;
    retVal = cudaMalloc(&vVelMax, sizeof(REAL));
    if (retVal != cudaSuccess) goto free;

    FieldMax(hVelMax, streams[0], hVel, iMax + 2, jMax + 2);

    retVal = cudaStreamSynchronize(streams[0]);
    if (retVal != cudaSuccess) goto free;

    FieldMax(vVelMax, streams[1], vVel, iMax + 2, jMax + 2);

    retVal = cudaStreamSynchronize(streams[1]);
    if (retVal != cudaSuccess) goto free;

    FinishComputeTimestep KERNEL_ARGS(1, 1, 0, streams[0]) (timestep, hVelMax, vVelMax, delX, delY, reynoldsNo, safetyFactor);

free:
    cudaFree(hVelMax);
    cudaFree(vVelMax);
    return retVal;
}

/// <summary>
/// Computes F on the top and bottom of the simulation domain. Requires jMax threads.
/// </summary>
__global__ void ComputeFBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> F, int iMax, int jMax) {
    int colNum = blockIdx.x * blockDim.x + threadIdx.x;
    if (colNum > jMax) return;

    F_PITCHACCESS(F.ptr, F.pitch, 0, colNum) = F_PITCHACCESS(hVel.ptr, hVel.pitch, 0, colNum);
    F_PITCHACCESS(F.ptr, F.pitch, iMax, colNum) = F_PITCHACCESS(hVel.ptr, hVel.pitch, iMax, colNum);
}

/// <summary>
/// Computes G on the left and right of the simulation domain. Requires iMax threads.
/// </summary>
__global__ void ComputeGBoundary(PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> G, int iMax, int jMax) {
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowNum > iMax) return;

    F_PITCHACCESS(G.ptr, G.pitch, rowNum, 0) = F_PITCHACCESS(vVel.ptr, vVel.pitch, rowNum, 0);
    F_PITCHACCESS(G.ptr, G.pitch, rowNum, jMax) = F_PITCHACCESS(vVel.ptr, vVel.pitch, rowNum, jMax);
}

/// <summary>
/// Computes quantity F. Requires (iMax - 1) x (jMax) threads.
/// </summary>
__global__ void ComputeF(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> F, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY, REAL xForce, REAL* gamma, REAL reynoldsNum) {
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int colNum = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (rowNum >= iMax) return;
    if (colNum > jMax) return;

    int selfBit = (B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & SELF) >> SELFSHIFT; // SELF bit of the cell's flag
    int eastBit = (B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & EAST) >> EASTSHIFT; // EAST bit of the cell's flag
    
    F_PITCHACCESS(F.ptr, F.pitch, rowNum, colNum) =
        F_PITCHACCESS(hVel.ptr, hVel.pitch, rowNum, colNum) * (selfBit | eastBit) // For boundary cells or fluid cells, add hVel
        + *timestep * (1 / reynoldsNum * (SecondPuPx(hVel, rowNum, colNum, delX) + SecondPuPy(hVel, rowNum, colNum, delY)) - PuSquaredPx(hVel, rowNum, colNum, delX, *gamma) - PuvPy(hVel, vVel, rowNum, colNum, delX, delY, *gamma) + xForce) * (selfBit & eastBit); // For fluid cells only, perform the computation. Obstacle cells without an eastern boundary are set to 0.
}

/// <summary>
/// Computes quantity G. Requires (iMax) x (jMax - 1) threads.
/// </summary>
__global__ void ComputeG(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> G, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY, REAL yForce, REAL* gamma, REAL reynoldsNum) {
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int colNum = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (rowNum > iMax) return;
    if (colNum >= jMax) return;

    int selfBit = (B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & SELF) >> SELFSHIFT;    // SELF bit of the cell's flag
    int northBit = (B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & NORTH) >> NORTHSHIFT; // NORTH bit of the cell's flag

    F_PITCHACCESS(G.ptr, G.pitch, rowNum, colNum) =
        F_PITCHACCESS(vVel.ptr, vVel.pitch, rowNum, colNum) * (selfBit | northBit) // For boundary cells or fluid cells, add vVel
        + *timestep * (1 / reynoldsNum * (SecondPvPx(vVel, rowNum, colNum, delX) + SecondPvPy(vVel, rowNum, colNum, delY)) - PuvPx(hVel, vVel, rowNum, colNum, delX, delY, *gamma) - PvSquaredPy(vVel, rowNum, colNum, delY, *gamma) + yForce) * (selfBit & northBit); // For fluid cells only, perform the computation. Obstacle cells without a northern boundary are set to 0.
}

cudaError_t ComputeFG(cudaStream_t* streams, dim3 threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> F, PointerWithPitch<REAL> G, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY, REAL xForce, REAL yForce, REAL* gamma, REAL reynoldsNum) {
    dim3 numBlocksF(INT_DIVIDE_ROUND_UP(iMax - 1, threadsPerBlock.x), INT_DIVIDE_ROUND_UP(jMax, threadsPerBlock.y));
    dim3 numBlocksG(INT_DIVIDE_ROUND_UP(iMax, threadsPerBlock.x), INT_DIVIDE_ROUND_UP(jMax - 1, threadsPerBlock.y));

    int threadsPerBlockFlat = threadsPerBlock.x * threadsPerBlock.y;
    int numBlocksIMax = INT_DIVIDE_ROUND_UP(iMax, threadsPerBlockFlat);
    int numBlocksJMax = INT_DIVIDE_ROUND_UP(jMax, threadsPerBlockFlat);

    ComputeF KERNEL_ARGS(numBlocksF, threadsPerBlock, 0, streams[0]) (hVel, vVel, F, flags, iMax, jMax, timestep, delX, delY, xForce, gamma, reynoldsNum); // Launch the kernels in separate streams, to be concurrently executed if the GPU is able to.
    ComputeG KERNEL_ARGS(numBlocksG, threadsPerBlock, 0, streams[1]) (hVel, vVel, G, flags, iMax, jMax, timestep, delX, delY, yForce, gamma, reynoldsNum);

    ComputeFBoundary KERNEL_ARGS(numBlocksJMax, threadsPerBlockFlat, 0, streams[2]) (hVel, F, iMax, jMax);
    ComputeGBoundary KERNEL_ARGS(numBlocksIMax, threadsPerBlockFlat, 0, streams[3]) (vVel, G, iMax, jMax);

    return cudaDeviceSynchronize();
}

__global__ void ComputeRHS(PointerWithPitch<REAL> F, PointerWithPitch<REAL> G, PointerWithPitch<REAL> RHS, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY) {
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int colNum = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (rowNum > iMax) return;
    if (colNum > jMax) return;
    
    F_PITCHACCESS(RHS.ptr, RHS.pitch, rowNum, colNum) = 
        ((B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & SELF) >> SELFSHIFT) // Sets the entire expression to 0 if the cell is not fluid
        * (1 / *timestep) * (((F_PITCHACCESS(F.ptr, F.pitch, rowNum, colNum) - F_PITCHACCESS(F.ptr, F.pitch, rowNum - 1, colNum)) / delX) + ((F_PITCHACCESS(G.ptr, G.pitch, rowNum, colNum) - F_PITCHACCESS(G.ptr, G.pitch, rowNum, colNum - 1)) / delY));
}

/// <summary>
/// Computes horizontal velocity. Requires iMax x jMax threads, called by ComputeVelocities.
/// </summary>
__global__ void ComputeHVel(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> F, PointerWithPitch<REAL> pressure, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX)
{
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int colNum = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (rowNum > iMax) return; // Bounds checking
    if (colNum > jMax) return;

    F_PITCHACCESS(hVel.ptr, hVel.pitch, rowNum, colNum) =
        ((B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & SELF) >> SELFSHIFT) // Equal to 0 if the cell is not a fluid cell
        * ((B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & EAST) >> EASTSHIFT) // Equal to 0 if the cell has an obstacle cell next to it in +ve x direction (east)
        * (F_PITCHACCESS(F.ptr, F.pitch, rowNum, colNum) - (*timestep / delX) * (F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum + 1, colNum) - F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, colNum)));
}

/// <summary>
/// Computes vertical velocity. Requires iMax x jMax threads, called by ComputeVelocities.
/// </summary>
__global__ void ComputeVVel(PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> G, PointerWithPitch<REAL> pressure, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delY)
{
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int colNum = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (rowNum > iMax) return; // Bounds checking
    if (colNum > jMax) return;

    F_PITCHACCESS(vVel.ptr, vVel.pitch, rowNum, colNum) =
        ((B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & SELF) >> SELFSHIFT) // Equal to 0 if the cell is not a fluid cell
        * ((B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & NORTH) >> NORTHSHIFT) // Equal to 0 if the cell has an obstacle cell next to it in +ve y direction (north)
        * (F_PITCHACCESS(G.ptr, G.pitch, rowNum, colNum) - (*timestep / delY) * (F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, colNum + 1) - F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, colNum)));
}

cudaError_t ComputeVelocities(cudaStream_t* streams, dim3 threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> F, PointerWithPitch<REAL> G, PointerWithPitch<REAL> pressure, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY)
{
    dim3 numBlocks(INT_DIVIDE_ROUND_UP(iMax, threadsPerBlock.x), INT_DIVIDE_ROUND_UP(jMax, threadsPerBlock.y));
    ComputeHVel KERNEL_ARGS(numBlocks, threadsPerBlock, 0, streams[0]) (hVel, F, pressure, flags, iMax, jMax, timestep, delX); // Launch the kernels in separate streams, to be concurrently executed if the GPU is able to.
    ComputeVVel KERNEL_ARGS(numBlocks, threadsPerBlock, 0, streams[1]) (vVel, G, pressure, flags, iMax, jMax, timestep, delY);
    return cudaDeviceSynchronize();
}

__global__ void ComputeStream(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> streamFunction, int iMax, int jMax, REAL delY)
{
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowNum > iMax) return;

    F_PITCHACCESS(streamFunction.ptr, streamFunction.pitch, rowNum, 0) = 0; // Stream function boundary condition
    for (int colNum = 1; colNum <= jMax; colNum++) {
        F_PITCHACCESS(streamFunction.ptr, streamFunction.pitch, rowNum, colNum) = F_PITCHACCESS(streamFunction.ptr, streamFunction.pitch, rowNum, colNum - 1) + F_PITCHACCESS(hVel.ptr, hVel.pitch, rowNum, colNum) * delY;
    }
}
