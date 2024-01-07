#include "Computation.cuh"
#include "DiscreteDerivatives.cuh"
#include <cmath>

constexpr BYTE SELF  = 0b00010000;
constexpr BYTE NORTH = 0b00001000;
constexpr BYTE EAST  = 0b00000100;
constexpr BYTE SELFSHIFT  = 4;
constexpr BYTE NORTHSHIFT = 3;
constexpr BYTE EASTSHIFT  = 2;


__global__ void ComputeFBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> F, int iMax, int jMax) {
    int colNum = blockIdx.x * blockDim.x + threadIdx.x;
    if (colNum > jMax) return;

    *F_PITCHACCESS(F.ptr, F.pitch, 0, colNum) = *F_PITCHACCESS(hVel.ptr, hVel.pitch, 0, colNum);
    *F_PITCHACCESS(F.ptr, F.pitch, iMax, colNum) = *F_PITCHACCESS(hVel.ptr, hVel.pitch, iMax, colNum);
}

__global__ void ComputeGBoundary(PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> G, int iMax, int jMax) {
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowNum > iMax) return;

    *F_PITCHACCESS(G.ptr, G.pitch, rowNum, 0) = *F_PITCHACCESS(vVel.ptr, vVel.pitch, rowNum, 0);
    *F_PITCHACCESS(G.ptr, G.pitch, rowNum, jMax) = *F_PITCHACCESS(vVel.ptr, vVel.pitch, rowNum, jMax);
}

__global__ void ComputeF(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> F, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY, REAL xForce, REAL* gamma, REAL reynoldsNum) {
    // Branchless plan:
    // if neither east nor self are set, set F to 0: east NOR self
    // if east or self but not both are set, set F to xVel: east XOR self
    // if both are set, set F to the equation: east AND self.
    // Compute east AND self. Multiply this by the larger computation.
    // Compute east OR self. Multiply this by xVel.
    // Add them.
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x;
    int colNum = blockIdx.y * blockDim.y + threadIdx.y;

    if (rowNum >= iMax) return;
    if (colNum >= jMax) return;

    int selfBit = (*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & SELF) >> SELFSHIFT;
    int eastBit = (*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & EAST) >> EASTSHIFT;

    *F_PITCHACCESS(F.ptr, F.pitch, rowNum, colNum) = 
        *F_PITCHACCESS(hVel.ptr, hVel.pitch, rowNum, colNum) * (selfBit | eastBit) // self OR east is either a fluid or boundary cell - these cells need xVel.
        + *timestep * (1 / reynoldsNum * (SecondPuPx(hVel, rowNum, colNum, delX) + SecondPuPy(hVel, rowNum, colNum, delY)) - PuSquaredPx(hVel, rowNum, colNum, delX, *gamma) - PuvPy(hVel, vVel, rowNum, colNum, delX, delY, *gamma) + xForce)
        * (selfBit & eastBit); // These are only fluid cells, perform the computation
}


__global__ void ComputeG(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> G, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY, REAL yForce, REAL* gamma, REAL reynoldsNum) {
    // Branchless plan:
    // if neither east nor self are set, set F to 0: east NOR self
    // if east or self but not both are set, set F to xVel: east XOR self
    // if both are set, set F to the equation: east AND self.
    // Compute east AND self. Multiply this by the larger computation.
    // Compute east OR self. Multiply this by xVel.
    // Add them.
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x;
    int colNum = blockIdx.y * blockDim.y + threadIdx.y;

    if (rowNum >= iMax) return;
    if (colNum >= jMax) return;

    int selfBit = (*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & SELF) >> SELFSHIFT;
    int northBit = (*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & NORTH) >> NORTHSHIFT;

    *F_PITCHACCESS(G.ptr, G.pitch, rowNum, colNum) =
        *F_PITCHACCESS(vVel.ptr, vVel.pitch, rowNum, colNum) * (selfBit | northBit) // self OR east is either a fluid or boundary cell - these cells need xVel.
        + *timestep * (1 / reynoldsNum * (SecondPvPx(vVel, rowNum, colNum, delX) + SecondPvPy(vVel, rowNum, colNum, delY)) - PuvPx(hVel, vVel, rowNum, colNum, delX, delY, *gamma) - PvSquaredPy(vVel, rowNum, colNum, delY, *gamma) + yForce)
        * (selfBit & northBit); // These are only fluid cells, perform the computation
}

cudaError_t ComputeFG(cudaStream_t* streams, dim3 threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> F, PointerWithPitch<REAL> G, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY, REAL xForce, REAL yForce, REAL* gamma, REAL reynoldsNum) {
    dim3 numBlocks((int)ceilf((float)(iMax - 1) / threadsPerBlock.x), (int)ceilf((float)(jMax - 1) / threadsPerBlock.y));

    int threadsPerBlockFlat = threadsPerBlock.x * threadsPerBlock.y;
    int numBlocksIMax = (int)ceilf((float)iMax / threadsPerBlockFlat);
    int numBlocksJMax = (int)ceilf((float)jMax / threadsPerBlockFlat);

    ComputeF<<<numBlocks, threadsPerBlock, 0, streams[0]>>>(hVel, vVel, F, flags, iMax, jMax, timestep, delX, delY, xForce, gamma, reynoldsNum); // Launch the kernels in separate streams, to be concurrently executed if the GPU is able to.
    ComputeG<<<numBlocks, threadsPerBlock, 0, streams[1]>>>(hVel, vVel, G, flags, iMax, jMax, timestep, delX, delY, yForce, gamma, reynoldsNum);

    ComputeFBoundary<<<numBlocksJMax, threadsPerBlockFlat, 0, streams[2]>>>(hVel, F, iMax, jMax);
    ComputeGBoundary<<<numBlocksIMax, threadsPerBlockFlat, 0, streams[3]>>>(vVel, G, iMax, jMax);

    return cudaDeviceSynchronize();
}

__global__ void ComputeRHS(PointerWithPitch<REAL> F, PointerWithPitch<REAL> G, PointerWithPitch<REAL> RHS, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY) {
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int colNum = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (rowNum > iMax) return;
    if (colNum > jMax) return;
    
    *F_PITCHACCESS(RHS.ptr, RHS.pitch, rowNum, colNum) = 
        ((*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & SELF) >> SELFSHIFT) // Sets the entire expression to 0 if the cell is not fluid
        * (1 / *timestep) * (((*F_PITCHACCESS(F.ptr, F.pitch, rowNum, colNum) - *F_PITCHACCESS(F.ptr, F.pitch, rowNum - 1, colNum)) / delX) + ((*F_PITCHACCESS(G.ptr, G.pitch, rowNum, colNum) - *F_PITCHACCESS(G.ptr, G.pitch, rowNum, colNum - 1)) / delY));
}

__global__ void ComputeHVel(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> F, PointerWithPitch<REAL> pressure, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX)
{
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int colNum = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (rowNum > iMax) return; // Bounds checking
    if (colNum > jMax) return;

    *F_PITCHACCESS(hVel.ptr, hVel.pitch, rowNum, colNum) =
        ((*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & SELF) >> SELFSHIFT) // Equal to 0 if the cell is not a fluid cell
        * ((*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & EAST) >> EASTSHIFT) // Equal to 0 if the cell has an obstacle cell next to it in +ve x direction (east)
        * (*F_PITCHACCESS(F.ptr, F.pitch, rowNum, colNum) - (*timestep / delX) * (*F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum + 1, colNum) - *F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, colNum)));
}

__global__ void ComputeVVel(PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> G, PointerWithPitch<REAL> pressure, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delY)
{
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int colNum = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (rowNum > iMax) return; // Bounds checking
    if (colNum > jMax) return;

    *F_PITCHACCESS(vVel.ptr, vVel.pitch, rowNum, colNum) =
        ((*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & SELF) >> SELFSHIFT) // Equal to 0 if the cell is not a fluid cell
        * ((*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & NORTH) >> NORTHSHIFT) // Equal to 0 if the cell has an obstacle cell next to it in +ve y direction (north)
        * (*F_PITCHACCESS(G.ptr, G.pitch, rowNum, colNum) - (*timestep / delY) * (*F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, colNum + 1) - *F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, colNum)));
}

cudaError_t ComputeVelocities(cudaStream_t* streams, dim3 threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> F, PointerWithPitch<REAL> G, PointerWithPitch<REAL> pressure, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY)
{
    dim3 numBlocks(iMax / threadsPerBlock.x, jMax / threadsPerBlock.y);
    ComputeHVel << <numBlocks, threadsPerBlock, 0, streams[0] >> > (hVel, F, pressure, flags, iMax, jMax, timestep, delX); // Launch the kernels in separate streams, to be concurrently executed if the GPU is able to.
    ComputeVVel << <numBlocks, threadsPerBlock, 0, streams[1] >> > (vVel, G, pressure, flags, iMax, jMax, timestep, delY);
    return cudaDeviceSynchronize();
}

__global__ void ComputeStream(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> streamFunction, int iMax, int jMax, REAL delY)
{
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowNum > iMax) return;

    *F_PITCHACCESS(streamFunction.ptr, streamFunction.pitch, rowNum, 0) = 0; // Stream function boundary condition
    for (int colNum = 1; colNum <= jMax; colNum++) {
        *F_PITCHACCESS(streamFunction.ptr, streamFunction.pitch, rowNum, colNum) = *F_PITCHACCESS(streamFunction.ptr, streamFunction.pitch, rowNum, colNum - 1) + *F_PITCHACCESS(hVel.ptr, hVel.pitch, rowNum, colNum) * delY;
    }
}
