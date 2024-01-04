#include "Computation.cuh"
#include <cmath>

constexpr BYTE SELF  = 0b00010000;
constexpr BYTE NORTH = 0b00001000;
constexpr BYTE EAST  = 0b00000100;
constexpr BYTE SOUTH = 0b00000010;
constexpr BYTE WEST  = 0b00000001;
constexpr BYTE SELFSHIFT  = 4;
constexpr BYTE NORTHSHIFT = 3;
constexpr BYTE EASTSHIFT  = 2;
constexpr BYTE SOUTHSHIFT = 1;
constexpr BYTE WESTSHIFT  = 0;

__global__ void ComputeF(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> F, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timeStep, REAL delX, REAL xForce, REAL* gamma, REAL reynoldsNum) {
    // Branchless plan:
    // if neither east nor self are set, set F to 0: east NOR self
    // if east or self but not both are set, set F to xVel: east XOR self
    // if both are set, set F to the equation: east AND self.
    // special case: if rowNum is 0, follow the path for setting F to xVel.
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x;
    int colNum = blockIdx.y * blockDim.y + threadIdx.y;

    if (rowNum > iMax) return; // Bounds checking
    if (colNum > jMax) return;

    int selfBit = (*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & SELF) >> SELFSHIFT;
    int eastBit = (*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & EAST) >> EASTSHIFT;
}


__global__ void ComputeG(PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> G, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timeStep, REAL delY, REAL yForce, REAL* gamma, REAL reynoldsNum) {
    // Branchless plan:
    // if neither north nor self are set, set G to 0: north NOR self
    // if north or self but not both are set, set G to yVel: north XOR self
    // if both are set, set G to the equation: north AND self.
    // special case: if colNum is 0, follow the path for setting G to yVel.
}

cudaError_t ComputeFG(cudaStream_t* streams, dim3 threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> F, PointerWithPitch<REAL> G, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY, REAL xForce, REAL yForce, REAL* gamma, REAL reynoldsNum) {
    dim3 numBlocks((int)ceilf((float)(iMax + 1) / threadsPerBlock.x), (int)ceilf((float)(jMax + 1) / threadsPerBlock.y));
    ComputeF<<<numBlocks, threadsPerBlock, 0, streams[0]>>>(hVel, F, flags, iMax, jMax, timestep, delX, xForce, gamma, reynoldsNum); // Launch the kernels in separate streams, to be concurrently executed if the GPU is able to.
    ComputeG<<<numBlocks, threadsPerBlock, 0, streams[1]>>>(vVel, G, flags, iMax, jMax, timestep, delY, yForce, gamma, reynoldsNum);
    return cudaDeviceSynchronize();
}

__global__ void ComputeRHS(PointerWithPitch<REAL> F, PointerWithPitch<REAL> G, PointerWithPitch<REAL> RHS, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY) {
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int colNum = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (rowNum > iMax) return; // Bounds checking
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
        * ((*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & EAST) >> EASTSHIFT) // Equal to 0 if the cell has an obstacle cell next to it
        * *F_PITCHACCESS(F.ptr, F.pitch, rowNum, colNum) - (*timestep / delX) * (*F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum + 1, colNum) - *F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, colNum));
}

__global__ void ComputeVVel(PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> G, PointerWithPitch<REAL> pressure, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delY)
{
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int colNum = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (rowNum > iMax) return; // Bounds checking
    if (colNum > jMax) return;

    *F_PITCHACCESS(vVel.ptr, vVel.pitch, rowNum, colNum) =
        ((*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & SELF) >> SELFSHIFT) // Equal to 0 if the cell is not a fluid cell
        * ((*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & NORTH) >> NORTHSHIFT) // Equal to 0 if the cell has an obstacle cell next to it
        * *F_PITCHACCESS(G.ptr, G.pitch, rowNum, colNum) - (*timestep / delY) * (*F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, colNum + 1) - *F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, colNum));
}

cudaError_t ComputeVelocities(cudaStream_t* streams, dim3 threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> F, PointerWithPitch<REAL> G, PointerWithPitch<REAL> pressure, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY)
{
    dim3 numBlocks((int)ceilf((float)iMax / threadsPerBlock.x), (int)ceilf((float)jMax / threadsPerBlock.y));
    ComputeHVel<<<numBlocks, threadsPerBlock, 0, streams[0]>>>(hVel, F, pressure, flags, iMax, jMax, timestep, delX); // Launch the kernels in separate streams, to be concurrently executed if the GPU is able to.
    ComputeVVel<<<numBlocks, threadsPerBlock, 0, streams[1]>>>(vVel, G, pressure, flags, iMax, jMax, timestep, delY);
    return cudaDeviceSynchronize();
}

__global__ void ComputeStream(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> streamFunction, int iMax, int jMax, REAL delY)
{
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowNum > iMax) return; // Bounds checking

    *F_PITCHACCESS(streamFunction.ptr, streamFunction.pitch, rowNum, 0) = 0; // Stream function boundary condition
    for (int colNum = 1; colNum <= jMax; colNum++) {
        *F_PITCHACCESS(streamFunction.ptr, streamFunction.pitch, rowNum, colNum) = *F_PITCHACCESS(streamFunction.ptr, streamFunction.pitch, rowNum, colNum - 1) + *F_PITCHACCESS(hVel.ptr, hVel.pitch, rowNum, colNum) * delY;
    }
}
