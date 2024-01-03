#include "Computation.cuh"

constexpr BYTE SELF = 0b00010000;
constexpr BYTE NORTH = 0b00001000;
constexpr BYTE EAST = 0b00000100;
constexpr BYTE SOUTH = 0b00000010;
constexpr BYTE WEST = 0b00000001;

__global__ void ComputeRHS(PointerWithPitch<REAL> F, PointerWithPitch<REAL> G, PointerWithPitch<REAL> RHS, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY) {
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int colNum = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (rowNum > iMax) return; // Bounds checking
    if (colNum > jMax) return;
    
    *F_PITCHACCESS(RHS.ptr, RHS.pitch, rowNum, colNum) = (*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & SELF) * (1 / *timestep) * (((*F_PITCHACCESS(F.ptr, F.pitch, rowNum, colNum) - *F_PITCHACCESS(F.ptr, F.pitch, rowNum - 1, colNum)) / delX) + ((*F_PITCHACCESS(G.ptr, G.pitch, rowNum, colNum) - *F_PITCHACCESS(G.ptr, G.pitch, rowNum, colNum - 1)) / delY));
}