#include "Boundary.cuh"
#include "math.h"
#include <vector>

__global__ void SetFlags(PointerWithPitch<bool> obstacles, PointerWithPitch<BYTE> flags, int iMax, int jMax) {
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int colNum = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (rowNum > iMax) return;
    if (colNum > jMax) return;

    F_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) = ((BYTE)B_PITCHACCESS(obstacles.ptr, obstacles.pitch, rowNum, colNum) << 4) + ((BYTE)B_PITCHACCESS(obstacles.ptr, obstacles.pitch, rowNum, colNum + 1) << 3) + ((BYTE)B_PITCHACCESS(obstacles.ptr, obstacles.pitch, rowNum + 1, colNum) << 2) + ((BYTE)B_PITCHACCESS(obstacles.ptr, obstacles.pitch, rowNum, colNum - 1) << 1) + (BYTE)B_PITCHACCESS(obstacles.ptr, obstacles.pitch, rowNum - 1, colNum); //5 bits in the format: self, north, east, south, west.
}

__global__ void TopBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int jMax)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;

    F_PITCHACCESS(hVel.ptr, hVel.pitch, index, jMax + 1) = F_PITCHACCESS(hVel.ptr, hVel.pitch, index, jMax); // Copy hVel from the cell below
    F_PITCHACCESS(vVel.ptr, vVel.pitch, index, jMax) = 0; // Set vVel along the top to 0
}

__global__ void BottomBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;

    F_PITCHACCESS(hVel.ptr, hVel.pitch, index, 0) = F_PITCHACCESS(hVel.ptr, hVel.pitch, index, 1); // Copy hVel from the cell above
    F_PITCHACCESS(vVel.ptr, vVel.pitch, index, 0) = 0; // Set vVel along the bottom to 0
}

__global__ void LeftBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, REAL inflowVelocity)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;

    F_PITCHACCESS(hVel.ptr, hVel.pitch, 0, index) = inflowVelocity; // Set hVel to inflow velocity on left boundary
    F_PITCHACCESS(vVel.ptr, vVel.pitch, 0, index) = 0; // Set vVel to 0
}

__global__ void RightBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int iMax)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;

    F_PITCHACCESS(hVel.ptr, hVel.pitch, iMax, index) = F_PITCHACCESS(hVel.ptr, hVel.pitch, iMax - 1, index); // Copy the velocity values from the previous cell (mass flows out at the boundary)
    F_PITCHACCESS(vVel.ptr, vVel.pitch, iMax + 1, index) = F_PITCHACCESS(vVel.ptr, vVel.pitch, iMax, index);
}

__global__ void ObstacleBoundary(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<BYTE> flags, uint2* coordinates, int coordinatesLength, REAL chi) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > coordinatesLength) return;
    
    uint2 coordinate = coordinates[index];

    BYTE flag = B_PITCHACCESS(flags.ptr, flags.pitch, coordinate.x, coordinate.y);
    int northBit = (flag & NORTH) >> NORTHSHIFT;
    int eastBit  = (flag & EAST)  >> EASTSHIFT;
    int southBit = (flag & SOUTH) >> SOUTHSHIFT;
    int westBit  = (flag & WEST) >> WESTSHIFT;

    REAL velocityModifier = 2 * chi - 1; // This converts chi from chi in [0,1] to in [-1,1]

    F_PITCHACCESS(hVel.ptr, hVel.pitch, coordinate.x, coordinate.y) = (1 - eastBit) // If the cell is an eastern boudary, hVel is 0
        * (northBit * velocityModifier * F_PITCHACCESS(hVel.ptr, hVel.pitch, coordinate.x, coordinate.y + 1) // For northern boundaries, use the horizontal velocity above...
            + southBit * velocityModifier * F_PITCHACCESS(hVel.ptr, hVel.pitch, coordinate.x, coordinate.y - 1)); // ...and for southern boundaries, use the horizontal velocity below.

    F_PITCHACCESS(vVel.ptr, vVel.pitch, coordinate.x, coordinate.y) = (1 - northBit) // If the cell is a northern boundary, vVel is 0
        * (eastBit * velocityModifier * F_PITCHACCESS(vVel.ptr, vVel.pitch, coordinate.x + 1, coordinate.y) // For eastern boundaries, use the vertical velocity to the right...
            + westBit * velocityModifier * F_PITCHACCESS(vVel.ptr, vVel.pitch, coordinate.x - 1, coordinate.y)); // ...and for western boundaries, use the vertical velocity to the left.

    // The following lines are unavoidable branches.
    if (southBit != 0) { // If south bit is set,...
        F_PITCHACCESS(vVel.ptr, vVel.pitch, coordinate.x, coordinate.y - 1) = 0; // ...then set the velocity coming into the boundary to 0.
    }

    if (westBit != 0) { // If west bit is set,...
        F_PITCHACCESS(hVel.ptr, hVel.pitch, coordinate.x - 1, coordinate.y) = 0; // ...then set the velocity coming into the boundary to 0.
    }
}

cudaError_t SetBoundaryConditions(cudaStream_t* streams, int threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<BYTE> flags, uint2* coordinates, int coordinatesLength, int iMax, int jMax, REAL inflowVelocity, REAL chi) {
    int numBlocksTopBottom = INT_DIVIDE_ROUND_UP(iMax, threadsPerBlock);
    int numBlocksLeftRight = INT_DIVIDE_ROUND_UP(jMax, threadsPerBlock);
    int numBlocksObstacle = INT_DIVIDE_ROUND_UP(coordinatesLength, threadsPerBlock);
    
    TopBoundary KERNEL_ARGS4(numBlocksTopBottom, threadsPerBlock, 0, streams[0]) (hVel, vVel, jMax);
    BottomBoundary KERNEL_ARGS4(numBlocksTopBottom, threadsPerBlock, 0, streams[1]) (hVel, vVel);
    LeftBoundary KERNEL_ARGS4(numBlocksLeftRight, threadsPerBlock, 0, streams[2]) (hVel, vVel, inflowVelocity);
    RightBoundary KERNEL_ARGS4(numBlocksLeftRight, threadsPerBlock, 0, streams[3]) (hVel, vVel, iMax);

    cudaStreamSynchronize(streams[0]);
    ObstacleBoundary KERNEL_ARGS4(numBlocksObstacle, threadsPerBlock, 0, streams[0]) (hVel, vVel, flags, coordinates, coordinatesLength, chi);

    return cudaDeviceSynchronize();
}

// Counts number of fluid cells in the region [1,iMax]x[1,jMax]
int CountFluidCells(BYTE** flags, int iMax, int jMax) {
    int count = 0;
    for (int i = 0; i <= iMax; i++) {
        for (int j = 0; j <= jMax; j++) {
            count += flags[i][j] >> 4; // This will include only the "self" bit, which is one for fluid cells and 0 for boundary and obstacle cells.
        }
    }
    return count;
}

void FindBoundaryCells(BYTE** flags, uint2*& coordinates, int& coordinatesLength, int iMax, int jMax) {
    std::vector<uint2> coordinatesVec;
    for (int i = 1; i <= iMax; i++) {
        for (int j = 1; j <= jMax; j++) {
            if (flags[i][j] >= 0b00000001 && flags[i][j] <= 0b00001111) { // This defines boundary cells - all cells without the self bit set except when no bits are set. This could probably be optimised.
                uint2 coordinate = uint2();
                coordinate.x = i;
                coordinate.y = j;
                coordinatesVec.push_back(coordinate);
            }
        }
    }
    coordinates = new uint2[coordinatesVec.size()]; // Allocate mem for array into already defined pointer
    std::copy(coordinatesVec.begin(), coordinatesVec.end(), coordinates); // Copy the elements from the vector to the array
    coordinatesLength = (int)coordinatesVec.size();
}