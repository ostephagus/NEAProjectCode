#include "PressureComputation.cuh"
#include "DiscreteDerivatives.cuh"
#include "ReductionKernels.cuh"
#include <cmath>


/// <summary>
/// Calculates and validates the coordinates of a thread based on the coloured cell system.
/// </summary>
/// <param name="rowNum">The output row number (x coordinate).</param>
/// <param name="colNum">The output column number (y coordinate).</param>
/// <param name="threadPosX">X position of the thread in the grid.</param>
/// <param name="threadPosY">Y position of the thread in the grid.</param>
/// <param name="colourNum">The desired colour of the returned coordinates</param>
/// <param name="numberOfColours">The number of different colours assigned to cells.</param>
/// <returns>Whether the thread coordinates map to a valid cell.</returns>
__device__ bool CalculateColouredCoordinates(int* rowNum, int* colNum, int threadPosX, int threadPosY, int colourNum, int numberOfColours, int iMax, int jMax) {
    // Require (rowNum + colNum) % numberOfColours = colourNum
    *rowNum = threadPosX + 1; // Normal rowNum - x position of thread in grid.
    int colOffset = colourNum - *rowNum % numberOfColours - 1; // The number to add to the column number (the required result of colNum % numberOfColours, subtracted 1 to avoid colNum being 0).
    if (colOffset < 0) {
        colOffset += numberOfColours;
    }

    *colNum = numberOfColours * threadPosY + colOffset + 1; // Generate colNum such that colNum % numberOfColours = colOffset.

    return *rowNum <= iMax && *colNum <= jMax;
}

/// <summary>
/// Gets the parity (number of bits that are set mod 2) of a byte.
/// </summary>
/// <param name="input">The input byte</param>
/// <returns>The parity of the byte, 1 or 0.</returns>
__host__ __device__ BYTE GetParity(BYTE input) {
    input ^= input >> 4; // Repeatedly shift-XOR to end up with the XOR of all of the bits in the LSB.
    input ^= input >> 2;
    input ^= input >> 1;
    return input & 1; // The parity is stored in the last bit, so XOR the result with 1 and return.
}


/// <summary>
/// Calculates pressures of one colour of the grid. Requires iMax x (jMax / numberOfColours) threads.
/// </summary>
__global__ void SingleColourSOR(int numberOfColours, int colourNum, PointerWithPitch<REAL> pressure, PointerWithPitch<REAL> RHS, PointerWithPitch<BYTE> flags, PointerWithPitch<REAL> residualArray, int iMax, int jMax, REAL delX, REAL delY, REAL omega, REAL boundaryFraction)
{
    int rowNum, colNum;
    bool validCell = CalculateColouredCoordinates(&rowNum, &colNum, blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, colourNum, numberOfColours, iMax, jMax);

    if (!validCell) return; // If the cell is not valid, do not perform the computation

    if ((B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & SELF) == 0) return; // If the cell is not a fluid cell, also do not perform the computation.

    REAL relaxedPressure = (1 - omega) * F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, colNum);
    REAL pressureAverages = ((F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum + 1, colNum) + F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum - 1, colNum)) / square(delX)) + ((F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, colNum + 1) + F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, colNum - 1)) / square(delY)) - F_PITCHACCESS(RHS.ptr, RHS.pitch, rowNum, colNum);
    F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, colNum) = relaxedPressure + boundaryFraction * pressureAverages;

    REAL currentResidual = pressureAverages - (2 * F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, colNum)) / square(delX) - (2 * F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, colNum)) / square(delY);

    F_PITCHACCESS(residualArray.ptr, residualArray.pitch, rowNum - 1, colNum - 1) = square(currentResidual); // Residual array is shifted down and left 1 so less memory is needed.
}

/// <summary>
/// Copies pressure values at the top and bottom of the simulation domain. Requires iMax threads.
/// </summary>
__global__ void CopyHorizontalPressures(PointerWithPitch<REAL> pressure, int iMax, int jMax) {
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (rowNum > iMax) return;
    
    F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, 0) = F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, 1);
    F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, jMax + 1) = F_PITCHACCESS(pressure.ptr, pressure.pitch, rowNum, jMax);
}

/// <summary>
/// Copies pressure values at the top and bottom of the simulation domain. Requires jMax threads.
/// </summary>
__global__ void CopyVerticalPressures(PointerWithPitch<REAL> pressure, int iMax, int jMax) {
    int colNum = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (colNum > jMax) return;

    F_PITCHACCESS(pressure.ptr, pressure.pitch, 0, colNum) = F_PITCHACCESS(pressure.ptr, pressure.pitch, 1, colNum);
    F_PITCHACCESS(pressure.ptr, pressure.pitch, iMax + 1, colNum) = F_PITCHACCESS(pressure.ptr, pressure.pitch, iMax, colNum);
}

/// <summary>
/// Copies the pressures for the boundary cells given in <paramref name="coordinates" />. Requires <paramref name="coordinatesLength" /> threads.
/// </summary>
__global__ void CopyBoundaryPressures(PointerWithPitch<REAL> pressure, uint2* coordinates, int coordinatesLength, PointerWithPitch<BYTE> flags, int iMax, int jMax) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= coordinatesLength) return;

    uint2 coordinate = coordinates[index]; // Get coordinate from global memory...
    BYTE relevantFlag = B_PITCHACCESS(flags.ptr, flags.pitch, coordinate.x, coordinate.y); // ...and the flag for that coordinate.

    int xShift = ((relevantFlag & EAST) >> EASTSHIFT) - ((relevantFlag & WEST) >> WESTSHIFT); // Relative position of cell to copy in x direction. -1, 0 or 1.
    int yShift = ((relevantFlag & NORTH) >> NORTHSHIFT) - ((relevantFlag & SOUTH) >> SOUTHSHIFT); // Relative position of cell to copy in y direction. -1, 0 or 1.

    if (GetParity(relevantFlag) == 1) { // Only boundary cells with one edge - copy from that fluid cell
        F_PITCHACCESS(pressure.ptr, pressure.pitch, coordinate.x, coordinate.y) = F_PITCHACCESS(pressure.ptr, pressure.pitch, coordinate.x + xShift, coordinate.y + yShift); // Copy from the cell determined by the shifts.
    }
    else { // These are boundary cells with 2 edges - take the average of the 2 cells with the boundary.
        F_PITCHACCESS(pressure.ptr, pressure.pitch, coordinate.x, coordinate.y) = (F_PITCHACCESS(pressure.ptr, pressure.pitch, coordinate.x + xShift, coordinate.y) + F_PITCHACCESS(pressure.ptr, pressure.pitch, coordinate.x, coordinate.y + yShift)) / (REAL)2; // Take the average of the one above/below and the one left/right by only using one shift for each of the field accesses.
    }
}

// Could implement this using cuda graphs
int Poisson(cudaStream_t* streams, dim3 threadsPerBlock, PointerWithPitch<REAL> pressure, PointerWithPitch<REAL> RHS, PointerWithPitch<BYTE> flags, uint2* coordinates, int coordinatesLength, int numFluidCells, int iMax, int jMax, int numColours, REAL delX, REAL delY, REAL residualTolerance, int minIterations, int maxIterations, REAL omega, REAL* residualNorm) {
    cudaError_t retVal;
    int numIterations = 0;
    REAL boundaryFraction = omega / ((2 / square(delX)) + (2 / square(delY))); // Only executed once so easier to execute on CPU and transfer.

    dim3 numBlocks(INT_DIVIDE_ROUND_UP(iMax, threadsPerBlock.x), INT_DIVIDE_ROUND_UP(jMax / numColours, threadsPerBlock.y)); // Number of blocks for an iMax x jMax launch.

    int threadsPerBlockFlattened = threadsPerBlock.x * threadsPerBlock.y;
    int numBlocksIMax = INT_DIVIDE_ROUND_UP(iMax, threadsPerBlockFlattened);
    int numBlocksJMax = INT_DIVIDE_ROUND_UP(jMax, threadsPerBlockFlattened);

    PointerWithPitch<REAL> residualArray = PointerWithPitch<REAL>(); // Create pointers and set them to null
    REAL* d_residualNorm = nullptr; // Create a device version of residualNorm

    retVal = cudaMallocPitch(&residualArray.ptr, &residualArray.pitch, jMax * sizeof(REAL), iMax); // Create residualArray with size iMax * jMax
    if (retVal != cudaSuccess) goto free;

    retVal = cudaMalloc(&d_residualNorm, sizeof(REAL)); // Allocate one REAL's worth of memory
    if (retVal != cudaSuccess) goto free;


    do {
        *residualNorm = 0; // Set both host and device residual norms to 0.
        retVal = cudaMemset(d_residualNorm, 0, sizeof(REAL));
        if (retVal != cudaSuccess) goto free;

        for (int colourNum = 0; colourNum < numColours; colourNum++) { // Loop through however many colours and perform SOR.
            SingleColourSOR KERNEL_ARGS(numBlocks, threadsPerBlock, 0, streams[0]) (numColours, colourNum, pressure, RHS, flags, residualArray, iMax, jMax, delX, delY, omega, boundaryFraction);
        }

        retVal = cudaStreamSynchronize(streams[0]);
        if (retVal != cudaSuccess) goto free;

        retVal = FieldSum(d_residualNorm, streams[0], residualArray, iMax, jMax);
        if (retVal != cudaSuccess) goto free;

        // Copy the boundary cell pressures all in different streams
        CopyHorizontalPressures KERNEL_ARGS(numBlocksIMax, threadsPerBlockFlattened, 0, streams[0]) (pressure, iMax, jMax);
        CopyVerticalPressures KERNEL_ARGS(numBlocksJMax, threadsPerBlockFlattened, 0, streams[1]) (pressure, iMax, jMax);
        CopyBoundaryPressures KERNEL_ARGS(numBlocks, threadsPerBlock, 0, streams[2]) (pressure, coordinates, coordinatesLength, flags, iMax, jMax);

        retVal = cudaMemcpyAsync(residualNorm, d_residualNorm, sizeof(REAL), cudaMemcpyDeviceToHost, streams[3]); // Also copy residual norm to host for conditional processing
        if (retVal != cudaSuccess) goto free;

        retVal = cudaStreamSynchronize(streams[3]);
        if (retVal != cudaSuccess) goto free;

        *residualNorm = sqrt(*residualNorm / numFluidCells);
        numIterations++;
    } while ((numIterations < maxIterations && *residualNorm > residualTolerance) || numIterations < minIterations);

free:
    cudaFree(residualArray.ptr);
    cudaFree(d_residualNorm);
    return retVal == cudaSuccess ? numIterations : 0; // Return 0 if there was an error, otherwise the number of iterations.
}