#include "Computation.cuh"
#include "DiscreteDerivatives.cuh"
#include <cmath>

#define INT_DIVIDE_ROUND_UP(numerator, denominator) (((numerator) + (denominator) - 1) / (denominator))

constexpr BYTE SELF  = 0b00010000;
constexpr BYTE NORTH = 0b00001000;
constexpr BYTE EAST  = 0b00000100;
constexpr BYTE SELFSHIFT  = 4;
constexpr BYTE NORTHSHIFT = 3;
constexpr BYTE EASTSHIFT  = 2;


__device__ void GroupMax(cg::thread_group group, volatile float* sharedArray, int arrayLength) {
    // Each thread starts by copying their value into a shared memory array for speed, then comparing their value to the one at index + indexThreshold and storing the max at index
    // Then the number of threads are halved, and this is repeated
    // The final thread returns a value

    int index = group.thread_rank();
    float val = 0;
    for (int indexThreshold = arrayLength / 2; indexThreshold > 1; indexThreshold = INT_DIVIDE_ROUND_UP(indexThreshold, 2)) {
        if (index < indexThreshold) { // Halve the number of threads each iteration
            val = fmaxf(val, sharedArray[index + indexThreshold]); // Get the max of the thread's own value and the one at index + indexThreshold
            sharedArray[index] = val; // Store the max into the shared array at the current index
            //if (blockIdx.x == 3) printf("Thread %i: value %f\n", index, val);
        }
        group.sync();
    }
    if (index == 0) { // Final iteration for the last thread (indices 0 and 1 to compare)
        sharedArray[index] = fmaxf(sharedArray[0], sharedArray[1]);
        //if (blockIdx.x == 3) printf("Thread %i: value %f\n", 0, sharedArray[0]);
    }
}

__global__ void ComputeMaxesSingleBlock(float* max, float* globalArray, int arrayLength) {

    extern __shared__ float sharedArray[];
    cg::thread_block threadBlock = cg::this_thread_block();
    int index = threadBlock.thread_rank();

    if (index < (arrayLength / 2)) { // Bounds checking
        *(float2*)(sharedArray + index * 2) = *(float2*)(globalArray + index * 2); // Copy 2 floats at once (vectorised access) because threadsPerBlock * 2 floats need to be copied.
    }

    threadBlock.sync(); // Sync all threads, including inactive ones.

    GroupMax(threadBlock, sharedArray, arrayLength); // Run the actual function. Inactive threads that run the function will be handled inside it.

    if (index == 0) { // Thread 0 sets the return value
        *max = sharedArray[0];
    }

}

__global__ void ComputeMaxesMultiBlock(float* maxesArray, float* globalArray, int arrayLength) {
    // Each thread performs GroupMax with this_thread_block() as parameter
    // Note: keep in mind that the below code is run by INDIVIDUAL THREADS IN DIFFERENT BLOCKS.
    // Also, all of the thread blocks here are independant

    int gridIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int currentBlock = blockIdx.x;

    extern __shared__ float sharedArray[];
    cg::thread_block threadBlock = cg::this_thread_block();

    if (gridIndex < arrayLength / 2) { // Bounds checking
        *(float2*)(sharedArray + threadIdx.x * 2) = *(float2*)(globalArray + gridIndex * 2); // Copy 2 floats at once (vectorised access) because (gridDim.x * 2) floats need to be copied.
        // The above line may not work correctly - each thread copies a section of global memory into a section of its block's shared memory (ideally)
    }

    threadBlock.sync(); // Wait for all threads to do their copy

    int subArrayLength = threadBlock.size() * 2;
    if (currentBlock == gridDim.x - 1) { // Last block may not have all threads active
        subArrayLength = arrayLength - blockDim.x * blockIdx.x * 2; // Subtract the size of all the previous blocks, multiply by 2.
    }

    GroupMax(threadBlock, sharedArray, subArrayLength); // Results are stored at each block's sharedArray[0] for i between 0 and number of blocks

    if (threadBlock.thread_rank() == 0) { // Thread 0 for each block sets the value in the max array in global mem
        maxesArray[currentBlock] = sharedArray[0];
    }
}

cudaError_t ArrayMax(cudaStream_t stream, float* max, int threadsPerBlock, float* values, int arrayLength) {
    cudaError_t status;
    int numElementsPerBlock = threadsPerBlock * 2;
    int numBlocks = INT_DIVIDE_ROUND_UP(arrayLength, numElementsPerBlock); // Each block processes (threadsPerBlock * 2) values
    float* partialSums;
    status = cudaMalloc(&partialSums, numBlocks * sizeof(float)); // Allocate an array for the blocks to store their return values.
    if (status != cudaSuccess) goto free;

    ComputeMaxesMultiBlock<<<numBlocks, threadsPerBlock, threadsPerBlock * 2 * sizeof(float), stream>>>(partialSums, values, arrayLength); // Gets the maxes and stores it in the secondary results array
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) goto free;

    ComputeMaxesSingleBlock<<<1, threadsPerBlock, numBlocks * sizeof(float), stream>>>(max, partialSums, numBlocks); // Use 1 thread block to compute
    status = cudaDeviceSynchronize();

    free:
    cudaFree(partialSums);

    return status;
}

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

    ArrayMax(streams[0], hVelMax, threadsPerBlock, hVel, (iMax + 2) * (jMax + 2));
    ArrayMax(streams[1], vVelMax, threadsPerBlock, vVel, (iMax + 2) * (jMax + 2));

    retVal = cudaStreamSynchronize(streams[0]);
    if (retVal != cudaSuccess) goto free;

    retVal = cudaStreamSynchronize(streams[1]);
    if (retVal != cudaSuccess) goto free;

    FinishComputeGamma<<<1, 1, 0, streams[0]>>>(gamma, hVelMax, vVelMax, timestep, delX, delY);

    free:
    cudaFree(hVelMax);
    cudaFree(vVelMax);
    return retVal;
}

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
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int colNum = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (rowNum >= iMax) return;
    if (colNum > jMax) return;

    int selfBit = (*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & SELF) >> SELFSHIFT; // SELF bit of the cell's flag
    int eastBit = (*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & EAST) >> EASTSHIFT; // EAST bit of the cell's flag
    
    *F_PITCHACCESS(F.ptr, F.pitch, rowNum, colNum) =
        *F_PITCHACCESS(hVel.ptr, hVel.pitch, rowNum, colNum) * (selfBit | eastBit) // For boundary cells or fluid cells, add hVel
        + *timestep * (1 / reynoldsNum * (SecondPuPx(hVel, rowNum, colNum, delX) + SecondPuPy(hVel, rowNum, colNum, delY)) - PuSquaredPx(hVel, rowNum, colNum, delX, *gamma) - PuvPy(hVel, vVel, rowNum, colNum, delX, delY, *gamma) + xForce) * (selfBit & eastBit); // For fluid cells only, perform the computation. Obstacle cells without an eastern boundary are set to 0.
}


__global__ void ComputeG(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> G, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY, REAL yForce, REAL* gamma, REAL reynoldsNum) {
    int rowNum = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int colNum = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (rowNum > iMax) return;
    if (colNum >= jMax) return;

    int selfBit = (*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & SELF) >> SELFSHIFT;    // SELF bit of the cell's flag
    int northBit = (*B_PITCHACCESS(flags.ptr, flags.pitch, rowNum, colNum) & NORTH) >> NORTHSHIFT; // NORTH bit of the cell's flag

    *F_PITCHACCESS(G.ptr, G.pitch, rowNum, colNum) =
        *F_PITCHACCESS(vVel.ptr, vVel.pitch, rowNum, colNum) * (selfBit | northBit) // For boundary cells or fluid cells, add vVel
        + *timestep * (1 / reynoldsNum * (SecondPvPx(vVel, rowNum, colNum, delX) + SecondPvPy(vVel, rowNum, colNum, delY)) - PuvPx(hVel, vVel, rowNum, colNum, delX, delY, *gamma) - PvSquaredPy(vVel, rowNum, colNum, delY, *gamma) + yForce) * (selfBit & northBit); // For fluid cells only, perform the computation. Obstacle cells without a northern boundary are set to 0.
}

cudaError_t ComputeFG(cudaStream_t* streams, dim3 threadsPerBlock, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> F, PointerWithPitch<REAL> G, PointerWithPitch<BYTE> flags, int iMax, int jMax, REAL* timestep, REAL delX, REAL delY, REAL xForce, REAL yForce, REAL* gamma, REAL reynoldsNum) {
    dim3 numBlocksF(INT_DIVIDE_ROUND_UP(iMax - 1, threadsPerBlock.x), INT_DIVIDE_ROUND_UP(jMax, threadsPerBlock.y));
    dim3 numBlocksG(INT_DIVIDE_ROUND_UP(iMax, threadsPerBlock.x), INT_DIVIDE_ROUND_UP(jMax - 1, threadsPerBlock.y));

    int threadsPerBlockFlat = threadsPerBlock.x * threadsPerBlock.y;
    int numBlocksIMax = INT_DIVIDE_ROUND_UP(iMax, threadsPerBlockFlat);
    int numBlocksJMax = INT_DIVIDE_ROUND_UP(jMax, threadsPerBlockFlat);

    ComputeF<<<numBlocksF, threadsPerBlock, 0, streams[0]>>>(hVel, vVel, F, flags, iMax, jMax, timestep, delX, delY, xForce, gamma, reynoldsNum); // Launch the kernels in separate streams, to be concurrently executed if the GPU is able to.
    ComputeG<<<numBlocksG, threadsPerBlock, 0, streams[1]>>>(hVel, vVel, G, flags, iMax, jMax, timestep, delX, delY, yForce, gamma, reynoldsNum);

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
    dim3 numBlocks(INT_DIVIDE_ROUND_UP(iMax, threadsPerBlock.x), INT_DIVIDE_ROUND_UP(jMax, threadsPerBlock.y));
    ComputeHVel<<<numBlocks, threadsPerBlock, 0, streams[0]>>>(hVel, F, pressure, flags, iMax, jMax, timestep, delX); // Launch the kernels in separate streams, to be concurrently executed if the GPU is able to.
    ComputeVVel<<<numBlocks, threadsPerBlock, 0, streams[1]>>>(vVel, G, pressure, flags, iMax, jMax, timestep, delY);
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
