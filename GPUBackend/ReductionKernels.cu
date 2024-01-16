#include "ReductionKernels.cuh"
#include <cmath>

#ifdef __INTELLISENSE__ // Allow intellisense to recognise cooperative groups
#define __CUDACC__
#endif // __INTELLISENSE__
#include <cooperative_groups.h>
#ifdef __INTELLISENSE__
#undef __CUDACC__
#endif // __INTELLISENSE__

namespace cg = cooperative_groups;

/// <summary>
/// Computes the max of the elements in <paramref name="sharedArray" />. Processes the number of elements equal to <paramref name="group" />'s size.
/// </summary>
/// <param name="group">The thread group of which the calling thread is a member.</param>
/// <param name="sharedArray">The array, in shared memory, to find the maximum of.</param>
__device__ void GroupMax(cg::thread_group group, volatile REAL* sharedArray) {
    int index = group.thread_rank();
    REAL val = sharedArray[index];
    for (int indexThreshold = group.size() / 2; indexThreshold > 0; indexThreshold /= 2) {
        if (index < indexThreshold) { // Halve the number of threads each iteration
            val = fmax(val, sharedArray[index + indexThreshold]); // Get the max of the thread's own value and the one at index + indexThreshold
            sharedArray[index] = val; // Store the max into the shared array at the current index
        }
        group.sync();
    }
}
/// <summary>
/// Computes the maximum of each column of a field. Requires xLength blocks, each of <c>field.pitch / sizeof(REAL)</c> threads, and 1 REAL's worth of shared memory per thread.
/// </summary>
/// <param name="partialMaxes">An array of length equal to the number of rows, for outputting the maxes of each column.</param>
/// <param name="field">The input field.</param>
/// <param name="yLength">The length of a column.</param>
__global__ void ComputePartialMaxes(REAL* partialMaxes, PointerWithPitch<REAL> field, int yLength) {
    cg::thread_block threadBlock = cg::this_thread_block();
    REAL* colBase = (REAL*)((char*)field.ptr + blockIdx.x * field.pitch);

    // Perform copy to shared memory.
    // Put a 0 in shared if current index is greater than yLength (this catches index in pitch padding, or index > size of a row)
    extern __shared__ REAL sharedArray[];

    if (threadIdx.x < yLength) { // the index of the thread is greater than the length of a column.
        sharedArray[threadIdx.x] = *(colBase + threadIdx.x);
    }
    else {
        sharedArray[threadIdx.x] = (REAL)0;
    }
    threadBlock.sync();

    GroupMax(threadBlock, sharedArray);

    if (threadIdx.x == 0) { // If the thread is the 0th in the block, store its result to global memory.
        partialMaxes[blockIdx.x] = sharedArray[0];
    }
}

/// <summary>
/// Computes the final max from a given array of partial maxes. Requires 1 block of <paramref name="xLength" /> threads, and 1 REAL's worth of shared memory per thread.
/// </summary>
/// <param name="max">The location to place the output.</param>
/// <param name="partialMaxes">An array of partial maxes, of size <paramref name="xLength" />.</param>
__global__ void ComputeFinalMax(REAL* max, REAL* partialMaxes, int xLength)
{
    cg::thread_block threadBlock = cg::this_thread_block();

    extern __shared__ REAL sharedMem[];

    // Copy to shared memory again
    if (threadIdx.x < xLength) {
        sharedMem[threadIdx.x] = partialMaxes[threadIdx.x];
    }
    else {
        sharedMem[threadIdx.x] = (REAL)0;
    }
    threadBlock.sync();

    GroupMax(threadBlock, sharedMem);
    if (threadIdx.x == 0) { // Thread 0 stores the final element.
        *max = sharedMem[0];
    }
}

/// <summary>
/// Computes the sum of the elements in <paramref name="sharedArray" />. Processes the number of elements equal to <paramref name="group" />'s size.
/// </summary>
/// <param name="group">The thread group of which the calling thread is a member.</param>
/// <param name="sharedArray">The array, in shared memory, to find the sum of.</param>
__device__ void GroupSum(cg::thread_group group, volatile REAL* sharedArray) {
    int index = group.thread_rank();
    for (int indexThreshold = group.size() / 2; indexThreshold > 0; indexThreshold /= 2) {
        if (index < indexThreshold) { // Halve the number of threads each iteration
            sharedArray[index] += sharedArray[index + indexThreshold]; // Add the value at index + indexThreshold to the value at the current index.
        }
        group.sync();
    }
}

/// <summary>
/// Computes the sum of each column of a field. Requires xLength blocks, each of <c>field.pitch / sizeof(REAL)</c> threads, and 1 REAL's worth of shared memory per thread.
/// </summary>
/// <param name="partialSums">An array of length equal to the number of rows, for outputting the sums of each column.</param>
/// <param name="field">The input field.</param>
/// <param name="yLength">The length of a column.</param>
__global__ void ComputePartialSums(REAL* partialSums, PointerWithPitch<REAL> field, int yLength) {
    cg::thread_block threadBlock = cg::this_thread_block();
    REAL* colBase = (REAL*)((char*)field.ptr + blockIdx.x * field.pitch);

    // Perform copy to shared memory.
    // Put a 0 in shared if current index is greater than yLength (this catches index in pitch padding, or index > size of a row)
    extern __shared__ REAL sharedArray[];

    if (threadIdx.x < yLength) { // the index of the thread is greater than the length of a column.
        sharedArray[threadIdx.x] = *(colBase + threadIdx.x);
    }
    else {
        sharedArray[threadIdx.x] = (REAL)0;
    }
    threadBlock.sync();

    GroupSum(threadBlock, sharedArray);

    if (threadIdx.x == 0) { // If the thread is the 0th in the block, store its result to global memory.
        partialSums[blockIdx.x] = sharedArray[0];
    }
}

/// <summary>
/// Computes the final sum from a given array of partial sums. Requires 1 block of <paramref name="xLength" /> threads, and 1 REAL's worth of shared memory per thread.
/// </summary>
/// <param name="sum">The location to place the output.</param>
/// <param name="partialSums">An array of partial sums, of size <paramref name="xLength" />.</param>
__global__ void ComputeFinalSum(REAL* sum, REAL* partialSums, int xLength)
{
    cg::thread_block threadBlock = cg::this_thread_block();

    extern __shared__ REAL sharedMem[];

    // Copy to shared memory again
    if (threadIdx.x < xLength) {
        sharedMem[threadIdx.x] = partialSums[threadIdx.x];
    }
    else {
        sharedMem[threadIdx.x] = (REAL)0;
    }
    threadBlock.sync();

    GroupSum(threadBlock, sharedMem);
    if (threadIdx.x == 0) { // Thread 0 stores the final element.
        *sum = sharedMem[0];
    }
}

cudaError_t FieldMax(REAL* max, cudaStream_t streamToUse, PointerWithPitch<REAL> field, int xLength, int yLength) {
    cudaError_t retVal;

    REAL* partialMaxes;
    retVal = cudaMalloc(&partialMaxes, xLength * sizeof(REAL));
    if (retVal != cudaSuccess) { // Return if there was an error in allocation
        return retVal;
    }

    // Run the GPU kernel:
    ComputePartialMaxes KERNEL_ARGS(xLength, (unsigned int)field.pitch / sizeof(REAL), field.pitch, streamToUse) (partialMaxes, field, yLength); // 1 block per row. Number of threads is equal to column pitch, and each thread has 1 REAL worth of shared memory.
    retVal = cudaStreamSynchronize(streamToUse);
    if (retVal != cudaSuccess) { // Skip the rest of the computation if there was an error
        goto free;
    }

    ComputeFinalMax KERNEL_ARGS(1, xLength, xLength * sizeof(REAL), streamToUse) (max, partialMaxes, xLength); // 1 block to process all of the partial maxes, number of threads equal to number of partial maxes (xLength is also this)
    retVal = cudaStreamSynchronize(streamToUse);


free:
    cudaFree(partialMaxes);
    return retVal;
}

cudaError_t FieldSum(REAL* sum, cudaStream_t streamToUse, PointerWithPitch<REAL> field, int xLength, int yLength) {
    cudaError_t retVal;

    REAL* partialSums;
    retVal = cudaMalloc(&partialSums, xLength * sizeof(REAL));
    if (retVal != cudaSuccess) { // Return if there was an error in allocation
        return retVal;
    }

    // Run the GPU kernel:
    ComputePartialSums KERNEL_ARGS(xLength, (unsigned int)field.pitch / sizeof(REAL), field.pitch, streamToUse) (partialSums, field, yLength); // 1 block per row. Number of threads is equal to column pitch, and each thread has 1 REAL worth of shared memory.
    retVal = cudaStreamSynchronize(streamToUse);
    if (retVal != cudaSuccess) { // Skip the rest of the computation if there was an error
        goto free;
    }

    ComputeFinalSum KERNEL_ARGS(1, xLength, xLength * sizeof(REAL), streamToUse) (sum, partialSums, xLength); // 1 block to process all of the partial sums, number of threads equal to number of partial sums (xLength is also this)
    retVal = cudaStreamSynchronize(streamToUse);

free:
    cudaFree(partialSums);
    return retVal;
}