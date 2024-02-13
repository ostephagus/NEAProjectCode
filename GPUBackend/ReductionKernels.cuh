#ifndef REDUCTION_KERNELS_CUH

#include "Definitions.cuh"

/// <summary>
/// Computes partial maxes of an array. Requires 1 REAL's worth of shared memory per thread.
/// </summary>
/// <param name="partialMaxes">The output array, length equal to the number of blocks spawned.</param>
/// <param name="array">The array to calculate the max of.</param>
/// <param name="arrayLength">The length of the array.</param>
__global__ void ComputePartialMaxes(REAL* partialMaxes, REAL* array, int arrayLength);

/// <summary>
/// Computes the final max from a given array of partial maxes. Requires 1 block of <paramref name="xLength" /> threads, and 1 REAL's worth of shared memory per thread.
/// </summary>
/// <param name="max">The location to place the output.</param>
/// <param name="partialMaxes">An array of partial maxes, of size <paramref name="xLength" />.</param>
__global__ void ComputeFinalMax(REAL* max, REAL* partialMaxes, int xLength);

/// <summary>
/// Computes the max of a given field.The field's width and height must each be no larger than the max number of threads per block.
/// </summary>
/// <param name="max">The location to place the output</param>
/// <returns>An error code, or <c>cudaSuccess</c>.</returns>
cudaError_t FieldMax(REAL* max, cudaStream_t streamToUse, PointerWithPitch<REAL> field, int xLength, int yLength);

/// <summary>
/// Computes partial sums of an array. Requires 1 REAL's worth of shared memory per thread.
/// </summary>
/// <param name="partialMaxes">The output array, length equal to the number of blocks spawned.</param>
/// <param name="array">The array to calculate the sum of.</param>
/// <param name="arrayLength">The length of the array.</param>
__global__ void ComputePartialSums(REAL* partialMaxes, REAL* array, int arrayLength);

/// <summary>
/// Computes the final sum from a given array of partial sums. Requires 1 block of <paramref name="xLength" /> threads, and 1 REAL's worth of shared memory per thread.
/// </summary>
/// <param name="sum">The location to place the output.</param>
/// <param name="partialSums">An array of partial sums, of size <paramref name="xLength" />.</param>
__global__ void ComputeFinalSum(REAL* sum, REAL* partialSums, int xLength);

/// <summary>
/// Computes the sum of a given field.The field's width and height must each be no larger than the max number of threads per block.
/// </summary>
/// <param name="sum">The location to place the output</param>
/// <returns>An error code, or <c>cudaSuccess</c>.</returns>
cudaError_t FieldSum(REAL* sum, cudaStream_t streamToUse, PointerWithPitch<REAL> field, int xLength, int yLength);

#endif // !REDUCTION_KERNELS_CUH