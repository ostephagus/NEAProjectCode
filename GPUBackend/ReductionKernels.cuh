#ifndef REDUCTION_KERNELS_CUH

#include "Definitions.cuh"

/// <summary>
/// Computes the max of a given field.The field's width and height must each be no larger than the max number of threads per block.
/// </summary>
/// <param name="max">The location to place the output</param>
/// <returns>An error code, or <c>cudaSuccess</c>.</returns>
cudaError_t FieldMax(REAL* max, cudaStream_t streamToUse, PointerWithPitch<REAL> field, int xLength, int yLength);

/// <summary>
/// Computes the sum of a given field.The field's width and height must each be no larger than the max number of threads per block.
/// </summary>
/// <param name="sum">The location to place the output</param>
/// <returns>An error code, or <c>cudaSuccess</c>.</returns>
cudaError_t FieldSum(REAL* sum, cudaStream_t streamToUse, PointerWithPitch<REAL> field, int xLength, int yLength);

#endif // !REDUCTION_KERNELS_CUH