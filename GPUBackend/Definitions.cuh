#ifndef DEFINITIONS_CUH
#define DEFINITIONS_CUH

#include "Definitions.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define F_PITCHACCESS(basePtr, pitch, i, j) (*((REAL*)((char*)(basePtr) + (i) * (pitch)) + (j))) // Used for accessing a location in a pitched array (F for float, FP for float pointer, B for byte, BP for byte pointer.)
#define FP_PITCHACCESS(basePtr, pitch, i, j) ((REAL*)((char*)(basePtr) + (i) * (pitch)) + (j)) // Used for accessing a location in a pitched array (F for float, FP for float pointer, B for byte, BP for byte pointer.)
#define B_PITCHACCESS(basePtr, pitch, i, j) (*((basePtr) + (i) * (pitch) + (j))) // Used for accessing a location in a pitched array (F for float, FP for float pointer, B for byte, BP for byte pointer.)
#define BP_PITCHACCESS(basePtr, pitch, i, j) ((basePtr) + (i) * (pitch) + (j)) // Used for accessing a location in a pitched array (F for float, FP for float pointer, B for byte, BP for byte pointer.)

// Horrific macros to make intellisense stop complaining about the triple angle bracket syntax for kernel launches
#ifndef __INTELLISENSE__
#define KERNEL_ARGS(numBlocks, numThreads, sh_mem, stream) <<< numBlocks, numThreads, sh_mem, stream >>> // Launch a kernel with shared memory and stream specified.
#else
#define KERNEL_ARGS(numBlocks, numThreads, sh_mem, stream) // Launch a kernel with shared memory and stream specified.
#endif

#define INT_DIVIDE_ROUND_UP(numerator, denominator) (((numerator) + (denominator) - 1) / (denominator))

#define make_REAL2 make_float2
typedef float2 REAL2;

template <typename T>
struct PointerWithPitch
{
    T* ptr;
    size_t pitch;
};

struct DragCoordinate
{
    uint2 coordinate;
    REAL2 unitNormal;
    REAL stepSize;
};

#endif // !DEFINITIONS_CUH