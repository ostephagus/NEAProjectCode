#ifndef DEFINITIONS_CUH
#define DEFINITIONS_CUH

#include "Definitions.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define F_PITCHACCESS(basePtr, pitch, i, j) ((REAL*)((char*)(basePtr) + (i) * (pitch)) + (j)) // Used for accessing a location in a pitched array (F for float, B for byte)
#define B_PITCHACCESS(basePtr, pitch, i, j) ((basePtr) + (i) * (pitch) + (j)) // Used for accessing a location in a pitched array (F for float, B for byte)

// Horrific macros to make intellisense stop complaining about the triple angle bracket syntax for kernel launches
#ifndef __INTELLISENSE__
#define KERNEL_ARGS2(numBlocks, numThreads) <<< numBlocks, numThreads >>> // Launch a kernel with no shared memory and the default stream.
#define KERNEL_ARGS3(numBlocks, numThreads, sh_mem) <<< numBlocks, numThreads, sh_mem >>> // Launch a kernel with a shared memory allocation and the default stream.
#define KERNEL_ARGS4(numBlocks, numThreads, sh_mem, stream) <<< numBlocks, numThreads, sh_mem, stream >>> // Launch a kernel with shared memory and stream specified.
#else
#define KERNEL_ARGS2(numBlocks, numThreads) // Launch a kernel with no shared memory and the default stream.
#define KERNEL_ARGS3(numBlocks, numThreads, sh_mem) // Launch a kernel with a shared memory allocation and the default stream.
#define KERNEL_ARGS4(numBlocks, numThreads, sh_mem, stream) // Launch a kernel with shared memory and stream specified.
#endif

#define INT_DIVIDE_ROUND_UP(numerator, denominator) (((numerator) + (denominator) - 1) / (denominator))

template <typename T>
struct PointerWithPitch
{
    T* ptr;
    size_t pitch;
};

#endif // !DEFINITIONS_CUH