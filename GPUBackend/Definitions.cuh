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

// Constants used for parsing of flags.
constexpr BYTE SELF  = 0b00010000; // SELF bit
constexpr BYTE NORTH = 0b00001000; // NORTH bit
constexpr BYTE EAST  = 0b00000100; // EAST bit
constexpr BYTE SOUTH = 0b00000010; // SOUTH bit
constexpr BYTE WEST  = 0b00000001; // WEST bit

constexpr BYTE SELFSHIFT  = 4; // Amount to shift for SELF bit at LSB.
constexpr BYTE NORTHSHIFT = 3; // Amount to shift for NORTH bit at LSB.
constexpr BYTE EASTSHIFT  = 2; // Amount to shift for EAST bit at LSB.
constexpr BYTE SOUTHSHIFT = 1; // Amount to shift for SOUTH bit at LSB.
constexpr BYTE WESTSHIFT  = 0; // Amount to shift for WEST bit at LSB.

template <typename T>
struct PointerWithPitch
{
    T* ptr;
    size_t pitch;
};

#endif // !DEFINITIONS_CUH