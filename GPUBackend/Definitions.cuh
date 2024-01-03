#ifndef DEFINITIONS_CUH
#define DEFINITIONS_CUH

#include "Definitions.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define F_PITCHACCESS(basePtr, pitch, i, j) ((REAL*)((char*)(basePtr) + (i) * (pitch)) + (j)) // Used for accessing a location in a pitched array (F for float, B for byte)
#define B_PITCHACCESS(basePtr, pitch, i, j) (basePtr) + (i) * (pitch) + (j) // Used for accessing a location in a pitched array (F for float, B for byte)

template <typename T>
struct PointerWithPitch
{
    T* ptr;
    size_t pitch;
};

#endif // !DEFINITIONS_CUH