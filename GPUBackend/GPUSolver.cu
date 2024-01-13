#include "GPUSolver.cuh"
#include "Init.h"
#include "Boundary.cuh"
#include "Computation.cuh"
#include "math.h"
#include <iostream>

constexpr int GPU_MIN_MAJOR_VERSION = 6;

GPUSolver::GPUSolver(SimulationParameters parameters, int iMax, int jMax) : Solver(parameters, iMax, jMax) {
    hVel = PointerWithPitch<REAL>();
    cudaMallocPitch(&hVel.ptr, &hVel.pitch, (jMax + 2) * sizeof(REAL), iMax + 2);

    vVel = PointerWithPitch<REAL>();
    cudaMallocPitch(&vVel.ptr, &vVel.pitch, (jMax + 2) * sizeof(REAL), iMax + 2);

    pressure = PointerWithPitch<REAL>();
    cudaMallocPitch(&pressure.ptr, &pressure.pitch, (jMax + 2) * sizeof(REAL), iMax + 2);

    RHS = PointerWithPitch<REAL>();
    cudaMallocPitch(&RHS.ptr, &RHS.pitch, (jMax + 2) * sizeof(REAL), iMax + 2);

    F = PointerWithPitch<REAL>();
    cudaMallocPitch(&F.ptr, &F.pitch, (jMax + 2) * sizeof(REAL), iMax + 2);

    G = PointerWithPitch<REAL>();
    cudaMallocPitch(&G.ptr, &G.pitch, (jMax + 2) * sizeof(REAL), iMax + 2);

    streamFunction = PointerWithPitch<REAL>();
    cudaMallocPitch(&streamFunction.ptr, &streamFunction.pitch, (jMax + 2) * sizeof(REAL), iMax + 2);

    devFlags = PointerWithPitch<BYTE>();
    cudaMallocPitch(&devFlags.ptr, &devFlags.pitch, (jMax + 2) * sizeof(BYTE), iMax + 2);
        
    hostFlags = FlagMatrixMAlloc(iMax + 2, jMax + 2);
    obstacles = nullptr;
    streams = nullptr;
}

GPUSolver::~GPUSolver() {
    if (streams != nullptr) {
        for (int i = 0; i < totalStreams; i++) {
            cudaStreamDestroy(streams[i]); // Destroy all of the streams
        }
    }

    cudaFree(hVel.ptr);
    cudaFree(vVel.ptr);
    cudaFree(pressure.ptr);
    cudaFree(RHS.ptr);
    cudaFree(F.ptr);
    cudaFree(G.ptr);
    cudaFree(streamFunction.ptr);
    cudaFree(devFlags.ptr);
    FreeMatrix(hostFlags, iMax + 2);
    FreeMatrix(obstacles, iMax + 2);
}

void GPUSolver::SetBlockDimensions()
{
    // The below code takes the square root of the number of threads, but if the number of threads per block is not a square it takes the powers of 2 either side of the square root.
    // For example, a maxThreadsPerBlock of 1024 would mean threadsPerBlock becomes 32 and 32, but a maxThreadsPerBlock of 512 would mean threadsPerBlock would become 32 and 16
    int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;
    int log2ThreadsPerBlock = (int)ceilf(log2f((float)maxThreadsPerBlock)); // Threads per block should be a power of 2, but ceil just in case
    int log2XThreadsPerBlock = (int)ceilf((float)log2ThreadsPerBlock / 2.0f); // Divide by 2, if log2(threadsPerBlock) was odd, ceil
    int log2YThreadsPerBlock = (int)floorf((float)log2ThreadsPerBlock / 2.0f); // As above, but floor for smaller one
    int xThreadsPerBlock = (int)powf((float)log2XThreadsPerBlock, 2); // Now exponentiate to get the actual number of threads
    int yThreadsPerBlock = (int)powf((float)log2YThreadsPerBlock, 2);
    threadsPerBlock = dim3(xThreadsPerBlock, yThreadsPerBlock);

    int blocksForIMax = (int)ceilf((float)iMax / threadsPerBlock.x);
    int blocksForJMax = (int)ceilf((float)jMax / threadsPerBlock.y);
    numBlocks = dim3(blocksForIMax, blocksForJMax);
}

void GPUSolver::CreatePointerArray(REAL** ptrArray, REAL* valueArray, int stride, int count)
{
    for (int i = 0; i < count; i++) {
        ptrArray[i] = valueArray + i * stride; // Set the pointer at the certain index to however far along the flattened array the next row is
    }
}

bool** GPUSolver::GetObstacles() {
    if (obstacles == nullptr) {
        obstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);
    }
    return obstacles;
}

void GPUSolver::ProcessObstacles() {
    SetFlags(obstacles, hostFlags, iMax + 2, jMax + 2); // SetFlags is done on the CPU


}

void GPUSolver::PerformSetup() {
    cudaGetDeviceProperties(&deviceProperties, 0);

    SetBlockDimensions();

    streams = new cudaStream_t[totalStreams];
    for (int i = 0; i < totalStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    delX = parameters.width / iMax;
    delY = parameters.height / jMax;
}

void GPUSolver::Timestep(REAL& simulationTime) {
    SetBoundaryConditions(streams, deviceProperties.maxThreadsPerBlock, hVel, vVel, devFlags, coordinates, coordinatesLength, iMax, jMax, parameters.inflowVelocity, parameters.surfaceFrictionalPermissibility);

    REAL* timestep;
    cudaMalloc(&timestep, sizeof(REAL));
    // Compute timestep
    //simulationTime += timestep;

    REAL* gamma;
    cudaMalloc(&gamma, sizeof(REAL));

    ComputeGamma(gamma, streams, threadsPerBlock.x * threadsPerBlock.y, hVel, vVel, iMax, jMax, timestep, delX, delY);

    ComputeFG(streams, threadsPerBlock, hVel, vVel, F, G, devFlags, iMax, jMax, timestep, delX, delY, parameters.bodyForces.x, parameters.bodyForces.y, gamma, parameters.reynoldsNo);
    
    ComputeRHS KERNEL_ARGS2(numBlocks, threadsPerBlock) (F, G, RHS, devFlags, iMax, jMax, timestep, delX, delY); // Tested working

    // Compute pressure Poisson

    ComputeVelocities(streams, threadsPerBlock, hVel, vVel, F, G, pressure, devFlags, iMax, jMax, timestep, delX, delY); // Tested working

    dim3 numBlocksForStreamCalc(INT_DIVIDE_ROUND_UP(iMax + 1, threadsPerBlock.x), INT_DIVIDE_ROUND_UP(jMax + 1, threadsPerBlock.y));
    ComputeStream KERNEL_ARGS2(numBlocksForStreamCalc, threadsPerBlock) (hVel, streamFunction, iMax, jMax, delY); // Untested

    // Perform memory copies asynchronously

    cudaFree(timestep);
    cudaFree(gamma);
}

bool GPUSolver::IsDeviceSupported() {
    int count;
    cudaGetDeviceCount(&count);
    if (count > 0) {
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, 0);
        if (properties.major >= GPU_MIN_MAJOR_VERSION) {
            return true;
        }
    }
    return false;
}