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
    
    devObstacles = PointerWithPitch<bool>();
    cudaMallocPitch(&devObstacles.ptr, &devObstacles.pitch, (jMax + 2) * sizeof(bool), iMax + 2);

    devCoordinates = nullptr;
    hostObstacles = nullptr;
    hostFlags = nullptr;
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
    cudaFree(devCoordinates);

    FreeMatrix(hostObstacles, iMax + 2);
    FreeMatrix(hostFlags, iMax + 2);
    delete[] streams;
}
template<typename T>
void GPUSolver::UnflattenArray(T** pointerArray, T* flattenedArray, int length, int divisions) {
    for (int i = 0; i < length / divisions; i++) {

        memcpy(
            pointerArray[i],                // Destination address - address at ith pointer
            flattenedArray + i * divisions, // Source start address - move (i * divisions) each iteration
            divisions * sizeof(T)           // Bytes to copy - divisions
        );

    }
}

template<typename T>
void GPUSolver::FlattenArray(T** pointerArray, T* flattenedArray, int length, int divisions) {
    for (int i = 0; i < length / divisions; i++) {

        memcpy(
            flattenedArray + i * divisions, // Destination address - move (i * divisions) each iteration
            pointerArray[i],                // Source start address - address at ith pointer
            divisions * sizeof(T)           // Bytes to copy - divisions
        );

    }
}

template<typename T>
cudaError_t GPUSolver::CopyFieldToDevice(PointerWithPitch<T> devField, T** hostField, int xLength, int yLength)
{
    T* hostFieldFlattened = new T[xLength * yLength];
    FlattenArray(hostField, hostFieldFlattened, xLength * yLength, yLength);

    cudaError_t retVal = cudaMemcpy2D(devField.ptr, devField.pitch, hostFieldFlattened, yLength * sizeof(T), yLength * sizeof(T), xLength, cudaMemcpyHostToDevice);
    delete[] hostFieldFlattened;

    return retVal;
}

template<typename T>
cudaError_t GPUSolver::CopyFieldToHost(PointerWithPitch<T> devField, T** hostField, int xLength, int yLength) {
    T* hostFieldFlattened = new T[xLength * yLength];

    cudaError_t retVal = cudaMemcpy2D(hostFieldFlattened, yLength * sizeof(T), devField.ptr, devField.pitch, yLength * sizeof(T), xLength, cudaMemcpyDeviceToHost);

    UnflattenArray(hostField, hostFieldFlattened, xLength * yLength, yLength);
    delete[] hostFieldFlattened;

    return retVal;
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

bool** GPUSolver::GetObstacles() {
    if (hostObstacles == nullptr) {
        hostObstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);
    }
    return hostObstacles;
}

void GPUSolver::ProcessObstacles() {
    CopyFieldToDevice(devObstacles, hostObstacles, iMax + 2, jMax + 2); // Copies obstacles to device to do SetFlags

    SetFlags KERNEL_ARGS2(numBlocks, threadsPerBlock) (devObstacles, devFlags, iMax, jMax);

    hostFlags = FlagMatrixMAlloc(iMax + 2, jMax + 2);
    CopyFieldToHost(devFlags, hostFlags, iMax + 2, jMax + 2); // Copy the flags back to do some CPU-specific processing

    uint2* hostCoordinates; // Obstacle coordinates are put here first, then copied to the GPU.
    FindBoundaryCells(hostFlags, hostCoordinates, coordinatesLength, iMax, jMax);

    numFluidCells = CountFluidCells(hostFlags, iMax, jMax);

    cudaMalloc(&devCoordinates, coordinatesLength * sizeof(uint2));
    cudaMemcpy(devCoordinates, hostCoordinates, coordinatesLength * sizeof(uint2), cudaMemcpyHostToDevice);

    delete[] hostCoordinates;
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
    SetBoundaryConditions(streams, deviceProperties.maxThreadsPerBlock, hVel, vVel, devFlags, devCoordinates, coordinatesLength, iMax, jMax, parameters.inflowVelocity, parameters.surfaceFrictionalPermissibility);

    REAL* timestep;
    cudaMalloc(&timestep, sizeof(REAL));
    
    ComputeTimestep(timestep, streams, hVel, vVel, iMax, jMax, delX, delY, parameters.reynoldsNo, parameters.timeStepSafetyFactor);
    REAL* hostTimestep = new REAL;
    cudaMemcpyAsync(hostTimestep, timestep, sizeof(REAL), cudaMemcpyDeviceToHost, *(streams + computationStreams));

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

    simulationTime += *hostTimestep;
    delete hostTimestep;

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