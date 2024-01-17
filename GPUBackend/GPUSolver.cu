#include "GPUSolver.cuh"
#include "Init.h"
#include "Boundary.cuh"
#include "Computation.cuh"
#include "PressureComputation.cuh"
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

    copiedHVel = new REAL[iMax * jMax];
    copiedVVel = new REAL[iMax * jMax];
    copiedPressure = new REAL[iMax * jMax];
    copiedStream = new REAL[iMax * jMax];

    copiedEnlargedHVel = new REAL[(iMax + 2) * (jMax + 2)];
    copiedEnlargedVVel = new REAL[(iMax + 2) * (jMax + 2)];
    copiedEnlargedPressure = new REAL[(iMax + 2) * (jMax + 2)];
    copiedEnlargedStream = new REAL[(iMax + 1) * (jMax + 2)];

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

    delete[] copiedHVel;
    delete[] copiedVVel;
    delete[] copiedPressure;
    delete[] copiedStream;

    delete[] copiedEnlargedHVel;
    delete[] copiedEnlargedVVel;
    delete[] copiedEnlargedPressure;
    delete[] copiedEnlargedStream;
}

template<typename T>
cudaError_t GPUSolver::CopyFieldToDevice(PointerWithPitch<T> devField, T** hostField, int xLength, int yLength)
{
    T* hostFieldFlattened = new T[xLength * yLength];
    FlattenArray(hostField, 0, 0, hostFieldFlattened, 0, 0, 0, xLength, yLength);

    cudaError_t retVal = cudaMemcpy2D(devField.ptr, devField.pitch, hostFieldFlattened, yLength * sizeof(T), yLength * sizeof(T), xLength, cudaMemcpyHostToDevice);
    delete[] hostFieldFlattened;

    return retVal;
}

template<typename T>
cudaError_t GPUSolver::CopyFieldToHost(PointerWithPitch<T> devField, T** hostField, int xLength, int yLength) {
    T* hostFieldFlattened = new T[xLength * yLength];

    cudaError_t retVal = cudaMemcpy2D(hostFieldFlattened, yLength * sizeof(T), devField.ptr, devField.pitch, yLength * sizeof(T), xLength, cudaMemcpyDeviceToHost);

    UnflattenArray(hostField, 0, 0, hostFieldFlattened, 0, 0, 0, xLength, yLength);
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

void GPUSolver::ResizeField(REAL* enlargedField, int enlargedXLength, int enlargedYLength, int xOffset, int yOffset, REAL* transmissionField, int xLength, int yLength) {
    for (int i = 0; i < xLength; i++) {
        memcpy(
            transmissionField + i * yLength,
            enlargedField + (i + xOffset) * enlargedYLength + yOffset,
            yLength * sizeof(REAL)
        );
    }
}

REAL* GPUSolver::GetHorizontalVelocity() const {
    return copiedHVel;
}

REAL* GPUSolver::GetVerticalVelocity() const {
    return copiedVVel;
}

REAL* GPUSolver::GetPressure() const {
    return copiedPressure;
}

REAL* GPUSolver::GetStreamFunction() const {
    return copiedStream;
}

bool** GPUSolver::GetObstacles() {
    if (hostObstacles == nullptr) {
        hostObstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);
    }
    return hostObstacles;
}

void GPUSolver::ProcessObstacles() {
    CopyFieldToDevice(devObstacles, hostObstacles, iMax + 2, jMax + 2); // Copies obstacles to device to do SetFlags

    SetFlags KERNEL_ARGS(numBlocks, threadsPerBlock, 0, 0 /*default stream*/) (devObstacles, devFlags, iMax, jMax);

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
    REAL* hostTimestep = nullptr; // Set heap or global mem pointers to nullptr so freeing has no effect and only free label is needed.
    REAL* timestep = nullptr;
    REAL* gamma = nullptr;
    REAL pressureResidualNorm = 0;
    dim3 numBlocksForStreamCalc(INT_DIVIDE_ROUND_UP(iMax + 1, threadsPerBlock.x), INT_DIVIDE_ROUND_UP(jMax + 1, threadsPerBlock.y));

    // Perform computations
    if (SetBoundaryConditions(streams, deviceProperties.maxThreadsPerBlock, hVel, vVel, devFlags, devCoordinates, coordinatesLength, iMax, jMax, parameters.inflowVelocity, parameters.surfaceFrictionalPermissibility) != cudaSuccess) goto free;

    cudaMalloc(&timestep, sizeof(REAL)); // Allocate a new device variable for timestep
    
    if (ComputeTimestep(timestep, streams, hVel, vVel, iMax, jMax, delX, delY, parameters.reynoldsNo, parameters.timeStepSafetyFactor) != cudaSuccess) goto free;

    hostTimestep = new REAL; // Copy the device timestep so it can be added to simulation time
    if (cudaMemcpyAsync(hostTimestep, timestep, sizeof(REAL), cudaMemcpyDeviceToHost, streams[computationStreams + 0]) != cudaSuccess) goto free;

    if (cudaMalloc(&gamma, sizeof(REAL)) != cudaSuccess) goto free; // Allocate gamma on the device and then calculate it
    if (ComputeGamma(gamma, streams, threadsPerBlock.x * threadsPerBlock.y, hVel, vVel, iMax, jMax, timestep, delX, delY) != cudaSuccess) goto free;

    if (ComputeFG(streams, threadsPerBlock, hVel, vVel, F, G, devFlags, iMax, jMax, timestep, delX, delY, parameters.bodyForces.x, parameters.bodyForces.y, gamma, parameters.reynoldsNo) != cudaSuccess) goto free;
    
    ComputeRHS KERNEL_ARGS(numBlocks, threadsPerBlock, 0, streams[0]) (F, G, RHS, devFlags, iMax, jMax, timestep, delX, delY); // ComputeRHS is simple enough not to need a wrapper
    if (cudaStreamSynchronize(streams[0]) != cudaSuccess) goto free; // Need to synchronise because pressure depends on RHS.

    if (Poisson(streams, threadsPerBlock, pressure, RHS, devFlags, devCoordinates, coordinatesLength, numFluidCells, iMax, jMax, numColoursSOR, delX, delY, parameters.pressureResidualTolerance, parameters.pressureMinIterations, parameters.pressureMaxIterations, parameters.relaxationParameter, &pressureResidualNorm) == 0) goto free; // Here 0 is the error case.
    
    cudaMemcpy2DAsync(copiedEnlargedPressure, (jMax + 2) * sizeof(REAL), pressure.ptr, pressure.pitch, (jMax + 2) * sizeof(REAL), iMax + 2, cudaMemcpyDeviceToHost, streams[computationStreams + 0]); // Pressure is unchanged after this point, so can copy it async

    if (ComputeVelocities(streams, threadsPerBlock, hVel, vVel, F, G, pressure, devFlags, iMax, jMax, timestep, delX, delY) != cudaSuccess) goto free;

    cudaMemcpy2DAsync(copiedEnlargedHVel, (jMax + 2) * sizeof(REAL), hVel.ptr, hVel.pitch, (jMax + 2) * sizeof(REAL), iMax + 2, cudaMemcpyDeviceToHost, streams[computationStreams + 1]); // Velocities are unchanged after this point, copy them
    cudaMemcpy2DAsync(copiedEnlargedVVel, (jMax + 2) * sizeof(REAL), vVel.ptr, vVel.pitch, (jMax + 2) * sizeof(REAL), iMax + 2, cudaMemcpyDeviceToHost, streams[computationStreams + 2]);

    ComputeStream KERNEL_ARGS(numBlocksForStreamCalc, threadsPerBlock, 0, streams[0]) (hVel, streamFunction, iMax, jMax, delY);

    cudaMemcpy2DAsync(copiedEnlargedStream, (jMax + 1) * sizeof(REAL), streamFunction.ptr, streamFunction.pitch, (jMax + 1) * sizeof(REAL), iMax + 1, cudaMemcpyDeviceToHost, streams[computationStreams + 3]); // Stream function is the last to be calculated, copy it once it is ready.
    
    // Resize all of the fields for transmission.
    ResizeField(copiedEnlargedHVel, iMax + 2, jMax + 2, 1, 1, copiedHVel, iMax, jMax);
    ResizeField(copiedEnlargedVVel, iMax + 2, jMax + 2, 1, 1, copiedVVel, iMax, jMax);
    ResizeField(copiedEnlargedPressure, iMax + 2, jMax + 2, 1, 1, copiedPressure, iMax, jMax);
    ResizeField(copiedEnlargedStream, iMax + 1, jMax + 1, 1, 1, copiedStream, iMax, jMax);

    simulationTime += *hostTimestep; // Only add to the simulation time if the timestep was successful. Error case is therefore simulationTime unchanged after Timestep returns.


free: // Pointers that need to be freed even if timestep is unsuccessful.
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