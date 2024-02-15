#include "GPUSolver.cuh"
#include "Init.h"
#include "Flags.h"
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
    cudaMallocPitch(&streamFunction.ptr, &streamFunction.pitch, (jMax + 1) * sizeof(REAL), iMax + 1);

    devFlags = PointerWithPitch<BYTE>();
    cudaMallocPitch(&devFlags.ptr, &devFlags.pitch, (jMax + 2) * sizeof(BYTE), iMax + 2);
    
    obstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);

    dragCalculator = DragCalculator();

    transmissionHVel = new REAL[iMax * jMax];
    transmissionVVel = new REAL[iMax * jMax];
    transmissionPressure = new REAL[iMax * jMax];
    transmissionStream = new REAL[iMax * jMax];

    copiedHVel = new REAL[(iMax + 2) * (jMax + 2)];
    copiedVVel = new REAL[(iMax + 2) * (jMax + 2)];
    copiedPressure = new REAL[(iMax + 2) * (jMax + 2)];
    copiedStream = new REAL[(iMax + 1) * (jMax + 1)];

    devCoordinates = nullptr; // Initialised in ProcessObstacles.
    streams = nullptr; // Initialised in PerformSetup.
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

    FreeMatrix(obstacles, iMax + 2);

    delete[] streams;

    delete[] transmissionHVel;
    delete[] transmissionVVel;
    delete[] transmissionPressure;
    delete[] transmissionStream;

    delete[] copiedHVel;
    delete[] copiedVVel;
    delete[] copiedPressure;
    delete[] copiedStream;
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
    int xThreadsPerBlock = (int)powf(2, (float)log2XThreadsPerBlock); // Now exponentiate to get the actual number of threads
    int yThreadsPerBlock = (int)powf(2, (float)log2YThreadsPerBlock);
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
    return transmissionHVel;
}

REAL* GPUSolver::GetVerticalVelocity() const {
    return transmissionVVel;
}

REAL* GPUSolver::GetPressure() const {
    return transmissionPressure;
}

REAL* GPUSolver::GetStreamFunction() const {
    return transmissionStream;
}

bool** GPUSolver::GetObstacles() const {
    return obstacles;
}

REAL GPUSolver::GetDragCoefficient() {
    return dragCalculator.GetDragCoefficient(streams[0], hVel, vVel, pressure, iMax, jMax, delX, delY, parameters.dynamicViscosity, parameters.fluidDensity, parameters.inflowVelocity);
}

void GPUSolver::ProcessObstacles() { // When this function is called, no streams have been created and block dimensions have not been calculated. Therefore, no kernels can be launched here.
    BYTE** hostFlags = FlagMatrixMAlloc(iMax + 2, jMax + 2);
    SetFlags(obstacles, hostFlags, iMax + 2, jMax + 2); // Set the flags on the host.

    uint2* hostCoordinates; // Obstacle coordinates are put here first, then copied to the GPU.
    FindBoundaryCells(hostFlags, hostCoordinates, coordinatesLength, iMax, jMax);

    numFluidCells = CountFluidCells(hostFlags, iMax, jMax);

    dragCalculator.ProcessObstacles(hostFlags, iMax, jMax, parameters.width / iMax, parameters.height / jMax);

    cudaMalloc(&devCoordinates, coordinatesLength * sizeof(uint2));

    cudaMemcpy(devCoordinates, hostCoordinates, coordinatesLength * sizeof(uint2), cudaMemcpyHostToDevice); // Copy the flags and coordinates arrays to the device.
    CopyFieldToDevice(devFlags, hostFlags, iMax + 2, jMax + 2);

    FreeMatrix(hostFlags, iMax + 2);
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
    int pressureIterations = 0;
    dim3 numBlocksForStreamCalc(INT_DIVIDE_ROUND_UP(iMax + 1, threadsPerBlock.x), INT_DIVIDE_ROUND_UP(jMax + 1, threadsPerBlock.y));

    // Perform computations
    if (SetBoundaryConditions(streams, threadsPerBlock.x * threadsPerBlock.y, hVel, vVel, devFlags, devCoordinates, coordinatesLength, iMax, jMax, parameters.inflowVelocity, parameters.surfaceFrictionalPermissibility) != cudaSuccess) goto free; // Illegal address

    cudaMalloc(&timestep, sizeof(REAL)); // Allocate a new device variable for timestep
    
    if (ComputeTimestep(timestep, streams, hVel, vVel, iMax, jMax, delX, delY, parameters.reynoldsNo, parameters.timeStepSafetyFactor) != cudaSuccess) goto free;

    hostTimestep = new REAL; // Copy the device timestep so it can be added to simulation time
    if (cudaMemcpyAsync(hostTimestep, timestep, sizeof(REAL), cudaMemcpyDeviceToHost, streams[computationStreams + 0]) != cudaSuccess) goto free;

    if (cudaMalloc(&gamma, sizeof(REAL)) != cudaSuccess) goto free; // Allocate gamma on the device and then calculate it
    if (ComputeGamma(gamma, streams, threadsPerBlock.x * threadsPerBlock.y, hVel, vVel, iMax, jMax, timestep, delX, delY) != cudaSuccess) goto free;

    if (ComputeFG(streams, threadsPerBlock, hVel, vVel, F, G, devFlags, iMax, jMax, timestep, delX, delY, parameters.bodyForces.x, parameters.bodyForces.y, gamma, parameters.reynoldsNo) != cudaSuccess) goto free;
    
    ComputeRHS KERNEL_ARGS(numBlocks, threadsPerBlock, 0, streams[0]) (F, G, RHS, devFlags, iMax, jMax, timestep, delX, delY); // ComputeRHS is simple enough not to need a wrapper
    if (cudaStreamSynchronize(streams[0]) != cudaSuccess) goto free; // Need to synchronise because pressure depends on RHS.

    pressureIterations = Poisson(streams, threadsPerBlock, pressure, RHS, devFlags, devCoordinates, coordinatesLength, numFluidCells, iMax, jMax, numColoursSOR, delX, delY, parameters.pressureResidualTolerance, parameters.pressureMinIterations, parameters.pressureMaxIterations, parameters.relaxationParameter, &pressureResidualNorm);
    if (pressureIterations == 0) goto free; // Here 0 is the error case.

    //printf("Number of iterations: %i, residual norm: %f.\n", pressureIterations, pressureResidualNorm);
    
    cudaMemcpy2DAsync(copiedPressure, (jMax + 2) * sizeof(REAL), pressure.ptr, pressure.pitch, (jMax + 2) * sizeof(REAL), iMax + 2, cudaMemcpyDeviceToHost, streams[computationStreams + 0]); // Pressure is unchanged after this point, so can copy it async

    if (ComputeVelocities(streams, threadsPerBlock, hVel, vVel, F, G, pressure, devFlags, iMax, jMax, timestep, delX, delY) != cudaSuccess) goto free;

    cudaMemcpy2DAsync(copiedHVel, (jMax + 2) * sizeof(REAL), hVel.ptr, hVel.pitch, (jMax + 2) * sizeof(REAL), iMax + 2, cudaMemcpyDeviceToHost, streams[computationStreams + 1]); // Velocities are unchanged after this point, copy them
    cudaMemcpy2DAsync(copiedVVel, (jMax + 2) * sizeof(REAL), vVel.ptr, vVel.pitch, (jMax + 2) * sizeof(REAL), iMax + 2, cudaMemcpyDeviceToHost, streams[computationStreams + 2]);

    ComputeStream KERNEL_ARGS(numBlocksForStreamCalc, threadsPerBlock, 0, streams[0]) (hVel, streamFunction, iMax, jMax, delY);

    cudaMemcpy2DAsync(copiedStream, (jMax + 1) * sizeof(REAL), streamFunction.ptr, streamFunction.pitch, (jMax + 1) * sizeof(REAL), iMax + 1, cudaMemcpyDeviceToHost, streams[computationStreams + 3]); // Stream function is the last to be calculated, copy it once it is ready.
    
    // Resize all of the fields for transmission.
    ResizeField(copiedHVel, iMax + 2, jMax + 2, 1, 1, transmissionHVel, iMax, jMax);
    ResizeField(copiedVVel, iMax + 2, jMax + 2, 1, 1, transmissionVVel, iMax, jMax);
    ResizeField(copiedPressure, iMax + 2, jMax + 2, 1, 1, transmissionPressure, iMax, jMax);
    ResizeField(copiedStream, iMax + 1, jMax + 1, 1, 1, transmissionStream, iMax, jMax);

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