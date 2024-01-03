#ifndef GPUSOLVER_CUH
#define GPUSOLVER_CUH

#include "Definitions.cuh"
#include "Solver.h"

class GPUSolver :
    public Solver
{
private:
    pointerWithPitch<REAL> hVel; // Horizontal velocity, resides on device.
    pointerWithPitch<REAL> vVel; // Vertical velocity, resides on device.
    pointerWithPitch<REAL> pressure; // Pressure, resides on device.
    pointerWithPitch<REAL> RHS; // Pressure equation RHS, resides on device.
    pointerWithPitch<REAL> streamFunction; // Stream function, resides on device.
    pointerWithPitch<REAL> F; // Quantity F, resides on device.
    pointerWithPitch<REAL> G; // Quantity G, resides on device.
    pointerWithPitch<BYTE> devFlags; // Cell flags, resides on device.

    REAL delX; // Step size in x direction, resides on host.
    REAL delY; // Step size in y direction, resides on host.
    REAL* timestep; // Timestep, resides on device.

    uint2* coordinates;
    int coordinatesLength;

    dim3 numBlocks;

    dim3 threadsPerBlock;

    bool** obstacles; // 2D array of obstacles, resides on host.
    BYTE** hostFlags; // 2D array of flags, resides on host.

    cudaDeviceProp deviceProperties;
    cudaStream_t** streams;

    void SetBlockDimensions();

    void CreatePointerArray(REAL** ptrArray, REAL* valueArray, int stride, int count);

public:
    GPUSolver(SimulationParameters parameters, int iMax, int jMax);

    ~GPUSolver();

    bool** GetObstacles();

    void ProcessObstacles();


    void PerformSetup();

    void Timestep(REAL& simulationTime); // Implementing abstract inherited method
};

#endif // !GPUSOLVER_CUH