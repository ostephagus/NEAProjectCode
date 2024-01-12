#ifndef GPUSOLVER_CUH
#define GPUSOLVER_CUH

#include "Definitions.cuh"
#include "Solver.h"

constexpr int computationStreams = 4; // Number of streams for launching parallelisable computation kernels
constexpr int memcpyStreams = 4; // Number of streams for launching parallel memory copies
constexpr int totalStreams = computationStreams + memcpyStreams;

class GPUSolver :
    public Solver
{
private:
    PointerWithPitch<REAL> hVel; // Horizontal velocity, resides on device.
    PointerWithPitch<REAL> vVel; // Vertical velocity, resides on device.
    PointerWithPitch<REAL> pressure; // Pressure, resides on device.
    PointerWithPitch<REAL> RHS; // Pressure equation RHS, resides on device.
    PointerWithPitch<REAL> streamFunction; // Stream function, resides on device.
    PointerWithPitch<REAL> F; // Quantity F, resides on device.
    PointerWithPitch<REAL> G; // Quantity G, resides on device.
    PointerWithPitch<BYTE> devFlags; // Cell flags, resides on device.

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
    cudaStream_t* streams; // Streams that can be used. First are the computation streams (0 to the number of computation streams), then are memcpy streams. To access memcpy streams, first add computationStreams then an offset

    void SetBlockDimensions();

    void CreatePointerArray(REAL** ptrArray, REAL* valueArray, int stride, int count);

public:
    GPUSolver(SimulationParameters parameters, int iMax, int jMax);

    ~GPUSolver();

    bool** GetObstacles();

    void ProcessObstacles();

    void PerformSetup();

    void Timestep(REAL& simulationTime); // Implementing abstract inherited method

    static bool IsDeviceSupported();
};

#endif // !GPUSOLVER_CUH