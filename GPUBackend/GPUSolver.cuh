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
    PointerWithPitch<bool> devObstacles; // Boolean obstacles array, resides on device.

    REAL* copiedHVel;
    REAL* copiedVVel;
    REAL* copiedPressure;
    REAL* copiedStream;
    REAL* copiedEnlargedHVel;
    REAL* copiedEnlargedVVel;
    REAL* copiedEnlargedPressure;
    REAL* copiedEnlargedStream;

    REAL delX; // Step size in x direction, resides on host.
    REAL delY; // Step size in y direction, resides on host.
    REAL* timestep; // Timestep, resides on device.

    uint2* devCoordinates; // Array of obstacle coordinates, resides on device.
    int coordinatesLength; // Length of coordinates array
    int numFluidCells;
    const int numColoursSOR = 2;

    dim3 numBlocks; // Number of blocks for a grid of iMax x jMax threads.
    dim3 threadsPerBlock; // Maximum number of threads per block in a 2D square allocation.

    bool** hostObstacles; // 2D array of obstacles, resides on host.
    BYTE** hostFlags; // 2D array of flags, resides on host.

    cudaDeviceProp deviceProperties;
    cudaStream_t* streams; // Streams that can be used. First are the computation streams (0 to the number of computation streams), then are memcpy streams. To access memcpy streams, first add computationStreams then an offset

    template<typename T>
    cudaError_t CopyFieldToDevice(PointerWithPitch<T> devField, T** hostField, int xLength, int yLength);

    template<typename T>
    cudaError_t CopyFieldToHost(PointerWithPitch<T> devField, T** hostField, int xLength, int yLength);

    void SetBlockDimensions();

    void ResizeField(REAL* enlargedField, int enlargedXLength, int enlargedYLength, int xOffset, int yOffset, REAL* transmissionField, int xLength, int yLength);

public:
    GPUSolver(SimulationParameters parameters, int iMax, int jMax);

    ~GPUSolver();

    REAL* GetHorizontalVelocity() const;

    REAL* GetVerticalVelocity() const;

    REAL* GetPressure() const;

    REAL* GetStreamFunction() const;

    bool** GetObstacles();

    void ProcessObstacles();

    void PerformSetup();

    void Timestep(REAL& simulationTime); // Implementing abstract inherited method

    static bool IsDeviceSupported();
};

#endif // !GPUSOLVER_CUH
