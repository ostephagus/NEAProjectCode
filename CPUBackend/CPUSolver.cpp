#include "CPUSolver.h"
#include "Init.h"
#include "Boundary.h"
#include "Flags.h"
#include "Computation.h"
#include "Drag.h"

CPUSolver::CPUSolver(SimulationParameters parameters, int iMax, int jMax) : Solver(parameters, iMax, jMax) {
    velocities.x = MatrixMAlloc(iMax + 2, jMax + 2);
    velocities.y = MatrixMAlloc(iMax + 2, jMax + 2);

    pressure = MatrixMAlloc(iMax + 2, jMax + 2);
    RHS = MatrixMAlloc(iMax + 2, jMax + 2);
    streamFunction = MatrixMAlloc(iMax + 1, jMax + 1);

    FG.x = MatrixMAlloc(iMax + 2, jMax + 2);
    FG.y = MatrixMAlloc(iMax + 2, jMax + 2);

    obstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);
    flags = FlagMatrixMAlloc(iMax + 2, jMax + 2);

    flattenedHVel = new REAL[iMax * jMax];
    flattenedVVel = new REAL[iMax * jMax];
    flattenedPressure = new REAL[iMax * jMax];
    flattenedStream = new REAL[iMax * jMax];

    coordinates = nullptr;
    coordinatesLength = 0;
    numFluidCells = 0;
    stepSizes = DoubleReal(0, 0);
}

CPUSolver::~CPUSolver() {
    FreeMatrix(velocities.x, iMax + 2);
    FreeMatrix(velocities.y, iMax + 2);
    FreeMatrix(pressure, iMax + 2);
    FreeMatrix(RHS, iMax + 2);
    FreeMatrix(streamFunction, iMax + 1);
    FreeMatrix(FG.x, iMax + 2);
    FreeMatrix(FG.y, iMax + 2);
    FreeMatrix(obstacles, iMax + 2);
    FreeMatrix(flags, iMax + 2);

    delete[] flattenedHVel;
    delete[] flattenedVVel;
    delete[] flattenedPressure;
    delete[] flattenedStream;
}

REAL* CPUSolver::GetHorizontalVelocity() const {
    return flattenedHVel;
}

REAL* CPUSolver::GetVerticalVelocity() const {
    return flattenedVVel;
}

REAL* CPUSolver::GetPressure() const {
    return flattenedPressure;
}

REAL* CPUSolver::GetStreamFunction() const {
    return flattenedStream;
}

bool** CPUSolver::GetObstacles() const {
    return obstacles;
}

REAL CPUSolver::GetDragCoefficient() {
    return ComputeDragCoefficient(velocities, pressure, flags, coordinates, coordinatesLength, iMax, jMax, stepSizes, parameters.dynamicViscosity, parameters.fluidDensity, parameters.inflowVelocity);
}

void CPUSolver::ProcessObstacles() {
    SetFlags(obstacles, flags, iMax + 2, jMax + 2);

    std::pair<std::pair<int, int>*, int> coordinatesWithLength = FindBoundaryCells(flags, iMax, jMax);
    coordinates = coordinatesWithLength.first;
    coordinatesLength = coordinatesWithLength.second;

    numFluidCells = CountFluidCells(flags, iMax, jMax);
}

void CPUSolver::PerformSetup() {
    stepSizes.x = parameters.width / iMax;
    stepSizes.y = parameters.height / jMax;
}

void CPUSolver::Timestep(REAL& simulationTime) {
    SetBoundaryConditions(velocities, flags, coordinates, coordinatesLength, iMax, jMax, parameters.inflowVelocity, parameters.surfaceFrictionalPermissibility);

    REAL timestep;
    ComputeTimestep(timestep, iMax, jMax, stepSizes, velocities, parameters.reynoldsNo, parameters.timeStepSafetyFactor);
    simulationTime += timestep;

    REAL gamma = ComputeGamma(velocities, iMax, jMax, timestep, stepSizes);
    ComputeFG(velocities, FG, flags, iMax, jMax, timestep, stepSizes, parameters.bodyForces, gamma, parameters.reynoldsNo);

    ComputeRHS(FG, RHS, flags, iMax, jMax, timestep, stepSizes);
    REAL pressureResidualNorm = 0;
    (void)PoissonMultiThreaded(pressure, RHS, flags, coordinates, coordinatesLength, numFluidCells, iMax, jMax, stepSizes, parameters.pressureResidualTolerance, parameters.pressureMinIterations, parameters.pressureMaxIterations, parameters.relaxationParameter, pressureResidualNorm);

    ComputeVelocities(velocities, FG, pressure, flags, iMax, jMax, timestep, stepSizes);
    ComputeStream(velocities, streamFunction, iMax, jMax, stepSizes);

    // Copy all of the 2D arrays to flattened arrays.
    // Parameters:     2D array        2D array offsets|flattened array and offsets|size of copy domain    
    FlattenArray<REAL>(velocities.x,   1, 1,            flattenedHVel,     0, 0, 0, iMax, jMax);
    FlattenArray<REAL>(velocities.y,   1, 1,            flattenedVVel,     0, 0, 0, iMax, jMax);
    FlattenArray<REAL>(pressure,       1, 1,            flattenedPressure, 0, 0, 0, iMax, jMax);
    FlattenArray<REAL>(streamFunction, 0, 0,            flattenedStream,   0, 0, 0, iMax, jMax);
}