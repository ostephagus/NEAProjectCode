#include "CPUSolver.h"
#include "Init.h" // MatrixMAlloc, FlagMatrixMAlloc
#include "Boundary.h" // SetBoundaryConditions
#include "Computation.h" // ComputeTimestep, ComputeFG, ComputeRHS, PoissonMultiThreaded, ComputeVelocities, ComputeStream

CPUSolver::CPUSolver(SimulationParameters parameters, int iMax, int jMax) : Solver(parameters, iMax, jMax) {
    velocities.x = MatrixMAlloc(iMax + 2, jMax + 2);
    velocities.y = MatrixMAlloc(iMax + 2, jMax + 2);

    pressure = MatrixMAlloc(iMax + 2, jMax + 2);
    RHS = MatrixMAlloc(iMax + 2, jMax + 2);
    streamFunction = MatrixMAlloc(iMax + 1, jMax + 1);

    FG.x = MatrixMAlloc(iMax + 2, jMax + 2);
    FG.y = MatrixMAlloc(iMax + 2, jMax + 2);

    flags = FlagMatrixMAlloc(iMax + 2, jMax + 2);
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
}

bool** CPUSolver::GetObstacles() {
    if (obstacles == nullptr) {
        obstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);
    }
    return obstacles;
}

void CPUSolver::ProcessObstacles() {
    SetFlags(obstacles, flags, iMax + 2, jMax + 2);

    std::pair<std::pair<int, int>*, int> coordinatesWithLength = FindBoundaryCells(flags, iMax, jMax);
    coordinates = coordinatesWithLength.first;
    coordinatesLength = coordinatesWithLength.second;

    numFluidCells = CountFluidCells(flags, iMax, jMax);
}

void CPUSolver::Timestep(REAL& simulationTime) {
    DoubleReal stepSizes = DoubleReal();
    stepSizes.x = parameters.width / iMax;
    stepSizes.y = parameters.height / jMax;

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
}