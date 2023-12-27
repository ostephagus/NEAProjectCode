#include "CPUSolver.h"
#include "Boundary.h" // SetBoundaryConditions
#include "Computation.h" // ComputeTimestep, ComputeFG, ComputeRHS, PoissonMultiThreaded, ComputeVelocities, ComputeStream

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
    ComputeStream(velocities, streamFunction, flags, iMax, jMax, stepSizes);
}