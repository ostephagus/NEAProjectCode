#include "pch.h"
#include "Solver.h"
#include "GPUSolver.cuh"
#include "BackendCoordinator.h"

int main(int argc, char** argv) {
    int iMax = 100;
    int jMax = 100;
    SimulationParameters parameters = SimulationParameters();
    Solver* solver = new GPUSolver(parameters, iMax, jMax);
    BackendCoordinator backendCoordinator(iMax, jMax, std::string("NEAFluidDynamicsPipe"), solver);
    int retValue = backendCoordinator.Run();
    delete solver;
    return retValue;
}
