#pragma once
#include "Solver.h"
class CPUSolver :
    public Solver
{
public:
    CPUSolver(SimulationParameters parameters, int iMax, int jMax);

    void Timestep(REAL& simulationTime); // Implementing abstract inherited method
};

