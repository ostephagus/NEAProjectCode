#pragma once
#include "Solver.h"
class CPUSolver :
    public Solver
{
    void Timestep(REAL& simulationTime); // Implementing abstract inherited method
};

