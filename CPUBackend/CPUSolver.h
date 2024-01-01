#ifndef CPUSOLVER_H
#define CPUSOLVER_H

#include "Solver.h"
class CPUSolver :
    public Solver
{
public:
    CPUSolver(SimulationParameters parameters, int iMax, int jMax);

    ~CPUSolver();

    bool** GetObstacles();

    void ProcessObstacles();

    void Timestep(REAL& simulationTime); // Implementing abstract inherited method
};

#endif // !CPUSOLVER_H

