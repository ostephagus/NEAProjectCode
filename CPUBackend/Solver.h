#ifndef SOLVER_H
#define SOLVER_H

#include "Definitions.h"

class Solver
{
protected:
    int iMax;
    int jMax;

    DoubleField velocities;
    REAL** pressure;
    REAL** RHS;
    REAL** streamFunction;
    DoubleField FG;

    SimulationParameters parameters;

    bool** obstacles;
    BYTE** flags;
    std::pair<int, int>* coordinates;
    int coordinatesLength;
    int numFluidCells;

public:
    /// <summary>
    /// Initialises the class's fields and parameters
    /// </summary>
    /// <param name="parameters">The parameters to use for simulation. This may be changed before calls to <see cref="Timestep" />.</param>
    /// <param name="iMax">The index of the rightmost fluid cell</param>
    /// <param name="jMax">The index of the topmost fluid cell</param>
    Solver(SimulationParameters parameters, int iMax, int jMax);

    SimulationParameters GetParameters() const;
    void SetParameters(SimulationParameters parameters);

    /// <summary>
    /// Embeds obstacles into the simulation domain. Assumes obstacles have already been set
    /// </summary>
    virtual void ProcessObstacles();

    /// <summary>
    /// Embeds <paramref name="obstacles" /> into the simulation domain
    /// </summary>
    /// <param name="obstacles">An array representing which cells are obstacle and which are fluid.</param>
    virtual void ProcessObstacles(bool** obstacles);

    /// <summary>
    /// Computes one timestep, solving each of the fields.
    /// </summary>
    /// <param name="simulationTime">The time that the simulation has been running, to be updated with the new time after the timestep has finished.</param>
    virtual void Timestep(REAL& simulationTime) = 0;
};
#endif // !SOLVER_H