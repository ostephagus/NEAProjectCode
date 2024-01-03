#ifndef SOLVER_H
#define SOLVER_H

#include "pch.h"

class Solver
{
protected:
    int iMax;
    int jMax;

    SimulationParameters parameters;

public:
    /// <summary>
    /// Initialises the class's fields and parameters
    /// </summary>
    /// <param name="parameters">The parameters to use for simulation. This may be changed before calls to <see cref="Timestep" />.</param>
    /// <param name="iMax">The index of the rightmost fluid cell</param>
    /// <param name="jMax">The index of the topmost fluid cell</param>
    Solver(SimulationParameters parameters, int iMax, int jMax);

    ~Solver();

    SimulationParameters GetParameters() const;
    void SetParameters(SimulationParameters parameters);

    int GetIMax() const;
    int GetJMax() const;

    virtual REAL** GetHorizontalVelocity() const = 0;

    virtual REAL** GetVerticalVelocity() const = 0;

    virtual REAL** GetPressure() const = 0;

    virtual REAL** GetStreamFunction() const = 0;

    virtual bool** GetObstacles() = 0;

    /// <summary>
    /// Embeds obstacles into the simulation domain. Assumes obstacles have already been set
    /// </summary>
    virtual void ProcessObstacles() = 0;

    /// <summary>
    /// Performs setup before calls to timestep are made.
    /// </summary>
    virtual void PerformSetup() = 0;

    /// <summary>
    /// Computes one timestep, solving each of the fields.
    /// </summary>
    /// <param name="simulationTime">The time that the simulation has been running, to be updated with the new time after the timestep has finished.</param>
    virtual void Timestep(REAL& simulationTime) = 0;
};
#endif // !SOLVER_H