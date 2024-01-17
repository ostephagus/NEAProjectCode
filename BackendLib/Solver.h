#ifndef SOLVER_H
#define SOLVER_H

#include "pch.h"

class Solver
{
protected:
    int iMax;
    int jMax;

    SimulationParameters parameters;
    
    /// <summary>
    /// Unflattens the array specified in <paramref name="flattenedArray" />, storing the result in <paramref name="pointerArray" />.
    /// </summary>
    /// <typeparam name="T">The type of the elements in the array.</typeparam>
    /// <param name="pointerArray">The output 2D array.</param>
    /// <param name="paDownOffset">The number of elements below the region of a column to be copied into the pointer array.</param>
    /// <param name="paLeftOffset">The number of elements left of the region of a row to be copied into the pointer array.</param>
    /// <param name="flattenedArray">The input flattened array</param>
    /// <param name="faDownOffset">The number of elements below the region of a column to be copied from the flattened array.</param>
    /// <param name="faUpOffset">The number of elements above the region of a column to be copied from the flattened array.</param>
    /// <param name="faLeftOffset">The number of elements left of the region of a row to be copied from the flattened array.</param>
    /// <param name="xLength">The number of elements in the x direction that are to be copied.</param>
    /// <param name="yLength">The number of elements in the y direction that are to be copied.</param>
    template<typename T>
    void UnflattenArray(T** pointerArray, int paDownOffset, int paLeftOffset, T* flattenedArray, int faDownOffset, int faUpOffset, int faLeftOffset, int xLength, int yLength);

    /// <summary>
    /// Flattens the 2D array specified in <paramref name="pointerArray" />, storing the result in <paramref name="flattenedArray" />.
    /// </summary>
    /// <typeparam name="T">The type of the elements in the array.</typeparam>
    /// <param name="pointerArray">The input 2D array.</param>
    /// <param name="paDownOffset">The number of elements below the region of a column to be copied from the pointer array.</param>
    /// <param name="paLeftOffset">The number of elements left of the region of a row to be copied from the pointer array.</param>
    /// <param name="flattenedArray">The output flattened array</param>
    /// <param name="faDownOffset">The number of elements below the region of a column to be copied into the flattened array.</param>
    /// <param name="faUpOffset">The number of elements above the region of a column to be copied into the flattened array.</param>
    /// <param name="faLeftOffset">The number of elements left of the region of a row to be copied into the flattened array.</param>
    /// <param name="xLength">The number of elements in the x direction that are to be copied.</param>
    /// <param name="yLength">The number of elements in the y direction that are to be copied.</param>
    template<typename T>
    void FlattenArray(T** pointerArray, int paDownOffset, int paLeftOffset, T* flattenedArray, int faDownOffset, int faUpOffset, int faLeftOffset, int xLength, int yLength);

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

    virtual REAL* GetHorizontalVelocity() const = 0;

    virtual REAL* GetVerticalVelocity() const = 0;

    virtual REAL* GetPressure() const = 0;

    virtual REAL* GetStreamFunction() const = 0;

    virtual bool** GetObstacles() const = 0;

    /// <summary>
    /// Embeds obstacles into the simulation domain. Assumes obstacles have already been set
    /// </summary>
    virtual void ProcessObstacles() = 0;

    /// <summary>
    /// Performs setup for executing timesteps. This function must be called once before the first call to <c>Timestep</c>, and after any changes to <c>parameters</c>.
    /// </summary>
    virtual void PerformSetup() = 0;

    /// <summary>
    /// Computes one timestep, solving each of the fields.
    /// </summary>
    /// <param name="simulationTime">The time that the simulation has been running, to be updated with the new time after the timestep has finished.</param>
    virtual void Timestep(REAL& simulationTime) = 0;
};
#endif // !SOLVER_H