#ifndef CPUSOLVER_H
#define CPUSOLVER_H

#include "Solver.h"
class CPUSolver :
    public Solver
{
private:
    DoubleField velocities;
    REAL** pressure;
    REAL** RHS;
    REAL** streamFunction;
    DoubleField FG;

    REAL* flattenedHVel;
    REAL* flattenedVVel;
    REAL* flattenedPressure;
    REAL* flattenedStream;

    DoubleReal stepSizes;

    bool** obstacles;
    BYTE** flags;
    std::pair<int, int>* coordinates;
    int coordinatesLength;
    int numFluidCells;
public:
    CPUSolver(SimulationParameters parameters, int iMax, int jMax);

    ~CPUSolver();

    bool** GetObstacles() const;

    REAL* GetHorizontalVelocity() const;

    REAL* GetVerticalVelocity() const;

    REAL* GetPressure() const;

    REAL* GetStreamFunction() const;

    REAL GetDragCoefficient();

    void ProcessObstacles();

    void PerformSetup();

    void Timestep(REAL& simulationTime); // Implementing abstract inherited method
};

#endif // !CPUSOLVER_H

