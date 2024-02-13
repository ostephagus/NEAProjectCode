#ifndef DRAG_CALCULATOR_H
#define DRAG_CALCULATOR_H

#include "Definitions.cuh"


class DragCalculator
{
private:
    DragCoordinate* viscosityCoordinates;
    int viscosityCoordinatesLength;

    DragCoordinate* pressureCoordinates;
    int pressureCoordinatesLength;

    REAL projectionArea;

    int threadsPerBlock;

    REAL ComputeObstacleDrag(cudaStream_t streamToUse, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> pressure, int iMax, int jMax, REAL delX, REAL delY, REAL viscosity);

    void FindPressureCoordinates(BYTE** flags, int iMax, int jMax, REAL delX, REAL delY);

    void FindViscosityCoordinates(BYTE** flags, int iMax, int jMax, REAL delX, REAL delY);
public:
    DragCalculator();
    DragCalculator(int threadsPerBlock);

    /// <summary>
    /// Performs internal processing when obstacles are changed.
    /// </summary>
    void ProcessObstacles(BYTE** flags, int iMax, int jMax, REAL delX, REAL delY);

    /// <summary>
    /// Gets the drag coefficient for the obstacle given the fluid flow.
    /// </summary>
    /// <returns>The drag coefficient for the obstacle.</returns>
    REAL GetDragCoefficient(cudaStream_t streamToUse, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> pressure, int iMax, int jMax, REAL delX, REAL delY, REAL viscosity, REAL density, REAL inflowVelocity);
};
#endif // !DRAG_CALCULATOR_H
