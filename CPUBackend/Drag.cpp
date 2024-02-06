#include "Drag.h"
#include "Math.h"
#include <bitset>

constexpr REAL DIAGONAL_CELL_DISTANCE = (REAL)1.41421356237; // Sqrt 2

REAL Magnitude(REAL x, REAL y) {
    return sqrtf(x * x + y * y);
}

REAL Dot(DoubleReal left, DoubleReal right) {
    return left.x * right.x + left.y * right.y;
}

/// <summary>
/// Calculates the partial derivative of the velocity field with respect to the distance from a point.
/// </summary>
/// <param name="unitVector">The direction vector to extend out from the point.</param>
/// <param name="distance">The separation of the points where velocity is taken</param>
/// <returns>The partial derivative of velocity with respect to distance from a point.</returns>
REAL PVPd(DoubleField velocities, REAL distance, int iStart, int jStart, int iExtended, int jExtended) { // As with DiscreteDerivatives, read this as "Partial V over Partial d" - Partial derivative of velocity wrt distance from a point.
    REAL extendedVelocityMagnitude = Magnitude(velocities.x[iExtended][jExtended], velocities.y[iExtended][jExtended]);
    REAL surfaceVelocityMagnitude = Magnitude(velocities.x[iStart][jStart], velocities.y[iStart][jStart]);
    return (extendedVelocityMagnitude - surfaceVelocityMagnitude) / distance;
}

/// <summary>
/// Computes wall shear stress for a single boundary cell.
/// </summary>
/// <param name="unitNormal">The unit vector perpendicular to the direction of the surface at the cell.</param>
/// <returns>The magnitude of the wall shear stress for one boundary cell.</returns>
REAL ComputeWallShear(DoubleField velocities, DoubleReal unitNormal, int i, int j, DoubleReal stepSizes, REAL viscosity) {
    if (unitNormal.x == 0 || unitNormal.y == 0) { // Parallel to an axis
        REAL relevantStepsize;
        if (unitNormal.x == 0) relevantStepsize = stepSizes.x;
        else relevantStepsize = stepSizes.y;

        return viscosity * PVPd(velocities, relevantStepsize, i, j, i + (int)unitNormal.x, j + (int)unitNormal.y);
    }
    else { // 45 degrees to an axis.
        REAL distance = Magnitude(unitNormal.x * stepSizes.x, unitNormal.y * stepSizes.y) * DIAGONAL_CELL_DISTANCE;
        int iExtended = i + (int)(unitNormal.x * DIAGONAL_CELL_DISTANCE);
        int jExtended = j + (int)(unitNormal.y * DIAGONAL_CELL_DISTANCE);
        return viscosity * PVPd(velocities, distance, i, j, iExtended, jExtended);
    }
}

/// <summary>
/// Computes the viscous drag on the obstacle.
/// </summary>
/// <returns>The magnitude of the viscous drag on the obstacle.</returns>
REAL ComputeViscousDrag(DoubleField velocities, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int iMax, int jMax, DoubleReal stepSizes, REAL viscosity) { // Calculates the cells that make up the surface and calls ComputeViscousDrag on each.
    return 0;
}

/// <summary>
/// Computes a baseline pressure by taking an average of all 4 corners' pressures.
/// </summary>
/// <returns>The average of the pressures in the 4 corners.</returns>
REAL ComputeBaselinePressure(REAL** pressure, int iMax, int jMax) {
    return (pressure[1][1] + pressure[1][jMax] + pressure[iMax][1] + pressure[iMax][jMax]) / 4;
}

REAL PressureIntegrand(REAL pressure, REAL baselinePressure, DoubleReal unitNormal, DoubleReal fluidVector) {
    return (pressure - baselinePressure) * Dot(unitNormal, fluidVector);
}

/// <summary>
/// Computes the pressure drag on the obstacle. Assumes fluid flowing left to right.
/// </summary>
/// <returns>The magnitude of the pressure drag on the obstacle.</returns>
REAL ComputePressureDrag(REAL** pressure, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int iMax, int jMax, DoubleReal stepSizes) {
    REAL totalPresureDrag = 0;
    DoubleReal fluidVector = DoubleReal(-1, 0);
    REAL baselinePressure = ComputeBaselinePressure(pressure, iMax, jMax);
    for (int coordinateNum = 0; coordinateNum < coordinatesLength; coordinateNum++) {
        BYTE flag = flags[coordinates[coordinateNum].first][coordinates[coordinateNum].second];

        BYTE northBit = (flag & NORTH) >> NORTHSHIFT;
        BYTE eastBit  = (flag & EAST)  >> EASTSHIFT;
        BYTE southBit = (flag & SOUTH) >> SOUTHSHIFT;
        BYTE westBit  = (flag & WEST)  >> WESTSHIFT;
        int numEdges = (int)std::bitset<8>(flag).count();

        if (numEdges == 2) { // Corner cell - compute the pressure integrand for the 3 fluid cells around it
            int xDirection = eastBit - westBit;
            int yDirection = northBit - southBit;

        }
        else if (numEdges == 1) { // Edge cell - compute the pressure integrand for the fluid cell next to it
            int i = coordinates[coordinateNum].first + eastBit - westBit;
            int j = coordinates[coordinateNum].second + northBit - southBit;
            DoubleReal unitNormal = DoubleReal((REAL)(eastBit - westBit), (REAL)(northBit - southBit));

            REAL stepSize = stepSizes.x;
            if ((northBit | southBit) == 1) {
                stepSize = stepSizes.y;
            }
            totalPresureDrag += PressureIntegrand(pressure[i][j], baselinePressure, unitNormal, fluidVector) * stepSize;
        }
        
    }

    return totalPresureDrag;
}

REAL ComputeObstacleDrag(DoubleField velocities, REAL** pressure, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int iMax, int jMax, DoubleReal stepSizes, REAL viscosity)
{
    return ComputeViscousDrag(velocities, flags, coordinates, coordinatesLength, iMax, jMax, stepSizes, viscosity) + ComputePressureDrag(pressure, flags, coordinates, coordinatesLength, iMax, jMax, stepSizes);
}
