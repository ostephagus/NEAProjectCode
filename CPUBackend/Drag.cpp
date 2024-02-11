#include "Drag.h"
#include "Math.h"
#include <bitset>
#include <cstdio> // TESTING

constexpr REAL DIAGONAL_CELL_DISTANCE = (REAL)1.41421356237; // Sqrt 2

REAL Magnitude(REAL x, REAL y) {
    return sqrtf(x * x + y * y);
}

REAL Dot(DoubleReal left, DoubleReal right) {
    return left.x * right.x + left.y * right.y;
}

DoubleReal GetUnitVector(DoubleReal vector) {
    REAL magnitude = Magnitude(vector.x, vector.y);
    return DoubleReal(vector.x / magnitude, vector.y / magnitude);
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
REAL ComputeWallShear(DoubleReal& shearUnitVector, DoubleField velocities, DoubleReal unitNormal, int i, int j, DoubleReal stepSizes, REAL viscosity) {
    int iExtended, jExtended;
    REAL distance;
    if (unitNormal.x == 0 || unitNormal.y == 0) { // Parallel to an axis
        if (unitNormal.x == 0) {
            distance = stepSizes.x;
        }
        else {
            distance = stepSizes.y;
        }
        iExtended = i + (int)unitNormal.x;
        jExtended = j + (int)unitNormal.y;
    }
    else { // 45 degrees to an axis.
        distance = Magnitude(unitNormal.x * stepSizes.x, unitNormal.y * stepSizes.y) * DIAGONAL_CELL_DISTANCE;
        iExtended = i + (int)roundf(unitNormal.x * DIAGONAL_CELL_DISTANCE) * 2;
        jExtended = j + (int)roundf(unitNormal.y * DIAGONAL_CELL_DISTANCE) * 2;
    }
    shearUnitVector = DoubleReal(velocities.x[iExtended][jExtended], velocities.y[iExtended][jExtended]);
    return viscosity * PVPd(velocities, distance, i, j, iExtended, jExtended);
}

/// <summary>
/// Computes the viscous drag on the obstacle.
/// </summary>
/// <returns>The magnitude of the viscous drag on the obstacle.</returns>
REAL ComputeViscousDrag(DoubleField velocities, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int iMax, int jMax, DoubleReal stepSizes, DoubleReal fluidVector, REAL viscosity) { // Calculates the cells that make up the surface and calls ComputeWallShear on each.
    REAL totalViscousDrag = 0;
    for (int coordinateNum = 0; coordinateNum < coordinatesLength; coordinateNum++) {
        std::pair<int, int> coordinate = coordinates[coordinateNum];
        BYTE flag = flags[coordinate.first][coordinate.second];

        BYTE northBit = (flag & NORTH) >> NORTHSHIFT;
        BYTE eastBit = (flag & EAST) >> EASTSHIFT;
        BYTE southBit = (flag & SOUTH) >> SOUTHSHIFT;
        BYTE westBit = (flag & WEST) >> WESTSHIFT;
        int numEdges = (int)std::bitset<8>(flag).count();

        int i = coordinate.first - westBit;
        int j = coordinate.second - southBit;
        DoubleReal unitNormal = DoubleReal((REAL)(eastBit - westBit), (REAL)(northBit - southBit));
        REAL stepSize;
        if (numEdges == 2) {
            unitNormal.x /= DIAGONAL_CELL_DISTANCE;
            unitNormal.y /= DIAGONAL_CELL_DISTANCE;
            stepSize = (stepSizes.x + stepSizes.y) / 2;
        }
        else if ((eastBit | westBit) == 1) {
            stepSize = stepSizes.x;
        }
        else {
            stepSize = stepSizes.y;
            
        }
        DoubleReal shearDirection;
        REAL wallShear = ComputeWallShear(shearDirection, velocities, unitNormal, i, j, stepSizes, viscosity);

        totalViscousDrag += wallShear * -Dot(fluidVector, shearDirection) * stepSize;
    }

    return totalViscousDrag;
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
REAL ComputePressureDrag(REAL** pressure, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int iMax, int jMax, DoubleReal stepSizes, DoubleReal fluidVector) {
    REAL totalPresureDrag = 0;
    REAL baselinePressure = ComputeBaselinePressure(pressure, iMax, jMax);
    for (int coordinateNum = 0; coordinateNum < coordinatesLength; coordinateNum++) {
        std::pair<int, int> coordinate = coordinates[coordinateNum];
        BYTE flag = flags[coordinate.first][coordinate.second];

        BYTE northBit = (flag & NORTH) >> NORTHSHIFT;
        BYTE eastBit  = (flag & EAST)  >> EASTSHIFT;
        BYTE southBit = (flag & SOUTH) >> SOUTHSHIFT;
        BYTE westBit  = (flag & WEST)  >> WESTSHIFT;
        int numEdges = (int)std::bitset<8>(flag).count();

        if (numEdges == 2) { // Corner cell - compute the pressure integrand for the 3 fluid cells around it
            int xDirection = eastBit - westBit;
            int yDirection = northBit - southBit;
            DoubleReal unitNormal;

            // Cell 1: east / west
            unitNormal = DoubleReal((REAL)xDirection, 0);
            totalPresureDrag += PressureIntegrand(pressure[coordinate.first + xDirection][coordinate.second], baselinePressure, unitNormal, fluidVector) * stepSizes.x;

            // Cell 2: north / south
            unitNormal = DoubleReal(0, (REAL)yDirection);
            totalPresureDrag += PressureIntegrand(pressure[coordinate.first][coordinate.second + yDirection], baselinePressure, unitNormal, fluidVector) * stepSizes.y;

            // Cell 3: diagonal
            unitNormal = DoubleReal(xDirection / DIAGONAL_CELL_DISTANCE, yDirection / DIAGONAL_CELL_DISTANCE);
            REAL stepSize = (stepSizes.x + stepSizes.y) / 2;
            totalPresureDrag += PressureIntegrand(pressure[coordinate.first + xDirection][coordinate.second + yDirection], baselinePressure, unitNormal, fluidVector) * stepSize;
        }
        else if (numEdges == 1) { // Edge cell - compute the pressure integrand for the fluid cell next to it
            int i = coordinate.first + eastBit - westBit;
            int j = coordinate.second + northBit - southBit;
            DoubleReal unitNormal = DoubleReal((REAL)(eastBit - westBit), (REAL)(northBit - southBit));
            REAL stepSize;
            if ((eastBit | westBit) == 1) {
                stepSize = stepSizes.x;
            }
            else {
                stepSize = stepSizes.y;
            }

            totalPresureDrag += PressureIntegrand(pressure[i][j], baselinePressure, unitNormal, fluidVector) * stepSize;
        }
        
    }

    return totalPresureDrag;
}

REAL ComputeObstacleDrag(DoubleField velocities, REAL** pressure, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int iMax, int jMax, DoubleReal stepSizes, REAL viscosity)
{
    DoubleReal fluidVector = DoubleReal(-1, 0);

    REAL viscousDrag = ComputeViscousDrag(velocities, flags, coordinates, coordinatesLength, iMax, jMax, stepSizes, fluidVector, viscosity);
    REAL pressureDrag = ComputePressureDrag(pressure, flags, coordinates, coordinatesLength, iMax, jMax, stepSizes, fluidVector);
    printf("Viscous drag: %f, pressure drag: %f.\n", viscousDrag, pressureDrag);
    return viscousDrag + pressureDrag;
}

/// <summary>
/// Finds the area of the object projected onto the y axis by finding the highest and lowest y coordinates of the obstacle.
/// </summary>
/// <returns>The area projected onto the y axis.</returns>
REAL ComputeProjectionArea(std::pair<int, int>* coordinates, int coordinatesLength, REAL delY) {
    int lowestY = coordinates[0].second;
    int highestY = coordinates[0].second;
    for (int i = 0; i < coordinatesLength; i++) {
        int yCoord = coordinates[i].second;
        if (yCoord > highestY) {
            highestY = yCoord;
        }
        if (yCoord < lowestY) {
            lowestY = yCoord;
        }
    }

    return (highestY - lowestY) * delY;
}

REAL ComputeDragCoefficient(DoubleField velocities, REAL** pressure, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int iMax, int jMax, DoubleReal stepSizes, REAL viscosity, REAL density, REAL inflowVelocity)
{
    REAL dragForce = ComputeObstacleDrag(velocities, pressure, flags, coordinates, coordinatesLength, iMax, jMax, stepSizes, viscosity);
    REAL projectedArea = ComputeProjectionArea(coordinates, coordinatesLength, stepSizes.y);
    return (2 * dragForce) / (density * inflowVelocity * inflowVelocity * projectedArea);
}

