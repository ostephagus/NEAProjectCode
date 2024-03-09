#include "Drag.cuh"
#include <vector>
#include <bitset>

__device__ REAL Magnitude(REAL x, REAL y) {
    return sqrtf(x * x + y * y);
}

__device__ REAL Dot(REAL2 left, REAL2 right) {
    return left.x * right.x + left.y * right.y;
}

__device__ REAL2 GetUnitVector(REAL2 vector) {
    REAL magnitude = Magnitude(vector.x, vector.y);
    return make_REAL2(vector.x / magnitude, vector.y / magnitude);
}

__device__ REAL PVPd(PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, REAL distance, int iStart, int jStart, int iExtended, int jExtended) { // As with DiscreteDerivatives, read this as "Partial V over Partial d" - Partial derivative of velocity wrt distance from a point.
    REAL extendedVelocityMagnitude = Magnitude(F_PITCHACCESS(hVel.ptr, hVel.pitch, iExtended, jExtended), F_PITCHACCESS(vVel.ptr, vVel.pitch, iExtended, jExtended));
    REAL surfaceVelocityMagnitude = Magnitude(F_PITCHACCESS(hVel.ptr, hVel.pitch, iStart, jStart), F_PITCHACCESS(vVel.ptr, vVel.pitch, iStart, jStart));
    return (extendedVelocityMagnitude - surfaceVelocityMagnitude) / distance;
}

__device__ REAL ComputeWallShear(REAL2* shearUnitVector, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, REAL2 unitNormal, int i, int j, REAL delX, REAL delY, REAL viscosity) {
    int iExtended, jExtended;
    REAL distance;
    if (unitNormal.x == 0 || unitNormal.y == 0) { // Parallel to an axis
        if (unitNormal.x == 0) {
            distance = delX;
        }
        else {
            distance = delY;
        }
        iExtended = i + (int)unitNormal.x;
        jExtended = j + (int)unitNormal.y;
    }
    else { // 45 degrees to an axis.
        distance = Magnitude(unitNormal.x * delX, unitNormal.y * delY) * DIAGONAL_CELL_DISTANCE;
        iExtended = i + (int)roundf(unitNormal.x * DIAGONAL_CELL_DISTANCE) * 2;
        jExtended = j + (int)roundf(unitNormal.y * DIAGONAL_CELL_DISTANCE) * 2;
    }
    *shearUnitVector = GetUnitVector(make_REAL2(F_PITCHACCESS(hVel.ptr, hVel.pitch, iExtended, jExtended), F_PITCHACCESS(vVel.ptr, vVel.pitch, iExtended, jExtended)));
    REAL wallShear = viscosity * PVPd(hVel, vVel, distance, i, j, iExtended, jExtended);
    return wallShear;
}

__global__ void ComputeViscousDrag(REAL* integrandArray, DragCoordinate* viscosityCoordinates, int viscosityCoordinatesLength, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, int iMax, int jMax, REAL2 fluidVector, REAL delX, REAL delY, REAL viscosity) {
    int coordinateNum = blockIdx.x * blockDim.x + threadIdx.x;
    if (coordinateNum >= viscosityCoordinatesLength) return;
    DragCoordinate dragCoordinate = viscosityCoordinates[coordinateNum];
    REAL2 shearDirection;
    REAL wallShear = ComputeWallShear(&shearDirection, hVel, vVel, dragCoordinate.unitNormal, dragCoordinate.coordinate.x, dragCoordinate.coordinate.y, delX, delY, viscosity);
    integrandArray[coordinateNum] = abs(wallShear * Dot(fluidVector, shearDirection)) * dragCoordinate.stepSize;
}

__global__ void ComputeBaselinePressure(REAL* baselinePressure, PointerWithPitch<REAL> pressure, int iMax, int jMax) {
    *baselinePressure = (F_PITCHACCESS(pressure.ptr, pressure.pitch, 1, 1) + F_PITCHACCESS(pressure.ptr, pressure.pitch, 1, jMax) + F_PITCHACCESS(pressure.ptr, pressure.pitch, iMax, 1) + F_PITCHACCESS(pressure.ptr, pressure.pitch, iMax, jMax)) / 4;
}

__device__ REAL PressureIntegrand(REAL pressure, REAL baselinePressure, REAL2 unitNormal, REAL2 fluidVector) {
    return (pressure - baselinePressure) * Dot(unitNormal, fluidVector);
}

__global__ void ComputePressureDrag(REAL* integrandArray, DragCoordinate* pressureCoordinates, int pressureCoordinatesLength, PointerWithPitch<REAL> pressure, int iMax, int jMax, REAL delX, REAL delY, REAL2 fluidVector, REAL* baselinePressure) {
    int coordinateNum = blockIdx.x * blockDim.x + threadIdx.x;
    if (coordinateNum >= pressureCoordinatesLength) return;
    DragCoordinate dragCoordinate = pressureCoordinates[coordinateNum];
    integrandArray[coordinateNum] = PressureIntegrand(F_PITCHACCESS(pressure.ptr, pressure.pitch, dragCoordinate.coordinate.x, dragCoordinate.coordinate.y), *baselinePressure, dragCoordinate.unitNormal, fluidVector) * dragCoordinate.stepSize;
}