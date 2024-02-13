#include "DragCalculator.cuh"
#include <vector>
#include <bitset>
#include "Drag.cuh"
#include "ReductionKernels.cuh"

REAL DragCalculator::ComputeObstacleDrag(cudaStream_t streamToUse, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> pressure, int iMax, int jMax, REAL delX, REAL delY, REAL viscosity)
{
    REAL2 fluidVector = make_REAL2(-1, 0);

    REAL* viscosityIntegrandArray;
    cudaMalloc(&viscosityIntegrandArray, viscosityCoordinatesLength * sizeof(REAL));
    REAL* pressureIntegrandArray;
    cudaMalloc(&pressureIntegrandArray, pressureCoordinatesLength * sizeof(REAL));

    // Viscous drag calculation
    ComputeViscousDrag KERNEL_ARGS(1, threadsPerBlock, 0, streamToUse) (viscosityIntegrandArray, viscosityCoordinates, viscosityCoordinatesLength, hVel, vVel, iMax, jMax, fluidVector, delX, delY, viscosity);

    REAL* viscosityIntegral;
    cudaMalloc(&viscosityIntegral, sizeof(REAL));
    ComputeFinalSum KERNEL_ARGS(1, threadsPerBlock, threadsPerBlock * sizeof(REAL), streamToUse) (viscosityIntegral, viscosityIntegrandArray, viscosityCoordinatesLength);
    REAL viscousDrag;
    cudaMemcpy(&viscousDrag, viscosityIntegral, sizeof(REAL), cudaMemcpyDeviceToHost);

    REAL* baselinePressure;
    cudaMalloc(&baselinePressure, sizeof(REAL));
    ComputeBaselinePressure KERNEL_ARGS(1, 1, 0, streamToUse) (baselinePressure, pressure, iMax, jMax);

    // Pressure drag calculation
    ComputePressureDrag KERNEL_ARGS(1, threadsPerBlock, 0, streamToUse) (pressureIntegrandArray, pressureCoordinates, pressureCoordinatesLength, pressure, iMax, jMax, delX, delY, fluidVector, baselinePressure);

    REAL* pressureIntegral;
    cudaMalloc(&pressureIntegral, sizeof(REAL));
    ComputeFinalSum KERNEL_ARGS(1, threadsPerBlock, threadsPerBlock * sizeof(REAL), streamToUse) (pressureIntegral, pressureIntegrandArray, pressureCoordinatesLength);
    REAL pressureDrag;
    cudaMemcpy(&pressureDrag, pressureIntegral, sizeof(REAL), cudaMemcpyDeviceToHost);

    cudaFree(viscosityIntegrandArray);
    cudaFree(pressureIntegrandArray);
    cudaFree(baselinePressure);
    cudaFree(pressureIntegral);
    return viscousDrag * VISCOSITY_CONVERSION + pressureDrag * PRESSURE_CONVERSION;
}

void DragCalculator::FindPressureCoordinates(BYTE** flags, int iMax, int jMax, REAL delX, REAL delY) {
    std::vector<DragCoordinate> coordinatesVec;
    for (int i = 1; i <= iMax; i++) {
        for (int j = 1; j <= jMax; j++) {
            BYTE flag = flags[i][j];
            if (flag >= 0b00000001 && flag <= 0b00001111) { // This defines boundary cells - all cells without the self bit set except when no bits are set.
                BYTE northBit = (flag & NORTH) >> NORTHSHIFT;
                BYTE eastBit = (flag & EAST) >> EASTSHIFT;
                BYTE southBit = (flag & SOUTH) >> SOUTHSHIFT;
                BYTE westBit = (flag & WEST) >> WESTSHIFT;
                int numEdges = (int)std::bitset<8>(flag).count();
                if (numEdges == 1) {
                    DragCoordinate dragCoordinate = DragCoordinate();
                    dragCoordinate.coordinate = make_uint2(i + eastBit - westBit, j + northBit - southBit);
                    dragCoordinate.unitNormal = make_REAL2((REAL)(eastBit - westBit), (REAL)(northBit - southBit));
                    if ((eastBit | westBit) == 1) {
                        dragCoordinate.stepSize = delX;
                    }
                    else {
                        dragCoordinate.stepSize = delY;
                    }
                    coordinatesVec.push_back(dragCoordinate);
                }
                else if (numEdges == 2) {
                    int xDirection = eastBit - westBit;
                    int yDirection = northBit - southBit;

                    // Cell 1: east / west
                    DragCoordinate horizontalCoordinate = DragCoordinate();
                    horizontalCoordinate.coordinate = make_uint2(i + xDirection, j);
                    horizontalCoordinate.unitNormal = make_REAL2((REAL)xDirection, 0);
                    horizontalCoordinate.stepSize = delX;
                    coordinatesVec.push_back(horizontalCoordinate);

                    // Cell 2: north / south
                    DragCoordinate verticalCoordinate = DragCoordinate();
                    verticalCoordinate.coordinate = make_uint2(i, j + yDirection);
                    verticalCoordinate.unitNormal = make_REAL2(0, (REAL)yDirection);
                    verticalCoordinate.stepSize = delY;
                    coordinatesVec.push_back(verticalCoordinate);

                    // Cell 3: diagonal
                    DragCoordinate diagonalCoordinate = DragCoordinate();
                    diagonalCoordinate.unitNormal = make_REAL2(xDirection / DIAGONAL_CELL_DISTANCE, yDirection / DIAGONAL_CELL_DISTANCE);
                    diagonalCoordinate.stepSize = (delX + delY) / 2;
                    diagonalCoordinate.coordinate = make_uint2(i + xDirection, j + yDirection);
                    coordinatesVec.push_back(diagonalCoordinate);
                }
            }
        }
    }

    pressureCoordinatesLength = (int)coordinatesVec.size();
    cudaMalloc(&pressureCoordinates, pressureCoordinatesLength * sizeof(DragCoordinate));
    cudaMemcpy(pressureCoordinates, coordinatesVec.data(), pressureCoordinatesLength * sizeof(DragCoordinate), cudaMemcpyHostToDevice);
}

void DragCalculator::FindViscosityCoordinates(BYTE** flags, int iMax, int jMax, REAL delX, REAL delY) {
    std::vector<DragCoordinate> coordinatesVec;
    int lowestY = -1, highestY = -1;
    for (int i = 1; i <= iMax; i++) {
        for (int j = 1; j <= jMax; j++) {
            BYTE flag = flags[i][j];
            if (flag >= 0b00000001 && flag <= 0b00001111) { // This defines boundary cells - all cells without the self bit set except when no bits are set.
                BYTE northBit = (flag & NORTH) >> NORTHSHIFT;
                BYTE eastBit = (flag & EAST) >> EASTSHIFT;
                BYTE southBit = (flag & SOUTH) >> SOUTHSHIFT;
                BYTE westBit = (flag & WEST) >> WESTSHIFT;
                int numEdges = (int)std::bitset<8>(flag).count();
                DragCoordinate dragCoordinate = DragCoordinate();
                dragCoordinate.coordinate = make_uint2(i - westBit, j - southBit);
                if (numEdges == 1) {
                    dragCoordinate.unitNormal = make_REAL2((REAL)(eastBit - westBit), (REAL)(northBit - southBit));
                    if ((eastBit | westBit) == 1) {
                        dragCoordinate.stepSize = delX;
                    }
                    else {
                        dragCoordinate.stepSize = delY;
                    }
                }
                else if (numEdges == 2) {
                    dragCoordinate.unitNormal = make_REAL2((REAL)(eastBit - westBit) / DIAGONAL_CELL_DISTANCE, (REAL)(northBit - southBit) / DIAGONAL_CELL_DISTANCE);
                    dragCoordinate.stepSize = (delX + delY) / 2;
                }
                coordinatesVec.push_back(dragCoordinate);

                if (lowestY == -1 || highestY == -1) {
                    highestY = j;
                    lowestY = j;
                }
                else {
                    if (j > highestY) {
                        highestY = j;
                    }
                    if (j < lowestY) {
                        lowestY = j;
                    }
                }
            }
        }
    }

    viscosityCoordinatesLength = (int)coordinatesVec.size();
    cudaMalloc(&viscosityCoordinates, viscosityCoordinatesLength * sizeof(DragCoordinate));
    cudaMemcpy(viscosityCoordinates, coordinatesVec.data(), viscosityCoordinatesLength * sizeof(DragCoordinate), cudaMemcpyHostToDevice);
    projectionArea = (highestY - lowestY) * delY;
}

DragCalculator::DragCalculator() : viscosityCoordinates(nullptr), viscosityCoordinatesLength(0), pressureCoordinates(nullptr), pressureCoordinatesLength(0), projectionArea(0), threadsPerBlock(1024) {}

DragCalculator::DragCalculator(int threadsPerBlock) : viscosityCoordinates(nullptr), viscosityCoordinatesLength(0), pressureCoordinates(nullptr), pressureCoordinatesLength(0), projectionArea(0), threadsPerBlock(threadsPerBlock) {}

void DragCalculator::ProcessObstacles(BYTE** flags, int iMax, int jMax, REAL delX, REAL delY) {
    FindPressureCoordinates(flags, iMax, jMax, delX, delY);
    FindViscosityCoordinates(flags, iMax, jMax, delX, delY);
}

REAL DragCalculator::GetDragCoefficient(cudaStream_t streamToUse, PointerWithPitch<REAL> hVel, PointerWithPitch<REAL> vVel, PointerWithPitch<REAL> pressure, int iMax, int jMax, REAL delX, REAL delY, REAL viscosity, REAL density, REAL inflowVelocity) {
    REAL dragForce = ComputeObstacleDrag(streamToUse, hVel, vVel, pressure, iMax, jMax, delX, delY, viscosity);
    return (2 * dragForce) / (density * inflowVelocity * inflowVelocity * projectionArea);
}