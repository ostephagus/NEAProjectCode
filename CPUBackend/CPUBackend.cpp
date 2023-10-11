#include <iostream>
#include <string>
#include "Boundary.h"
#include "Computation.h"
#include "DiscreteDerivatives.h"
#include "Init.h"
#include <bitset>

void PrintField(REAL** field, int iMax, int jMax, std::string name) {
    std::cout.precision(3);
    std::cout << name << ": " << std::endl;
    for (int i = iMax-1; i >= 0; i--) {
        for (int j = 0; j < jMax; j++) {

            std::cout << field[j][i] << ' '; //i and j are swapped here because we print first in the horizontal direction (i or u) then in the vertical (j or v)
        }
        std::cout << std::endl;
    }
}
void PrintField(BYTE** flags, int iMax, int jMax, std::string name) {
    std::cout << name << ": " << std::endl;
    for (int i = iMax - 1; i >= 0; i--) {
        for (int j = 0; j < jMax; j++) {
            std::bitset<8> element(flags[i][j]);
            std::cout << element << ' '; //i and j are swapped here because we print first in the horizontal direction (i or u) then in the vertical (j or v)
        }
        std::cout << std::endl;
    }
}

void StepTestSquare() {
    int iMax = 300, jMax = 300;

    DoubleField velocities;
    velocities.x = MatrixMAlloc(iMax + 2, jMax + 2);
    velocities.y = MatrixMAlloc(iMax + 2, jMax + 2);

    REAL** pressure = MatrixMAlloc(iMax + 2, jMax + 2);
    REAL** RHS = MatrixMAlloc(iMax + 2, jMax + 2);

    DoubleField FG;
    FG.x = MatrixMAlloc(iMax + 2, jMax + 2);
    FG.y = MatrixMAlloc(iMax + 2, jMax + 2);

    BYTE** flags = FlagMatrixMAlloc(iMax + 2, jMax + 2);
    bool** obstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);
    SetFlags(obstacles, flags, iMax + 2, jMax + 2);
    std::pair<std::pair<int, int>*, int> coordinatesWithLength = FindBoundaryCells(flags, iMax, jMax);
    std::pair<int, int>* coordinates = coordinatesWithLength.first;
    int coordinatesLength = coordinatesWithLength.second;

    const REAL width = 0.03;
    const REAL height = 0.03;
    const REAL timeStepSafetyFactor = 0.8;
    const REAL relaxationParameter = 1.7;
    const REAL pressureResidualTolerance = 1; //Needs experimentation
    const int pressureMaxIterations = 1000; //Needs experimentation
    const REAL reynoldsNo = 2000;
    const REAL inflowVelocity = 5;
    REAL pressureResidualNorm = 0;

    DoubleReal bodyForces;
    bodyForces.x = 0;
    bodyForces.y = 0;

    REAL timestep;
    DoubleReal stepSizes;
    stepSizes.x = width / iMax;
    stepSizes.y = height / jMax;

    for (int i = 0; i <= iMax+1; i++) {
        for (int j = 0; j <= jMax+1; j++) {
            pressure[i][j] = 1000;
        }
    }
    //PrintField(pressure, iMax+2, jMax+2, "Pressure");
    for (int i = 1; i <= iMax; i++) {
        for (int j = 1; j <= jMax; j++) {
            velocities.x[i][j] = 4;
            velocities.y[i][j] = 0;
        }
    }
    //PrintField(velocities.x, iMax + 2, jMax + 2, "Horizontal velocities");
    //PrintField(velocities.y, iMax + 2, jMax + 2, "Vertical velocities");

    while (true) {//BREAKPOINT REQUIRED
        ComputeTimestep(timestep, iMax, jMax, stepSizes, velocities, reynoldsNo, timeStepSafetyFactor);
        std::cout << timestep << std::endl;
        SetBoundaryConditions(velocities, iMax, jMax, inflowVelocity);
        //PrintField(velocities.x, iMax + 2, jMax + 2, "Horizontal velocities");
        //PrintField(velocities.y, iMax + 2, jMax + 2, "Vertical velocities");
        REAL gamma = ComputeGamma(velocities, iMax, jMax, timestep, stepSizes);
        ComputeFG(velocities, FG, iMax, jMax, timestep, stepSizes, bodyForces, gamma, reynoldsNo); 
        //PrintField(FG.x, iMax + 2, jMax + 2, "F");
        //PrintField(FG.y, iMax + 2, jMax + 2, "G");
        ComputeRHS(FG, RHS, iMax, jMax, timestep, stepSizes); //Finished debugging up to here
        //PrintField(RHS, iMax + 2, jMax + 2, "Pressure RHS");
        Poisson(pressure, RHS, flags, coordinates, coordinatesLength, iMax, jMax, stepSizes, pressureResidualTolerance, pressureMaxIterations, relaxationParameter, pressureResidualNorm);
        //PrintField(pressure, iMax + 2, jMax + 2, "Pressure");
        std::cout << pressureResidualNorm << std::endl;
        ComputeVelocities(velocities, FG, pressure, iMax, jMax, timestep, stepSizes);
    }


    FreeMatrix(velocities.x, iMax + 2);
    FreeMatrix(velocities.y, iMax + 2);
    FreeMatrix(pressure, iMax + 2);
    FreeMatrix(RHS, iMax + 2);
    FreeMatrix(FG.x, iMax + 2);
    FreeMatrix(FG.y, iMax + 2);
}

void TestCopyBoundaryPressures() {
    int iMax = 5, jMax = 5;
    bool** obstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);
    obstacles[1][1] = 1;
    obstacles[1][2] = 1;
    obstacles[1][3] = 1;
    obstacles[1][4] = 1;
    obstacles[1][5] = 1;
    obstacles[2][1] = 1;
    obstacles[2][5] = 1;
    obstacles[3][1] = 1;
    obstacles[3][5] = 1;
    obstacles[4][1] = 1;
    obstacles[4][5] = 1;
    obstacles[5][1] = 1;
    obstacles[5][2] = 1;
    obstacles[5][3] = 1;
    obstacles[5][4] = 1;
    obstacles[5][5] = 1;
    BYTE** flags = FlagMatrixMAlloc(iMax + 2, jMax + 2);
    SetFlags(obstacles, flags, iMax + 2, jMax + 2);
    PrintField(flags, iMax + 2, jMax + 2, "Flags");
    REAL** pressure = MatrixMAlloc(iMax + 2, jMax + 2);
    REAL pressureCount = 0;
    for (int i = 1; i <= iMax; i++) {
        for (int j = 1; j <= jMax; j++) {
            if (obstacles[i][j]) {
                pressure[i][j] = pressureCount;
                pressureCount++;
            }
        }
    }
    PrintField(pressure, iMax + 2, jMax + 2, "Pressure");
    std::pair<std::pair<int, int>*, int> coordArrayWithLength = FindBoundaryCells(flags, iMax, jMax);
    CopyBoundaryPressures(pressure, coordArrayWithLength.first, coordArrayWithLength.second, flags, iMax, jMax);
    PrintField(pressure, iMax + 2, jMax + 2, "Pressure (with copies)");
}

int main() {
    TestCopyBoundaryPressures();
    return 0;
}