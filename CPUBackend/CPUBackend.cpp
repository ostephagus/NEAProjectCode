#include <iostream>
#include <string>
#include "Boundary.h"
#include "Computation.h"
#include "DiscreteDerivatives.h"
#include "Init.h"
#include <bitset>
#include <chrono>
#include "PipeManager.h"
#include "PipeConstants.h"
#include "FrontendManager.h"

void PrintField(REAL** field, int xLength, int yLength, std::string name) {
    std::cout.precision(3);
    std::cout << name << ": " << std::endl;
    for (int i = xLength-1; i >= 0; i--) {
        for (int j = 0; j < yLength; j++) {

            std::cout << field[j][i] << ' '; //i and j are swapped here because we print first in the horizontal direction (i or u) then in the vertical (j or v)
        }
        std::cout << std::endl;
    }
}
void PrintField(BYTE** flags, int xLength, int yLength, std::string name) {
    std::cout << name << ":" << std::endl;
    for (int i = xLength - 1; i >= 0; i--) {
        for (int j = 0; j < yLength; j++) {
            std::bitset<8> element(flags[j][i]);
            std::cout << element << ' '; //i and j are swapped here because we print first in the horizontal direction (i or u) then in the vertical (j or v)
        }
        std::cout << std::endl;
    }
}


void SetObstacles(bool** obstacles) { // Input: a 2D array of bools all set to 1
    obstacles[16][25] = 0;
    obstacles[16][24] = 0;

    obstacles[17][23] = 0;
    obstacles[17][24] = 0;
    obstacles[17][25] = 0;
    obstacles[17][26] = 0;

    obstacles[18][22] = 0;
    obstacles[18][23] = 0;
    obstacles[18][24] = 0;
    obstacles[18][25] = 0;
    obstacles[18][26] = 0;
    obstacles[18][27] = 0;

    for (int i = 19; i < 34; i++) { // body of the obstacle
        obstacles[i][21] = 0;
        obstacles[i][22] = 0;
        obstacles[i][23] = 0;
        obstacles[i][24] = 0;
        obstacles[i][25] = 0;
        obstacles[i][26] = 0;
        obstacles[i][27] = 0;
        obstacles[i][28] = 0;
    }

    obstacles[34][22] = 0;
    obstacles[34][23] = 0;
    obstacles[34][24] = 0;
    obstacles[34][25] = 0;
    obstacles[34][26] = 0;
    obstacles[34][27] = 0;

    obstacles[35][23] = 0;
    obstacles[35][24] = 0;
    obstacles[35][25] = 0;
    obstacles[35][26] = 0;

    obstacles[36][25] = 0;
    obstacles[36][24] = 0;
}

void StepTestSquare(int squareLength) {
    int iMax = squareLength, jMax = squareLength;

    DoubleField velocities;
    velocities.x = MatrixMAlloc(iMax + 2, jMax + 2);
    velocities.y = MatrixMAlloc(iMax + 2, jMax + 2);

    REAL** pressure = MatrixMAlloc(iMax + 2, jMax + 2);
    REAL** nextPressure = MatrixMAlloc(iMax + 2, jMax + 2);
    REAL** RHS = MatrixMAlloc(iMax + 2, jMax + 2);

    DoubleField FG;
    FG.x = MatrixMAlloc(iMax + 2, jMax + 2);
    FG.y = MatrixMAlloc(iMax + 2, jMax + 2);

    BYTE** flags = FlagMatrixMAlloc(iMax + 2, jMax + 2);
    bool** obstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);
    for (int i = 1; i <= iMax; i++) { for (int j = 1; j <= jMax; j++) { obstacles[i][j] = 1; } } //Set all the cells to fluid
    //SetObstacles(obstacles);
    SetFlags(obstacles, flags, iMax + 2, jMax + 2);
    //PrintField(flags, iMax + 2, jMax + 2, "flags");

    std::pair<std::pair<int, int>*, int> coordinatesWithLength = FindBoundaryCells(flags, iMax, jMax);
    std::pair<int, int>* coordinates = coordinatesWithLength.first;
    int coordinatesLength = coordinatesWithLength.second;
    
    int numFluidCells = CountFluidCells(flags, iMax, jMax);

    const REAL width = 1;
    const REAL height = 1;
    const REAL timeStepSafetyFactor = 0.8;
    const REAL relaxationParameter = 1.2;
    const REAL pressureResidualTolerance = 1; //Needs experimentation
    const int pressureMinIterations = 10;
    const int pressureMaxIterations = 1000; //Needs experimentation
    const REAL reynoldsNo = 2000;
    const REAL inflowVelocity = 5;
    const REAL surfaceFrictionalPermissibility = 0;
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

    int iteration = 0;
    std::cout << "Enter number of iterations ";
    int iterMax;
    std::cin >> iterMax;
    //int iterMax = 10; //TESTING
    int pressureIterations;
    //auto startTime = std::chrono::high_resolution_clock::now(); //TESTING
    while (iteration < iterMax) {
        ComputeTimestep(timestep, iMax, jMax, stepSizes, velocities, reynoldsNo, timeStepSafetyFactor);
        //std::cout << timestep << std::endl;
        SetBoundaryConditions(velocities, flags, coordinates, coordinatesLength, iMax, jMax, inflowVelocity, surfaceFrictionalPermissibility);
        //PrintField(velocities.x, iMax + 2, jMax + 2, "Horizontal velocities");
        //PrintField(velocities.y, iMax + 2, jMax + 2, "Vertical velocities");
        REAL gamma = ComputeGamma(velocities, iMax, jMax, timestep, stepSizes);
        ComputeFG(velocities, FG, flags, iMax, jMax, timestep, stepSizes, bodyForces, gamma, reynoldsNo); 
        //PrintField(FG.x, iMax + 2, jMax + 2, "F");
        //PrintField(FG.y, iMax + 2, jMax + 2, "G");
        ComputeRHS(FG, RHS, flags, iMax, jMax, timestep, stepSizes);
        //PrintField(RHS, iMax + 2, jMax + 2, "Pressure RHS");
        pressureIterations = Poisson(pressure, nextPressure, RHS, flags, coordinates, coordinatesLength, numFluidCells, iMax, jMax, stepSizes, pressureResidualTolerance, pressureMinIterations, pressureMaxIterations, relaxationParameter, pressureResidualNorm);
        //PrintField(pressure, iMax + 2, jMax + 2, "Pressure");
        //std::cout << pressureResidualNorm << std::endl;
        ComputeVelocities(velocities, FG, pressure, flags, iMax, jMax, timestep, stepSizes);
        std::cout << "Iteration " << iteration << ": velocity before " << velocities.x[14][25] << ", velocity after " << velocities.x[37][25] << ", pressure before " << pressure[14][25] << ", pressure after " << pressure[37][25] << ", residual norm " << pressureResidualNorm << std::endl;
        iteration++;
    }
    //auto endTime = std::chrono::high_resolution_clock::now(); //TESTING

    //std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 1000.0 << " seconds.";

    FreeMatrix(velocities.x, iMax + 2);
    FreeMatrix(velocities.y, iMax + 2);
    FreeMatrix(pressure, iMax + 2);
    FreeMatrix(nextPressure, iMax + 2);
    FreeMatrix(RHS, iMax + 2);
    FreeMatrix(FG.x, iMax + 2);
    FreeMatrix(FG.y, iMax + 2);
}

float TestParameters(REAL parameterValue, int iterations) {
    int iMax = 50, jMax = 50;

    DoubleField velocities;
    velocities.x = MatrixMAlloc(iMax + 2, jMax + 2);
    velocities.y = MatrixMAlloc(iMax + 2, jMax + 2);

    REAL** pressure = MatrixMAlloc(iMax + 2, jMax + 2);
    REAL** nextPressure = MatrixMAlloc(iMax + 2, jMax + 2);
    REAL** RHS = MatrixMAlloc(iMax + 2, jMax + 2);

    DoubleField FG;
    FG.x = MatrixMAlloc(iMax + 2, jMax + 2);
    FG.y = MatrixMAlloc(iMax + 2, jMax + 2);

    BYTE** flags = FlagMatrixMAlloc(iMax + 2, jMax + 2);
    bool** obstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);
    for (int i = 1; i <= iMax; i++) { for (int j = 1; j <= jMax; j++) { obstacles[i][j] = 1; } }
    SetFlags(obstacles, flags, iMax + 2, jMax + 2);

    std::pair<std::pair<int, int>*, int> coordinatesWithLength = FindBoundaryCells(flags, iMax, jMax);
    std::pair<int, int>* coordinates = coordinatesWithLength.first;
    int coordinatesLength = coordinatesWithLength.second;

    int numFluidCells = CountFluidCells(flags, iMax, jMax);

    const REAL width = 1;
    const REAL height = 1;
    const REAL timeStepSafetyFactor = 0.8;
    REAL relaxationParameter = 1.7;
    relaxationParameter = parameterValue;
    const REAL pressureResidualTolerance = 1;
    const int pressureMinIterations = 10;
    const int pressureMaxIterations = 1000;
    const REAL reynoldsNo = 2000;
    const REAL inflowVelocity = 5;
    const REAL surfaceFrictionalPermissibility = 0;
    REAL pressureResidualNorm = 0;

    DoubleReal bodyForces;
    bodyForces.x = 0;
    bodyForces.y = 0;

    REAL timestep;
    DoubleReal stepSizes;
    stepSizes.x = width / iMax;
    stepSizes.y = height / jMax;

    for (int i = 0; i <= iMax + 1; i++) {
        for (int j = 0; j <= jMax + 1; j++) {
            pressure[i][j] = 10;
        }
    }
    for (int i = 1; i <= iMax; i++) {
        for (int j = 1; j <= jMax; j++) {
            velocities.x[i][j] = 4;
            velocities.y[i][j] = 0;
        }
    }
    int iteration = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    while (iteration < iterations) {
        ComputeTimestep(timestep, iMax, jMax, stepSizes, velocities, reynoldsNo, timeStepSafetyFactor);
        SetBoundaryConditions(velocities, flags, coordinates, coordinatesLength, iMax, jMax, inflowVelocity, surfaceFrictionalPermissibility);
        REAL gamma = ComputeGamma(velocities, iMax, jMax, timestep, stepSizes);
        ComputeFG(velocities, FG, flags, iMax, jMax, timestep, stepSizes, bodyForces, gamma, reynoldsNo);
        ComputeRHS(FG, RHS, flags, iMax, jMax, timestep, stepSizes);
        Poisson(pressure, nextPressure, RHS, flags, coordinates, coordinatesLength, numFluidCells, iMax, jMax, stepSizes, pressureResidualTolerance, pressureMinIterations, pressureMaxIterations, relaxationParameter, pressureResidualNorm);
        ComputeVelocities(velocities, FG, pressure, flags, iMax, jMax, timestep, stepSizes);
        iteration++;
    }
    auto endTime = std::chrono::high_resolution_clock::now();

    FreeMatrix(velocities.x, iMax + 2);
    FreeMatrix(velocities.y, iMax + 2);
    FreeMatrix(pressure, iMax + 2);
    FreeMatrix(nextPressure, iMax + 2);
    FreeMatrix(RHS, iMax + 2);
    FreeMatrix(FG.x, iMax + 2);
    FreeMatrix(FG.y, iMax + 2);
    return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 1000.0; // Time taken for n iterations, seconds
}

int main(int argc, char** argv) {
    
    if (argc < 2) {
        std::cout << "Command-line arguments in incorrect format. There must be one argument. Arguments are pipe or compute" << std::endl;
        return -1;
    }
    if (std::string(argv[1]) == std::string("pipe")) {
        FrontendManager frontendManager(50, 50, "NEAFluidDynamicsPipe");
        return frontendManager.Run();
    }
    else {
        std::cout << "Enter domain size" << std::endl;
        int squareLength;
        std::cin >> squareLength;
        StepTestSquare(squareLength);
        return 0;
    }
}