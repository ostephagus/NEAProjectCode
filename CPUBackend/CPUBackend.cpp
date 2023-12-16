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
#define DEBUGOUT
//#define FIELDOUT
//#define MULTITHREADING

void PrintField(REAL** field, int xLength, int yLength, std::string name) {
    std::cout << name << ": " << std::endl;
    for (int i = xLength-1; i >= 0; i--) {
        for (int j = 0; j < yLength; j++) {

            printf("%-8.3f", field[j][i]); //i and j are swapped here because we print first in the horizontal direction (i or u) then in the vertical (j or v)
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

void PrintFlagsArrows(BYTE** flags, int xLength, int yLength) {
    for (int i = xLength - 1; i >= 0; i--) {
        for (int j = 0; j < yLength; j++) {
            switch (flags[j][i]) {
            case B_N:
                std::cout << "^^";
                break;
            case B_NE:
                std::cout << "^>";
                break;
            case B_E:
                std::cout << ">>";
                break;
            case B_SE:
                std::cout << "v>";
                break;
            case B_S:
                std::cout << "vv";
                break;
            case B_SW:
                std::cout << "<v";
                break;
            case B_W:
                std::cout << "<<";
                break;
            case B_NW:
                std::cout << "<^";
                break;
            case OBS:
                std::cout << "()";
                break;
            case FLUID:
                std::cout << "  ";
                break;
            default:
                std::cout << "  ";
                break;
            }
        }
        std::cout << std::endl;
    }
}

void UnflattenArray(bool** pointerArray, bool* flattenedArray, int length, int divisions) {
    for (int i = 0; i < length / divisions; i++) {
        pointerArray[i] = flattenedArray + i * divisions;
    }
}

REAL TestParameters(REAL parameterValue, int iterations) {
    int iMax = 50, jMax = 50;

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
        Poisson(pressure, RHS, flags, coordinates, coordinatesLength, numFluidCells, iMax, jMax, stepSizes, pressureResidualTolerance, pressureMinIterations, pressureMaxIterations, relaxationParameter, pressureResidualNorm);
        ComputeVelocities(velocities, FG, pressure, flags, iMax, jMax, timestep, stepSizes);
        iteration++;
    }
    auto endTime = std::chrono::high_resolution_clock::now();

    FreeMatrix(velocities.x, iMax + 2);
    FreeMatrix(velocities.y, iMax + 2);
    FreeMatrix(pressure, iMax + 2);
    FreeMatrix(RHS, iMax + 2);
    FreeMatrix(FG.x, iMax + 2);
    FreeMatrix(FG.y, iMax + 2);
    return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / (REAL)1000.0; // Time taken for n iterations, seconds
}

void TestBoundaryHandling() {
    int iMax = 13, jMax = 13;
    bool obstaclesFlattened[225] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 
    };
    bool** obstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);
    UnflattenArray(obstacles, obstaclesFlattened, 255, 15);
    BYTE** flags = FlagMatrixMAlloc(iMax + 2, jMax + 2);
    SetFlags(obstacles, flags, iMax + 2, jMax + 2);

    std::pair<std::pair<int, int>*, int> coordinatesWithLength = FindBoundaryCells(flags, iMax, jMax);
    std::pair<int, int>* coordinates = coordinatesWithLength.first;
    int coordinatesLength = coordinatesWithLength.second;

    int numFluidCells = CountFluidCells(flags, iMax, jMax);
    
    //PrintFlagsArrows(flags, iMax + 2, jMax + 2);

    REAL** pressure = MatrixMAlloc(iMax + 2, jMax + 2);
    for (int i = 1; i <= iMax; i++) {
        for (int j = 1; j <= jMax; j++) {
            if (flags[i][j] & SELF) {
                pressure[i][j] = 1;
            }
        }
    }
    PrintField(pressure, iMax + 2, jMax + 2, "Pressure before");


}

void StepTestSquare(int squareLength, bool multiThreading) {
    int iMax = squareLength, jMax = squareLength;

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
    for (int i = 1; i <= iMax; i++) { for (int j = 1; j <= jMax; j++) { obstacles[i][j] = 1; } } //Set all the cells to fluid

    int boundaryLeft = (int)(0.45 * iMax);
    int boundaryRight = (int)(0.55 * iMax);
    int boundaryBottom = (int)(0.45 * jMax);
    int boundaryTop = (int)(0.55 * jMax);
    for (int i = boundaryLeft; i < boundaryRight; i++) { // Create a square of boundary cells
        for (int j = boundaryBottom; j < boundaryTop; j++) {
            obstacles[i][j] = 0;
        }
    }
    SetFlags(obstacles, flags, iMax + 2, jMax + 2);
#ifdef FIELDSOUT
    PrintFlagsArrows(flags, iMax + 2, jMax + 2);
#endif // FIELDSOUT



    std::pair<std::pair<int, int>*, int> coordinatesWithLength = FindBoundaryCells(flags, iMax, jMax);
    std::pair<int, int>* coordinates = coordinatesWithLength.first;
    int coordinatesLength = coordinatesWithLength.second;
    
    int numFluidCells = CountFluidCells(flags, iMax, jMax);

    const REAL width = 1;
    const REAL height = 1;
    const REAL timeStepSafetyFactor = 0.5;
    const REAL relaxationParameter = 1.7;
    const REAL pressureResidualTolerance = 1; //Needs experimentation
    const int pressureMinIterations = 10;
    const int pressureMaxIterations = 1000; //Needs experimentation
    const REAL reynoldsNo = 1000;
    const REAL inflowVelocity = 1;
    const REAL surfaceFrictionalPermissibility = 0;
    REAL pressureResidualNorm = 0;

    DoubleReal bodyForces;
    bodyForces.x = 0;
    bodyForces.y = 0;

    REAL timestep;
    DoubleReal stepSizes;
    stepSizes.x = width / iMax;
    stepSizes.y = height / jMax;

    /*for (int i = 0; i <= iMax+1; i++) {
        for (int j = 0; j <= jMax+1; j++) {
            pressure[i][j] = 1000;
        }
    }*/
    //PrintField(pressure, iMax+2, jMax+2, "Pressure");
    /*for (int i = 1; i <= iMax; i++) {
        for (int j = 1; j <= jMax; j++) {
            velocities.x[i][j] = 4;
            velocities.y[i][j] = 0;
        }
    }*/
    //PrintField(velocities.x, iMax + 2, jMax + 2, "Horizontal velocities");
    //PrintField(velocities.y, iMax + 2, jMax + 2, "Vertical velocities");

    int iteration = 0;
    std::cout << "Enter number of iterations ";
    int iterMax;
    std::cin >> iterMax;
    //int iterMax = 10; //TESTING
    int pressureIterations;
    while (iteration < iterMax) {
        std::cout << "Iteration " << iteration << std::endl;
        //auto startTime = std::chrono::high_resolution_clock::now(); //TESTING
        SetBoundaryConditions(velocities, flags, coordinates, coordinatesLength, iMax, jMax, inflowVelocity, surfaceFrictionalPermissibility);
        ComputeTimestep(timestep, iMax, jMax, stepSizes, velocities, reynoldsNo, timeStepSafetyFactor);
#ifdef DEBUGOUT
        std::cout << "Timestep: " << timestep << std::endl;
#endif // DEBUGOUT
#ifdef FIELDOUT
        PrintField(velocities.x, iMax + 2, jMax + 2, "Horizontal velocities");
        PrintField(velocities.y, iMax + 2, jMax + 2, "Vertical velocities");
#endif // FIELDOUT
        REAL gamma = ComputeGamma(velocities, iMax, jMax, timestep, stepSizes);
        ComputeFG(velocities, FG, flags, iMax, jMax, timestep, stepSizes, bodyForces, gamma, reynoldsNo);
#ifdef FIELDOUT
        PrintField(FG.x, iMax + 2, jMax + 2, "F");
        PrintField(FG.y, iMax + 2, jMax + 2, "G");
#endif // FIELDOUT
        ComputeRHS(FG, RHS, flags, iMax, jMax, timestep, stepSizes);
#ifdef FIELDOUT
        PrintField(RHS, iMax + 2, jMax + 2, "Pressure RHS");
#endif // FIELDOUT
#ifdef MULTITHREADING
        pressureIterations = PoissonMultiThreaded(pressure, RHS, flags, coordinates, coordinatesLength, numFluidCells, iMax, jMax, stepSizes, pressureResidualTolerance, pressureMinIterations, pressureMaxIterations, relaxationParameter, pressureResidualNorm);
#else
        pressureIterations = Poisson(pressure, RHS, flags, coordinates, coordinatesLength, numFluidCells, iMax, jMax, stepSizes, pressureResidualTolerance, pressureMinIterations, pressureMaxIterations, relaxationParameter, pressureResidualNorm);
#endif // MULTITHREADING
#ifdef FIELDOUT
        PrintField(pressure, iMax + 2, jMax + 2, "Pressure");
#endif // FIELDOUT
        ComputeVelocities(velocities, FG, pressure, flags, iMax, jMax, timestep, stepSizes);
        //auto endTime = std::chrono::high_resolution_clock::now(); //TESTING
#ifdef DEBUGOUT
        std::cout << "Iteration " << iteration << ": starting velocity " << velocities.x[2][jMax / 2] << ", velocity before " << velocities.x[boundaryLeft-1][(boundaryTop + boundaryBottom)/2] << ", velocity after " << velocities.x[boundaryRight + 1][(boundaryTop + boundaryBottom) / 2] << ", pressure before " << pressure[boundaryLeft - 1][(boundaryTop + boundaryBottom) / 2] << ", pressure after " << pressure[boundaryRight + 1][(boundaryTop + boundaryBottom) / 2] << ", residual norm " << pressureResidualNorm << std::endl;
#endif // DEBUGOUT
        //std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 1000.0 << " seconds." << std::endl;
        iteration++;
    }


    FreeMatrix(velocities.x, iMax + 2);
    FreeMatrix(velocities.y, iMax + 2);
    FreeMatrix(pressure, iMax + 2);
    FreeMatrix(RHS, iMax + 2);
    FreeMatrix(FG.x, iMax + 2);
    FreeMatrix(FG.y, iMax + 2);
}

int main(int argc, char** argv) {
    
    if (argc < 2) {
        std::cout << "Command-line arguments in incorrect format. There must be one argument." << std::endl;
        return -1;
    }
    if (std::string(argv[1]) == std::string("pipe")) {
        FrontendManager frontendManager(100, 100, "NEAFluidDynamicsPipe");
        return frontendManager.Run();
    }
    else if (std::string(argv[1]) == std::string("compute")) {
        std::cout << "Enter domain size" << std::endl;
        int squareLength;
        std::cin >> squareLength;
        bool multiThreading = false;
        if (argc > 2) {
            multiThreading = true;
        }
        StepTestSquare(squareLength, multiThreading);
        return 0;
    }
    else {
        TestBoundaryHandling();
    }
}