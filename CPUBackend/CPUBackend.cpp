#include <iostream>
#include <string>
#include "Boundary.h"
#include "Computation.h"
#include "DiscreteDerivatives.h"
#include "Init.h"

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

void GranularTesting() {
    REAL** testu = MatrixMAlloc(3, 3);
    testu[0][0] = 0;
    testu[0][1] = 5;
    testu[0][2] = 3;
    testu[1][0] = 2;
    testu[1][1] = 1;
    testu[1][2] = 6;
    testu[2][0] = 3;
    testu[2][1] = 4;
    testu[2][2] = 0;

    REAL** testv = MatrixMAlloc(3, 3);
    testv[0][0] = 1;
    testv[0][1] = 2;
    testv[0][2] = 5;
    testv[1][0] = 2;
    testv[1][1] = 3;
    testv[1][2] = 4;
    testv[2][0] = 5;
    testv[2][1] = 6;
    testv[2][2] = 1;

    PrintField(testu, 3, 3, "Test horizontal velocities");
    PrintField(testv, 3, 3, "Test vertical velocities");
    DoubleField velocities;
    velocities.x = testu;
    velocities.y = testv;
    DoubleReal stepsizes;
    stepsizes.x = 0.1;
    stepsizes.y = 1.25;

    std::cout << PvSquaredPy(velocities.y, 1, 1, stepsizes.y, 1.7);
    FreeMatrix(testu, 3);
    FreeMatrix(testv, 3);
}

void StepTest5x5() {
    int iMax = 3, jMax = 3;

    DoubleField velocities;
    velocities.x = MatrixMAlloc(iMax + 2, jMax + 2);
    velocities.y = MatrixMAlloc(iMax + 2, jMax + 2);

    REAL** pressure = MatrixMAlloc(iMax + 2, jMax + 2);
    REAL** RHS = MatrixMAlloc(iMax + 2, jMax + 2);

    DoubleField FG;
    FG.x = MatrixMAlloc(iMax + 2, jMax + 2);
    FG.y = MatrixMAlloc(iMax + 2, jMax + 2);

    const REAL width = 2;
    const REAL height = 2;
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
            pressure[i][j] = 1;
        }
    }
    PrintField(pressure, iMax+2, jMax+2, "Pressure");
    for (int i = 1; i <= iMax; i++) {
        for (int j = 1; j <= jMax; j++) {
            velocities.x[i][j] = 4;
            velocities.y[i][j] = 0;
        }
    }
    PrintField(velocities.x, iMax + 2, jMax + 2, "Horizontal velocities");
    PrintField(velocities.y, iMax + 2, jMax + 2, "Vertical velocities");

    while (true) {//BREAAKPOINT REQUIRED
        ComputeTimestep(timestep, iMax, jMax, stepSizes, velocities, reynoldsNo, timeStepSafetyFactor);
        std::cout << timestep << std::endl;
        SetBoundaryConditions(velocities, iMax, jMax, inflowVelocity);
        PrintField(velocities.x, iMax + 2, jMax + 2, "Horizontal velocities");
        PrintField(velocities.y, iMax + 2, jMax + 2, "Vertical velocities");
        REAL gamma = ComputeGamma(velocities, iMax, jMax, timestep, stepSizes);
        ComputeFG(velocities, FG, iMax, jMax, timestep, stepSizes, bodyForces, gamma, reynoldsNo);
        ComputeRHS(FG, RHS, iMax, jMax, timestep, stepSizes);
        Poisson(pressure, RHS, iMax, jMax, stepSizes, pressureResidualTolerance, pressureMaxIterations, relaxationParameter, pressureResidualNorm);
        ComputeVelocities(velocities, FG, pressure, iMax, jMax, timestep, stepSizes);
    }
}

void TestAll()
{

    std::cout << "Initialising variables";

    int iMax = 50;
    int jMax = 50;

    DoubleField velocities;
    velocities.x = MatrixMAlloc(iMax+2, jMax+2);
    velocities.y = MatrixMAlloc(iMax + 2, jMax + 2);

    std::cout << '.';

    REAL** pressure = MatrixMAlloc(iMax + 2, jMax + 2);
    REAL** RHS = MatrixMAlloc(iMax + 2, jMax + 2);
    
    DoubleField FG;
    FG.x = MatrixMAlloc(iMax + 2, jMax + 2);
    FG.y = MatrixMAlloc(iMax + 2, jMax + 2);

    std::cout << '.';

    const REAL timeStepSafetyFactor = 0.8;
    const REAL relaxationParameter = 1.7;
    const REAL pressureResidualTolerance = 1; //Needs experimentation
    const int pressureMaxIterations = 1000; //Needs experimentation
    const REAL reynoldsNo = 2000;
    const REAL inflowVelocity = 2;
    REAL pressureResidualNorm = 0;

    DoubleReal bodyForces;
    bodyForces.x = 0;
    bodyForces.y = 0;

    REAL timestep;
    DoubleReal stepSizes;
    stepSizes.x = (REAL)2 / iMax;
    stepSizes.y = (REAL)2 / jMax;

    std::cout << '.';

    for (int j = 1; j < jMax; j++) {
        velocities.x[0][j] = 2;
    }
    for (int i = 1; i <= iMax; i++) {
        for (int j = 1; j <= jMax; j++) {
            pressure[i][j] = 1;
        }
    }


    std::cout << "\nInitialised. Press any key to begin and q to halt" << std::endl;
    std::string userInput;
    std::cin >> userInput;
    while (userInput != "q") {
        ComputeTimestep(timestep, iMax, jMax, stepSizes, velocities, reynoldsNo, timeStepSafetyFactor);
        SetBoundaryConditions(velocities, iMax, jMax, inflowVelocity);
        ComputeFG(velocities, FG, iMax, jMax, timestep, stepSizes, bodyForces, ComputeGamma(velocities, iMax, jMax, timestep, stepSizes), reynoldsNo);
        PrintField(FG.x, iMax + 2, jMax + 2, "F");
        PrintField(FG.y, iMax + 2, jMax + 2, "G");

        ComputeRHS(FG, RHS, iMax, jMax, timestep, stepSizes);
        PrintField(RHS, iMax + 2, jMax + 2, "RHS");

        std::cout << "Number of pressure iterations: " << Poisson(pressure, RHS, iMax, jMax, stepSizes, pressureResidualTolerance, pressureMaxIterations, relaxationParameter, pressureResidualNorm) << std::endl;
        std::cout << "Residual norm: " << pressureResidualNorm << std::endl;
        PrintField(pressure, iMax + 2, jMax + 2, "Pressure");

        ComputeVelocities(velocities, FG, pressure, iMax, jMax, timestep, stepSizes);
        PrintField(velocities.x, iMax + 2, jMax + 2, "Horizontal velocities");
        PrintField(velocities.y, iMax + 2, jMax + 2, "Vertical velocities");

        std::cout << "Press enter to begin next iteration, press q to exit." << std::endl;
        std::cin >> userInput;
    }

    FreeMatrix(velocities.x, iMax);
    FreeMatrix(velocities.y, iMax);
    FreeMatrix(pressure, iMax);
    FreeMatrix(RHS, iMax);
    FreeMatrix(FG.x, iMax);
    FreeMatrix(FG.y, iMax);
}

int main() {
    StepTest5x5();
    return 0;
}