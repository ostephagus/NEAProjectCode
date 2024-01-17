﻿#include "pch.h"
#include "Solver.h"
#include "GPUSolver.cuh"
#include "BackendCoordinator.h"
#include <iostream>

int main(int argc, char** argv) {
    int iMax = 100;
    int jMax = 100;
    SimulationParameters parameters = SimulationParameters();
    if (argc == 1) { // Not linked to a frontend.
        parameters.width = 1;
        parameters.height = 1;
        parameters.timeStepSafetyFactor = (REAL)0.5;
        parameters.relaxationParameter = (REAL)1.7;
        parameters.pressureResidualTolerance = 1;
        parameters.pressureMinIterations = 10;
        parameters.pressureMaxIterations = 1000;
        parameters.reynoldsNo = 1000;
        parameters.inflowVelocity = 1;
        parameters.surfaceFrictionalPermissibility = 0;
        DoubleReal bodyForces = DoubleReal();
        bodyForces.x = 0;
        bodyForces.y = 0;
        parameters.bodyForces = bodyForces;

        GPUSolver solver = GPUSolver(parameters, iMax, jMax);

        bool** obstacles = solver.GetObstacles();
        for (int i = 1; i <= iMax; i++) { for (int j = 1; j <= jMax; j++) { obstacles[i][j] = 1; } } // Set all the cells to fluid

        int boundaryLeft = (int)(0.45 * iMax);
        int boundaryRight = (int)(0.55 * iMax);
        int boundaryBottom = (int)(0.45 * jMax);
        int boundaryTop = (int)(0.55 * jMax);
        for (int i = boundaryLeft; i < boundaryRight; i++) { // Create a square of boundary cells
            for (int j = boundaryBottom; j < boundaryTop; j++) {
                obstacles[i][j] = 0;
            }
        }

        solver.ProcessObstacles();
        solver.PerformSetup();

        REAL cumulativeTimestep = 0;

        int numIterations = 1;
        //std::cout << "Enter number of iterations: ";
        //std::cin >> numIterations;

        for (int i = 0; i < numIterations; i++) {
            solver.Timestep(cumulativeTimestep);
            std::cout << "Iteration " << i << ", time taken: " << cumulativeTimestep << "." << std::endl;
        }
        return 0;
    }
    else if (argc == 2) { // Linked to a frontend.
        char* pipeName = argv[1];
        Solver* solver = new GPUSolver(parameters, iMax, jMax);
        BackendCoordinator backendCoordinator(iMax, jMax, std::string(pipeName), solver);
        int retValue = backendCoordinator.Run();
        delete solver;
        return retValue;
    }
    else {
        std::cerr << "Incorrect number of command-line arguments. Run the executable with the pipe name to connect to a frontend, or without to run without a frontend." << std::endl;
        return -1;
    }
}