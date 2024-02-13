﻿#include "pch.h"
#include "Solver.h"
#include "GPUSolver.cuh"
#include "BackendCoordinator.h"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    int iMax = 200;
    int jMax = 100;
    SimulationParameters parameters = SimulationParameters();
    if (argc == 1 || (argc == 2 && strcmp(argv[1], "debug") == 0)) { // Not linked to a frontend.
        std::cout << "Running without a fronted attached.\n";
        parameters.width = 1;
        parameters.height = 1;
        parameters.timeStepSafetyFactor = (REAL)0.5;
        parameters.relaxationParameter = (REAL)1.7;
        parameters.pressureResidualTolerance = 2;
        parameters.pressureMinIterations = 5;
        parameters.pressureMaxIterations = 1000;
        parameters.reynoldsNo = 2000;
        parameters.dynamicViscosity = (REAL)0.00001983;
        parameters.fluidDensity = (REAL)1.293;
        parameters.inflowVelocity = 5;
        parameters.surfaceFrictionalPermissibility = 0;
        parameters.bodyForces.x = 0;
        parameters.bodyForces.y = 0;

        GPUSolver solver = GPUSolver(parameters, iMax, jMax);

        bool** obstacles = solver.GetObstacles();
        for (int i = 1; i <= iMax; i++) { for (int j = 1; j <= jMax; j++) { obstacles[i][j] = 1; } } // Set all the cells to fluid

        int boundaryLeft = (int)(0.45 * iMax);
        int boundaryRight = (int)(0.45 * iMax + 2);
        int boundaryBottom = (int)(0.45 * jMax);
        int boundaryTop = (int)(0.55 * jMax);
        for (int i = boundaryLeft; i < boundaryRight; i++) { // Create a square of boundary cells
            for (int j = boundaryBottom; j < boundaryTop; j++) {
                obstacles[i][j] = 0;
            }
        }

        std::cout << "Obstacle set to a line." << std::endl;

        solver.ProcessObstacles();
        solver.PerformSetup();

        REAL cumulativeTimestep = 0;

        int numIterations = 100;
        std::cerr << "2 seconds to attach profiler / debugger\n";
        Sleep(2000);
        /*std::cout << "Enter number of iterations: ";
        std::cin >> numIterations;*/

        float timeTakenSum = 0;

        for (int i = 0; i < numIterations; i++) {
            solver.Timestep(cumulativeTimestep);
            REAL dragCoefficient = solver.GetDragCoefficient();
            std::cout << "Iteration " << i << ", time taken: " << cumulativeTimestep << ", drag coefficient " << dragCoefficient << ".\n";
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
        std::cerr << "Incorrect number of command-line arguments. Run the executable with the pipe name to connect to a frontend, or without to run without a frontend.\n";
        return -1;
    }
}