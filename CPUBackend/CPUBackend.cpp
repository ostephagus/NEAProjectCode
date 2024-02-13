#include "pch.h"
#include "Solver.h"
#include "CPUSolver.h"
#include "BackendCoordinator.h"
#include <iostream>

//#define WAIT_FOR_DEBUGGER_ATTACH

int main(int argc, char** argv) {
#ifdef WAIT_FOR_DEBUGGER_ATTACH
    char nonsense;
    std::cout << "Press a character and press enter once debugger is attched. ";
    std::cin >> nonsense;
#endif // WAIT_FOR_DEBUGGER_ATTACH

    int iMax = 200;
    int jMax = 100;
    SimulationParameters parameters = SimulationParameters();
    if (argc == 1) { // Not linked to a frontend.
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

        CPUSolver solver = CPUSolver(parameters, iMax, jMax);

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

        int numIterations = 0;
        std::cout << "Enter number of iterations: ";
        std::cin >> numIterations;

        for (int i = 0; i < numIterations; i++) {
            solver.Timestep(cumulativeTimestep);
            REAL dragCoefficient = solver.GetDragCoefficient();
            std::cout << "Iteration " << i << ", time taken: " << cumulativeTimestep << ", drag coefficient " << dragCoefficient << ".\n";
        }
        return 0;
    }
    else if (argc == 2) { // Linked to a frontend.
        char* pipeName = argv[1];
        Solver* solver = new CPUSolver(parameters, iMax, jMax);
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