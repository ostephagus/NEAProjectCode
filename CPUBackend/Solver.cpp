#include "Solver.h"
#include "Boundary.h" // FindBoundaryCells, CountFluidCells
#include "Init.h" // MatrixMAlloc, FlagMatrixMAlloc, SetFlags

Solver::Solver(SimulationParameters parameters, int iMax, int jMax) : iMax(iMax), jMax(jMax), parameters(parameters) {
    velocities = DoubleField();
    velocities.x = MatrixMAlloc(iMax + 2, jMax + 2);
    velocities.y = MatrixMAlloc(iMax + 2, jMax + 2);

    pressure = MatrixMAlloc(iMax + 2, jMax + 2);
    RHS = MatrixMAlloc(iMax + 2, jMax + 2);
    streamFunction = MatrixMAlloc(iMax + 1, jMax + 1);

    FG.x = MatrixMAlloc(iMax + 2, jMax + 2);
    FG.y = MatrixMAlloc(iMax + 2, jMax + 2);

    obstacles = nullptr; // nullptr to represent uninitialised
    flags = FlagMatrixMAlloc(iMax + 2, jMax + 2);
    coordinates = nullptr;
    coordinatesLength = 0;
    numFluidCells = 0;
}

SimulationParameters Solver::GetParameters() const {
    return parameters;
}

void Solver::SetParameters(SimulationParameters parameters) {
    this->parameters = parameters;
}

void Solver::ProcessObstacles() {
    SetFlags(obstacles, flags, iMax + 2, jMax + 2);

    std::pair<std::pair<int, int>*, int> coordinatesWithLength = FindBoundaryCells(flags, iMax, jMax);
    coordinates = coordinatesWithLength.first;
    coordinatesLength = coordinatesWithLength.second;

    numFluidCells = CountFluidCells(flags, iMax, jMax);
}

void Solver::ProcessObstacles(bool** obstacles) {
    this->obstacles = obstacles;
    ProcessObstacles();
}