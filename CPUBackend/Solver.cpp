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

Solver::~Solver() {
    FreeMatrix(velocities.x, iMax + 2);
    FreeMatrix(velocities.y, iMax + 2);
    FreeMatrix(pressure, iMax + 2);
    FreeMatrix(RHS, iMax + 2);
    FreeMatrix(streamFunction, iMax + 1);
    FreeMatrix(FG.x, iMax + 2);
    FreeMatrix(FG.y, iMax + 2);
    FreeMatrix(obstacles, iMax + 2);
    FreeMatrix(flags, iMax + 2);
}

SimulationParameters Solver::GetParameters() const {
    return parameters;
}

void Solver::SetParameters(SimulationParameters parameters) {
    this->parameters = parameters;
}

int Solver::GetIMax() const {
    return iMax;
}

int Solver::GetJMax() const {
    return jMax;
}

REAL** Solver::GetHorizontalVelocity() const {
    return velocities.x;
}

REAL** Solver::GetVerticalVelocity() const {
    return velocities.y;
}

REAL** Solver::GetPressure() const {
    return pressure;
}

REAL** Solver::GetStreamFunction() const {
    return streamFunction;
}

bool** Solver::GetObstacles() {
    if (obstacles == nullptr) {
        obstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);
    }
    return obstacles;
}

void Solver::ProcessObstacles() {
    SetFlags(obstacles, flags, iMax + 2, jMax + 2);

    std::pair<std::pair<int, int>*, int> coordinatesWithLength = FindBoundaryCells(flags, iMax, jMax);
    coordinates = coordinatesWithLength.first;
    coordinatesLength = coordinatesWithLength.second;

    numFluidCells = CountFluidCells(flags, iMax, jMax);
}
