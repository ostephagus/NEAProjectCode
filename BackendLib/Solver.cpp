#include "pch.h"
#include "Solver.h"

Solver::Solver(SimulationParameters parameters, int iMax, int jMax) : iMax(iMax), jMax(jMax), parameters(parameters) {
    velocities = DoubleField();
    velocities.x = nullptr;
    velocities.y = nullptr;

    pressure = nullptr;
    RHS = nullptr;
    streamFunction = nullptr;

    FG.x = nullptr;
    FG.y = nullptr;

    obstacles = nullptr; // nullptr to represent uninitialised
    flags = nullptr;
    coordinates = nullptr;
    coordinatesLength = 0;
    numFluidCells = 0;
}

Solver::~Solver() {

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


