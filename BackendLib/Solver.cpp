#include "pch.h"
#include "Solver.h"

Solver::Solver(SimulationParameters parameters, int iMax, int jMax) : iMax(iMax), jMax(jMax), parameters(parameters) {}

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
