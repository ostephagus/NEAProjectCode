#include "FrontendManager.h"
#include "PipeConstants.h"
#include <iostream>
#include "PipeManager.h"
#include "Computation.h"
#include "Init.h"
#include "Boundary.h"

void FrontendManager::UnflattenArray(bool** pointerArray, bool* flattenedArray, int length, int divisions) {
    for (int i = 0; i < length / divisions; i++) {
        pointerArray[i] = flattenedArray + i * divisions;
    }
}

void FrontendManager::SetupParameters(REAL** horizontalVelocity, REAL** verticalVelocity, REAL** pressure, SimulationParameters& parameters, DoubleReal stepSizes) {
    parameters.width = 1;
    parameters.height = 1;
    parameters.timeStepSafetyFactor = 0.8;
    parameters.relaxationParameter = 1.7;
    parameters.pressureResidualTolerance = 1;
    parameters.pressureMaxIterations = 1000;
    parameters.reynoldsNo = 2000;
    parameters.inflowVelocity = 5;
    parameters.surfaceFrictionalPermissibility = 0;
    parameters.bodyForces.x = 0;
    parameters.bodyForces.y = 0;
    stepSizes.x = parameters.width / iMax;
    stepSizes.y = parameters.height / jMax;

    for (int i = 0; i < iMax; i++) {
        for (int j = 0; j < jMax; j++) {
            pressure[i][j] = 100;
        }
    }
}

void FrontendManager::TimeStep(DoubleField velocities, DoubleField FG, REAL** pressure, REAL** nextPressure, REAL** RHS, REAL** streamFunction, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int numFluidCells, const SimulationParameters& parameters, DoubleReal stepSizes) {
    
    REAL timestep;
    REAL pressureResidualNorm;
    ComputeTimestep(timestep, iMax, jMax, stepSizes, velocities, parameters.reynoldsNo, parameters.timeStepSafetyFactor);
    SetBoundaryConditions(velocities, flags, coordinates, coordinatesLength, iMax, jMax, parameters.inflowVelocity, parameters.surfaceFrictionalPermissibility);
    REAL gamma = ComputeGamma(velocities, iMax, jMax, timestep, stepSizes);
    ComputeFG(velocities, FG, flags, iMax, jMax, timestep, stepSizes, parameters.bodyForces, gamma, parameters.reynoldsNo);
    ComputeRHS(FG, RHS, flags, iMax, jMax, timestep, stepSizes);
    Poisson(pressure, nextPressure, RHS, flags, coordinates, coordinatesLength, numFluidCells, iMax, jMax, stepSizes, parameters.pressureResidualTolerance, parameters.pressureMaxIterations, parameters.relaxationParameter, pressureResidualNorm);
    ComputeVelocities(velocities, FG, pressure, flags, iMax, jMax, timestep, stepSizes);
    std::cerr << "Timestep" << std::endl;
}

void FrontendManager::HandleRequest(BYTE requestByte) {
    if ((requestByte & ~PipeConstants::Request::PARAMMASK) == PipeConstants::Request::CONTREQ) {
        if (requestByte == PipeConstants::Request::CONTREQ) {
            pipeManager.SendByte(PipeConstants::Error::BADPARAM);
            std::cerr << "Server sent a blank request, exiting";
            return;
        }
        bool hVelWanted = requestByte & PipeConstants::Request::HVEL;
        bool vVelWanted = requestByte & PipeConstants::Request::VVEL;
        bool pressureWanted = requestByte & PipeConstants::Request::PRES;
        bool streamWanted = requestByte & PipeConstants::Request::STRM;

        DoubleField velocities;
        velocities.x = MatrixMAlloc(iMax + 2, jMax + 2);
        velocities.y = MatrixMAlloc(iMax + 2, jMax + 2);

        REAL** pressure = MatrixMAlloc(iMax + 2, jMax + 2);
        REAL** nextPressure = MatrixMAlloc(iMax + 2, jMax + 2);
        REAL** RHS = MatrixMAlloc(iMax + 2, jMax + 2);
        REAL** streamFunction = MatrixMAlloc(iMax, jMax);

        BYTE** flags = FlagMatrixMAlloc(iMax + 2, jMax + 2);
        bool** obstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);
        for (int i = 1; i <= iMax; i++) { for (int j = 1; j <= jMax; j++) { obstacles[i][j] = 1; } } //Set all the cells to fluid
        SetFlags(obstacles, flags, iMax + 2, jMax + 2);

        std::pair<std::pair<int, int>*, int> coordinatesWithLength = FindBoundaryCells(flags, iMax, jMax);
        std::pair<int, int>* coordinates = coordinatesWithLength.first;
        int coordinatesLength = coordinatesWithLength.second;

        int numFluidCells = CountFluidCells(flags, iMax, jMax);

        DoubleField FG;
        FG.x = MatrixMAlloc(iMax + 2, jMax + 2);
        FG.y = MatrixMAlloc(iMax + 2, jMax + 2);

        SimulationParameters parameters;
        DoubleReal stepSizes;
        stepSizes.x = 0;
        stepSizes.y = 0;
        SetupParameters(velocities.x, velocities.y, pressure, parameters, stepSizes);

        bool stopRequested = false;

        pipeManager.SendByte(PipeConstants::Status::OK); // Send OK to say backend is set up and about to start executing

        while (!stopRequested) {
            pipeManager.SendByte(PipeConstants::Marker::ITERSTART);
            TimeStep(velocities, FG, pressure, nextPressure, RHS, streamFunction, flags, coordinates, coordinatesLength, numFluidCells, parameters, stepSizes);
            if (hVelWanted) {
                pipeManager.SendByte(PipeConstants::Marker::FLDSTART | PipeConstants::Marker::HVEL);
                pipeManager.SendField(velocities.x, iMax, jMax, 1, 1); // Set the offsets to 1 and the length to iMax / jMax to exclude boundary cells at cells 0 and max
                pipeManager.SendByte(PipeConstants::Marker::FLDEND | PipeConstants::Marker::HVEL);
            }
            if (vVelWanted) {
                pipeManager.SendByte(PipeConstants::Marker::FLDSTART | PipeConstants::Marker::VVEL);
                pipeManager.SendField(velocities.y, iMax, jMax, 1, 1);
                pipeManager.SendByte(PipeConstants::Marker::FLDEND | PipeConstants::Marker::VVEL);
            }
            if (pressureWanted) {
                pipeManager.SendByte(PipeConstants::Marker::FLDSTART | PipeConstants::Marker::PRES);
                pipeManager.SendField(pressure, iMax, jMax, 1, 1);
                pipeManager.SendByte(PipeConstants::Marker::FLDEND | PipeConstants::Marker::PRES);
            }
            if (streamWanted) {
                pipeManager.SendByte(PipeConstants::Marker::FLDSTART | PipeConstants::Marker::STRM);
                pipeManager.SendField(streamFunction, iMax, jMax, 0, 0);
                pipeManager.SendByte(PipeConstants::Marker::FLDEND | PipeConstants::Marker::STRM);
            }

            pipeManager.SendByte(PipeConstants::Marker::ITEREND);

            BYTE receivedByte = pipeManager.ReadByte(); // This may require duplex communication

            stopRequested = receivedByte == PipeConstants::Status::STOP || receivedByte == PipeConstants::Error::INTERNAL; // Stop if requested or the frontend fatally errors
        }

        pipeManager.SendByte(PipeConstants::Status::OK); // Send OK then stop executing

    }
    else { // Only continuous requests are supported
        std::cerr << "Server sent an unsupported request" << std::endl;
        pipeManager.SendByte(PipeConstants::Error::BADREQ);
    }
}

void FrontendManager::ReceiveData(BYTE startMarker) {
    if (startMarker == (PipeConstants::Marker::FLDSTART | PipeConstants::Marker::OBST)) { // Only supported use is obstacle send
        bool* obstaclesFlattened = new bool[fieldSize];
        bool** obstacles = ObstacleMatrixMAlloc(iMax, jMax);
        UnflattenArray(obstacles, obstaclesFlattened, fieldSize, jMax);
    }
    else {
        std::cerr << "Server sent unsupported data" << std::endl;
        pipeManager.SendByte(PipeConstants::Error::BADREQ); // All others not supported at the moment
    }
}

FrontendManager::FrontendManager(int iMax, int jMax, std::string pipeName)
    : iMax(iMax), jMax(jMax), fieldSize(iMax * jMax), pipeManager(pipeName)
{}

int FrontendManager::Run() {
    pipeManager.Handshake(iMax, jMax);
    std::cerr << "Handshake completed ok" << std::endl;

    bool closeRequested = false;

    while (!closeRequested) {
        BYTE receivedByte = pipeManager.ReadByte();
        switch (receivedByte & PipeConstants::CATEGORYMASK) { // Gets the category of control byte
        case PipeConstants::Status::GENERIC: // Status bytes
            switch (receivedByte & ~PipeConstants::Status::PARAMMASK) {
            case PipeConstants::Status::HELLO:
            case PipeConstants::Status::BUSY:
            case PipeConstants::Status::OK:
            case PipeConstants::Status::STOP:
                std::cerr << "Server sent a status byte out of sequence, request not understood" << std::endl;
                pipeManager.SendByte(PipeConstants::Error::BADREQ);
                break;
            case PipeConstants::Status::CLOSE:
                closeRequested = true;
                break;
            default:
                std::cerr << "Server sent a malformed status byte, request not understood" << std::endl;
                pipeManager.SendByte(PipeConstants::Error::BADREQ);
                break;
            }
            break;
        case PipeConstants::Request::GENERIC: // Request bytes have a separate handler
            HandleRequest(receivedByte);
            break;
        case PipeConstants::Marker::GENERIC: // So do marker bytes
            ReceiveData(receivedByte);
            break;
        default: // Error bytes
            break;
        }
    }

    //delete[] obstaclesFlattened;
    return 0;
}