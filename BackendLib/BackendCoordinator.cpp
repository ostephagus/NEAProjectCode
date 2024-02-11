#include "pch.h"
#include "BackendCoordinator.h"
#include "PipeConstants.h"
#include <iostream>
#define OBSTACLES

void BackendCoordinator::UnflattenArray(bool** pointerArray, bool* flattenedArray, int length, int divisions) {
    for (int i = 0; i < length / divisions; i++) {

        memcpy(
            pointerArray[i],                // Destination address - address at ith pointer
            flattenedArray + i * divisions, // Source start address - move (i * divisions) each iteration
            divisions * sizeof(bool)        // Bytes to copy - divisions
        );

    }
}

void BackendCoordinator::HandleRequest(BYTE requestByte) {
    std::cout << "Starting execution of timestepping loop\n";
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

        bool closeRequested = false;
        pipeManager.SendByte(PipeConstants::Status::OK); // Send OK to say backend is set up and about to start executing

        int iteration = 0;
        REAL cumulativeTimestep = 0;
        solver->PerformSetup();
        while (!closeRequested) {
            std::cout << "Iteration " << iteration << ", " << cumulativeTimestep << " seconds passed. \n";
            pipeManager.SendByte(PipeConstants::Marker::ITERSTART);

            solver->Timestep(cumulativeTimestep);

            int iMax = solver->GetIMax();
            int jMax = solver->GetJMax();

            if (hVelWanted) {
                pipeManager.SendByte(PipeConstants::Marker::FLDSTART | PipeConstants::Marker::HVEL);
                pipeManager.SendField(solver->GetHorizontalVelocity(), iMax * jMax);
                pipeManager.SendByte(PipeConstants::Marker::FLDEND | PipeConstants::Marker::HVEL);
            }
            if (vVelWanted) {
                pipeManager.SendByte(PipeConstants::Marker::FLDSTART | PipeConstants::Marker::VVEL);
                pipeManager.SendField(solver->GetVerticalVelocity(), iMax * jMax);
                pipeManager.SendByte(PipeConstants::Marker::FLDEND | PipeConstants::Marker::VVEL);
            }
            if (pressureWanted) {
                pipeManager.SendByte(PipeConstants::Marker::FLDSTART | PipeConstants::Marker::PRES);
                pipeManager.SendField(solver->GetPressure(), iMax * jMax);
                pipeManager.SendByte(PipeConstants::Marker::FLDEND | PipeConstants::Marker::PRES);
            }
            if (streamWanted) {
                pipeManager.SendByte(PipeConstants::Marker::FLDSTART | PipeConstants::Marker::STRM);
                pipeManager.SendField(solver->GetStreamFunction(), iMax * jMax);
                pipeManager.SendByte(PipeConstants::Marker::FLDEND | PipeConstants::Marker::STRM);
            }

            pipeManager.SendByte(PipeConstants::Marker::PRMSTART | PipeConstants::Marker::DRAGCOEF);
            pipeManager.SendReal(solver->GetDragCoefficient());
            pipeManager.SendByte(PipeConstants::Marker::PRMEND | PipeConstants::Marker::DRAGCOEF);

            pipeManager.SendByte(PipeConstants::Marker::ITEREND);

            BYTE receivedByte = pipeManager.ReadByte();
            if (receivedByte == PipeConstants::Status::STOP) { // Stop means just wait for the next read
                pipeManager.SendByte(PipeConstants::Status::OK);
                std::cout << "Backend paused.\n";
                receivedByte = pipeManager.ReadByte();
            }
            if (receivedByte == PipeConstants::Status::CLOSE || receivedByte == PipeConstants::Error::INTERNAL) {
                closeRequested = true; // Stop if requested or the frontend fatally errors
            }
            else { // Anything other than a CLOSE request
                while ((receivedByte & ~PipeConstants::Marker::PRMMASK) == PipeConstants::Marker::PRMSTART || receivedByte == (PipeConstants::Marker::FLDSTART | PipeConstants::Marker::OBST)) { // While the received byte is a PRMSTART or obstacle send...
                    ReceiveData(receivedByte); // ...pass the received byte to ReceiveData to handle parameter reading...
                    receivedByte = pipeManager.ReadByte(); // ...then read the next byte
                }
                if (receivedByte != PipeConstants::Status::OK) { // Require an OK at the end, whether parameters were sent or not
                    std::cerr << "Server sent malformed data\n";
                    pipeManager.SendByte(PipeConstants::Error::BADREQ);
                }
            }

            iteration++;
        }
        std::cout << "Backend stopped.\n";

        pipeManager.SendByte(PipeConstants::Status::OK); // Send OK then stop executing

    }
    else { // Only continuous requests are supported
        std::cerr << "Server sent an unsupported request\n";
        pipeManager.SendByte(PipeConstants::Error::BADREQ);
    }
}



void BackendCoordinator::ReceiveObstacles()
{
    int iMax = solver->GetIMax();
    int jMax = solver->GetJMax();
    bool* obstaclesFlattened = new bool[(iMax + 2) * (jMax + 2)]();
    pipeManager.ReceiveObstacles(obstaclesFlattened, iMax + 2, jMax + 2);
    bool** obstacles = solver->GetObstacles();
    UnflattenArray(obstacles, obstaclesFlattened, (iMax + 2) * (jMax + 2), jMax + 2);
    delete[] obstaclesFlattened;
}

void BackendCoordinator::ReceiveParameters(const BYTE parameterBits, SimulationParameters& parameters)
{
    if (parameterBits == PipeConstants::Marker::ITERMAX) {
        parameters.pressureMaxIterations = pipeManager.ReadInt();
    }
    else {
        REAL parameterValue = pipeManager.ReadReal(); // All of the other possible parameters have the data type REAL, so read the pipe and convert it to a REAL beforehand
        switch (parameterBits) { // AND the start marker with the parameter mask to see which parameter is sent
        case PipeConstants::Marker::WIDTH:
            parameters.width = parameterValue;
            break;
        case PipeConstants::Marker::HEIGHT:
            parameters.height = parameterValue;
            break;
        case PipeConstants::Marker::TAU:
            parameters.timeStepSafetyFactor = parameterValue;
            break;
        case PipeConstants::Marker::OMEGA:
            parameters.relaxationParameter = parameterValue;
            break;
        case PipeConstants::Marker::RMAX:
            parameters.pressureResidualTolerance = parameterValue;
            break;
        case PipeConstants::Marker::REYNOLDS:
            parameters.reynoldsNo = parameterValue;
            break;
        case PipeConstants::Marker::INVEL:
            parameters.inflowVelocity = parameterValue;
            break;
        case PipeConstants::Marker::CHI:
            parameters.surfaceFrictionalPermissibility = parameterValue;
            break;
        case PipeConstants::Marker::MU:
            parameters.dynamicViscosity = parameterValue;
            break;
        case PipeConstants::Marker::DENSITY:
            parameters.fluidDensity = parameterValue;
            break;
        default:
            break;
        }
    }
}

void BackendCoordinator::ReceiveData(BYTE startMarker) {
    if (startMarker == (PipeConstants::Marker::FLDSTART | PipeConstants::Marker::OBST)) { // Obstacles have a separate handler
        ReceiveObstacles();
        solver->ProcessObstacles();
    }
    else if ((startMarker & ~PipeConstants::Marker::PRMMASK) == PipeConstants::Marker::PRMSTART) { // Check if startMarker is a PRMSTART by ANDing it with the inverse of the parameter mask
        BYTE parameterBits = startMarker & PipeConstants::Marker::PRMMASK;
        SimulationParameters parameters = solver->GetParameters();
        ReceiveParameters(parameterBits, parameters);

        if (pipeManager.ReadByte() != (PipeConstants::Marker::PRMEND | parameterBits)) { // Need to receive the corresponding PRMEND
            std::cerr << "Server sent malformed data\n";
            pipeManager.SendByte(PipeConstants::Error::BADREQ);
        }

        solver->SetParameters(parameters);
        pipeManager.SendByte(PipeConstants::Status::OK); // Send an OK to say parameters were received correctly
    }
    else {
        std::cerr << "Server sent unsupported data\n";
        pipeManager.SendByte(PipeConstants::Error::BADREQ); // Error if the start marker was unrecognised.
    }
}

void BackendCoordinator::SetDefaultParameters(SimulationParameters& parameters) {
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
    parameters.inflowVelocity = 1;
    parameters.surfaceFrictionalPermissibility = 0;
    parameters.bodyForces.x = 0;
    parameters.bodyForces.y = 0;
}

BackendCoordinator::BackendCoordinator(int iMax, int jMax, std::string pipeName, Solver* solver)
    : pipeManager(pipeName), solver(solver)
{
    SimulationParameters parameters = SimulationParameters();
    SetDefaultParameters(parameters);
    solver->SetParameters(parameters);
}

int BackendCoordinator::Run() {
    pipeManager.Handshake(solver->GetIMax(), solver->GetJMax());
    std::cout << "Handshake completed ok\n";

    bool closeRequested = false;

    while (!closeRequested) {
        std::cout << "In read loop\n";
        BYTE receivedByte = pipeManager.ReadByte();
        switch (receivedByte & PipeConstants::CATEGORYMASK) { // Gets the category of control byte
        case PipeConstants::Status::GENERIC: // Status bytes
            switch (receivedByte & ~PipeConstants::Status::PARAMMASK) {
            case PipeConstants::Status::HELLO:
            case PipeConstants::Status::BUSY:
            case PipeConstants::Status::OK:
            case PipeConstants::Status::STOP:
                std::cerr << "Server sent a status byte out of sequence, request not understood\n";
                pipeManager.SendByte(PipeConstants::Error::BADREQ);
                break;
            case PipeConstants::Status::CLOSE:
                closeRequested = true;
                pipeManager.SendByte(PipeConstants::Status::OK);
                std::cout << "Backend closing...\n";
                break;
            default:
                std::cerr << "Server sent a malformed status byte, request not understood\n";
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
    return 0;
}