#include "BackendCoordinator.h"
#include "PipeConstants.h"
#include <iostream>
#include "PipeManager.h"
#include "Computation.h"
#include "Init.h"
#include "Boundary.h"
#include "CPUSolver.h"
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

void BackendCoordinator::PrintFlagsArrows(BYTE** flags, int xLength, int yLength) {
    for (int i = xLength - 1; i >= 0; i--) {
        for (int j = 0; j < yLength; j++) {
            switch (flags[j][i]) {
            case B_N:
                std::cout << "^^";
                break;
            case B_NE:
                std::cout << "^>";
                break;
            case B_E:
                std::cout << ">>";
                break;
            case B_SE:
                std::cout << "v>";
                break;
            case B_S:
                std::cout << "vv";
                break;
            case B_SW:
                std::cout << "<v";
                break;
            case B_W:
                std::cout << "<<";
                break;
            case B_NW:
                std::cout << "<^";
                break;
            case OBS:
                std::cout << "()";
                break;
            case FLUID:
                std::cout << "  ";
                break;
            default:
                std::cout << "  ";
                break;
            }
        }
        std::cout << std::endl;
    }
}

void BackendCoordinator::HandleRequest(BYTE requestByte) {
    std::cout << "Starting execution of timestepping loop" << std::endl;
    if ((requestByte & ~PipeConstants::Request::PARAMMASK) == PipeConstants::Request::CONTREQ) {
        if (requestByte == PipeConstants::Request::CONTREQ) {
            pipeManager.SendByte(PipeConstants::Error::BADPARAM);
            std::cerr << "Server sent a blank request, exiting";
            return;
        }

        solver->ProcessObstacles();

        bool hVelWanted = requestByte & PipeConstants::Request::HVEL;
        bool vVelWanted = requestByte & PipeConstants::Request::VVEL;
        bool pressureWanted = requestByte & PipeConstants::Request::PRES;
        bool streamWanted = requestByte & PipeConstants::Request::STRM;

        bool stopRequested = false;
        pipeManager.SendByte(PipeConstants::Status::OK); // Send OK to say backend is set up and about to start executing

        int iteration = 0;
        REAL cumulativeTimestep = 0;
        while (!stopRequested) {
            std::cout << "Iteration " << iteration << ", " << cumulativeTimestep << " seconds passed. " << std::endl;
            pipeManager.SendByte(PipeConstants::Marker::ITERSTART);

            solver->Timestep(cumulativeTimestep);

            int iMax = solver->GetIMax();
            int jMax = solver->GetJMax();

            if (hVelWanted) {
                pipeManager.SendByte(PipeConstants::Marker::FLDSTART | PipeConstants::Marker::HVEL);
                pipeManager.SendField(solver->GetHorizontalVelocity(), iMax, jMax, 1, 1); // Set the offsets to 1 and the length to iMax / jMax to exclude boundary cells at cells 0 and max
                pipeManager.SendByte(PipeConstants::Marker::FLDEND | PipeConstants::Marker::HVEL);
            }
            if (vVelWanted) {
                pipeManager.SendByte(PipeConstants::Marker::FLDSTART | PipeConstants::Marker::VVEL);
                pipeManager.SendField(solver->GetVerticalVelocity(), iMax, jMax, 1, 1);
                pipeManager.SendByte(PipeConstants::Marker::FLDEND | PipeConstants::Marker::VVEL);
            }
            if (pressureWanted) {
                pipeManager.SendByte(PipeConstants::Marker::FLDSTART | PipeConstants::Marker::PRES);
                pipeManager.SendField(solver->GetPressure(), iMax, jMax, 1, 1);
                pipeManager.SendByte(PipeConstants::Marker::FLDEND | PipeConstants::Marker::PRES);
            }
            if (streamWanted) {
                pipeManager.SendByte(PipeConstants::Marker::FLDSTART | PipeConstants::Marker::STRM);
                pipeManager.SendField(solver->GetStreamFunction(), iMax, jMax, 0, 0); // Stream function does not include boundary cells
                pipeManager.SendByte(PipeConstants::Marker::FLDEND | PipeConstants::Marker::STRM);
            }

            pipeManager.SendByte(PipeConstants::Marker::ITEREND);

            BYTE receivedByte = pipeManager.ReadByte();
            if (receivedByte == PipeConstants::Status::STOP || receivedByte == PipeConstants::Error::INTERNAL) { 
                stopRequested = true; // Stop if requested or the frontend fatally errors
            }
            else { // If stop was requested, skip parameter reading.
                while ((receivedByte & ~PipeConstants::Marker::PRMMASK) == PipeConstants::Marker::PRMSTART) { // While the received byte is a PRMSTART
                    ReceiveData(receivedByte);
                    receivedByte = pipeManager.ReadByte(); // And read the next byte
                }
                if (receivedByte != PipeConstants::Status::OK) { // Require an OK at the end, whether parameters were sent or not
                    std::cerr << "Server sent malformed data" << std::endl;
                    pipeManager.SendByte(PipeConstants::Error::BADREQ);
                }
            }

            iteration++;
        }
        std::cout << "Backend stopped." << std::endl;

        pipeManager.SendByte(PipeConstants::Status::OK); // Send OK then stop executing

    }
    else { // Only continuous requests are supported
        std::cerr << "Server sent an unsupported request" << std::endl;
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
        std::cout << "IterMax changed" << std::endl;
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
        default:
            break;
        }
    }
}

void BackendCoordinator::ReceiveData(BYTE startMarker) {
    if (startMarker == (PipeConstants::Marker::FLDSTART | PipeConstants::Marker::OBST)) { // Only supported fixed-length field is obstacles
        ReceiveObstacles();
    }
    else if ((startMarker & ~PipeConstants::Marker::PRMMASK) == PipeConstants::Marker::PRMSTART) { // Check if startMarker is a PRMSTART by ANDing it with the inverse of the parameter mask
        BYTE parameterBits = startMarker & PipeConstants::Marker::PRMMASK;
        SimulationParameters parameters = solver->GetParameters();
        ReceiveParameters(parameterBits, parameters);

        if (pipeManager.ReadByte() != (PipeConstants::Marker::PRMEND | parameterBits)) { // Need to receive the corresponding PRMEND
            std::cerr << "Server sent malformed data" << std::endl;
            pipeManager.SendByte(PipeConstants::Error::BADREQ);
        }

        solver->SetParameters(parameters);
        pipeManager.SendByte(PipeConstants::Status::OK); // Send an OK to say parameters were received correctly
    }
    else {
        std::cerr << "Server sent unsupported data" << std::endl;
        pipeManager.SendByte(PipeConstants::Error::BADREQ); // All others not supported at the moment
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
    parameters.reynoldsNo = 1000;
    parameters.inflowVelocity = 1;
    parameters.surfaceFrictionalPermissibility = 0;
    parameters.bodyForces.x = 0;
    parameters.bodyForces.y = 0;
}

BackendCoordinator::BackendCoordinator(int iMax, int jMax, std::string pipeName)
    : pipeManager(pipeName)
{
    SimulationParameters parameters = SimulationParameters();
    SetDefaultParameters(parameters);
    solver = new CPUSolver(parameters, iMax, jMax);
}

int BackendCoordinator::Run() {
    pipeManager.Handshake(solver->GetIMax(), solver->GetJMax());
    std::cout << "Handshake completed ok" << std::endl;
    
    /*std::cout << "Enter a character and press enter: ";
    char nonsense;
    std::cin >> nonsense;*/

    bool closeRequested = false;

    while (!closeRequested) {
        std::cout << "In read loop" << std::endl;
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
                pipeManager.SendByte(PipeConstants::Status::OK);
                delete solver;
                std::cout << "Backend closing..." << std::endl;
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
    return 0;
}