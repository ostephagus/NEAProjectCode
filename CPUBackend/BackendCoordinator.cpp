#include "BackendCoordinator.h"
#include "PipeConstants.h"
#include <iostream>
#include "PipeManager.h"
#include "Computation.h"
#include "Init.h"
#include "Boundary.h"
#define OBSTACLES

void BackendCoordinator::UnflattenArray(bool** pointerArray, bool* flattenedArray, int length, int divisions) {
    for (int i = 0; i < length / divisions; i++) {

        memcpy(
            pointerArray[i],                // Destination address - address at ith pointer
            flattenedArray + i * divisions, // Source start address - move (i * divisions) each iteration
            divisions * sizeof(bool)        // Bytes to copy - divisions
        );

        //pointerArray[i] = flattenedArray + i * divisions;
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

void BackendCoordinator::Timestep(REAL& timestep, const DoubleReal& stepSizes, const DoubleField& velocities, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, const DoubleField& FG, REAL** RHS, REAL** pressure, REAL** streamFunction, int numFluidCells, REAL& pressureResidualNorm)
{
    SetBoundaryConditions(velocities, flags, coordinates, coordinatesLength, iMax, jMax, parameters.inflowVelocity, parameters.surfaceFrictionalPermissibility);
    ComputeTimestep(timestep, iMax, jMax, stepSizes, velocities, parameters.reynoldsNo, parameters.timeStepSafetyFactor);
    REAL gamma = ComputeGamma(velocities, iMax, jMax, timestep, stepSizes);
    ComputeFG(velocities, FG, flags, iMax, jMax, timestep, stepSizes, parameters.bodyForces, gamma, parameters.reynoldsNo);
    ComputeRHS(FG, RHS, flags, iMax, jMax, timestep, stepSizes);
    int pressureIterations = PoissonMultiThreaded(pressure, RHS, flags, coordinates, coordinatesLength, numFluidCells, iMax, jMax, stepSizes, parameters.pressureResidualTolerance, parameters.pressureMinIterations, parameters.pressureMaxIterations, parameters.relaxationParameter, pressureResidualNorm);
    ComputeVelocities(velocities, FG, pressure, flags, iMax, jMax, timestep, stepSizes);
    ComputeStream(velocities, streamFunction, flags, iMax, jMax, stepSizes);
    std::cout << "Pressure iterations: " << pressureIterations << ", residual norm " << pressureResidualNorm << std::endl;
}

void BackendCoordinator::HandleRequest(BYTE requestByte) {
    std::cout << "Starting execution of timestepping loop" << std::endl;
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
        REAL** pressure;
        REAL** RHS;
        REAL** streamFunction;
        DoubleField FG;
        BYTE** flags;
        std::pair<int, int>* coordinates;
        int coordinatesLength;
        int numFluidCells;
        REAL timestep;
        DoubleReal stepSizes;

        PerformInitialisation(velocities, pressure, RHS, streamFunction, FG, flags, coordinates, coordinatesLength, numFluidCells, stepSizes);

        REAL pressureResidualNorm = 0;
        bool stopRequested = false;
        pipeManager.SendByte(PipeConstants::Status::OK); // Send OK to say backend is set up and about to start executing

        int iteration = 0;
        REAL cumulativeTimestep = 0;
        while (!stopRequested) {
            std::cout << "Iteration " << iteration << ", " << cumulativeTimestep << " seconds passed. ";
            pipeManager.SendByte(PipeConstants::Marker::ITERSTART);

            Timestep(timestep, stepSizes, velocities, flags, coordinates, coordinatesLength, FG, RHS, pressure, streamFunction, numFluidCells, pressureResidualNorm);
            cumulativeTimestep += timestep;

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

        FreeMatrix(velocities.x, iMax + 2);
        FreeMatrix(velocities.y, iMax + 2);
        FreeMatrix(pressure, iMax + 2);
        FreeMatrix(RHS, iMax + 2);
        FreeMatrix(FG.x, iMax + 2);
        FreeMatrix(FG.y, iMax + 2);

        std::cout << "Backend stopped." << std::endl;

        pipeManager.SendByte(PipeConstants::Status::OK); // Send OK then stop executing

    }
    else { // Only continuous requests are supported
        std::cerr << "Server sent an unsupported request" << std::endl;
        pipeManager.SendByte(PipeConstants::Error::BADREQ);
    }
}

void BackendCoordinator::PerformInitialisation(DoubleField& velocities, REAL**& pressure, REAL**& RHS, REAL**& streamFunction, DoubleField& FG, BYTE**& flags, std::pair<int, int>*& coordinates, int& coordinatesLength, int& numFluidCells, DoubleReal& stepSizes)
{
    velocities.x = MatrixMAlloc(iMax + 2, jMax + 2);
    velocities.y = MatrixMAlloc(iMax + 2, jMax + 2);

    pressure = MatrixMAlloc(iMax + 2, jMax + 2);
    RHS = MatrixMAlloc(iMax + 2, jMax + 2);
    streamFunction = MatrixMAlloc(iMax + 1, jMax + 1);

    FG.x = MatrixMAlloc(iMax + 2, jMax + 2);
    FG.y = MatrixMAlloc(iMax + 2, jMax + 2);

    flags = FlagMatrixMAlloc(iMax + 2, jMax + 2);
    if (obstacles == nullptr) { // Perform default initialisation if not already initialised
        std::cout << "Initialising obstacles because frontend did not send obstacles" << std::endl;
        obstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);
        for (int i = 1; i <= iMax; i++) { for (int j = 1; j <= jMax; j++) { obstacles[i][j] = 1; } } // Set all the cells to fluid
#ifdef OBSTACLES
        int boundaryLeft = (int)(0.25 * iMax);
        int boundaryRight = (int)(0.35 * iMax);
        int boundaryBottom = (int)(0.45 * jMax);
        int boundaryTop = (int)(0.55 * jMax);

        for (int i = boundaryLeft; i < boundaryRight; i++) { // Create a square of boundary cells
            for (int j = boundaryBottom; j < boundaryTop; j++) {
                obstacles[i][j] = 0;
            }
        }
#endif // OBSTACLES
    }
    /*for (int i = 0; i < iMax + 2; i++) {
        for (int j = 0; j < jMax + 2; j++) {
            std::cout << obstacles[i][j];
        }
        std::cout << std::endl;
    }*/
    //SetObstacles(obstacles);
    SetFlags(obstacles, flags, iMax + 2, jMax + 2);

    //PrintFlagsArrows(flags, iMax + 2, jMax + 2);

    /*std::cout << "Type a character and press enter to continue: ";
    char nonsense;
    std::cin >> nonsense;*/

    std::pair<std::pair<int, int>*, int> coordinatesWithLength = FindBoundaryCells(flags, iMax, jMax);
    coordinates = coordinatesWithLength.first;
    coordinatesLength = coordinatesWithLength.second;

    numFluidCells = CountFluidCells(flags, iMax, jMax);

    stepSizes.x = parameters.width / iMax;
    stepSizes.y = parameters.height / jMax;

    /*for (int i = 0; i <= iMax + 1; i++) {
        for (int j = 0; j <= jMax + 1; j++) {
            pressure[i][j] = 1000;
        }
    }*/
}

void BackendCoordinator::ReceiveData(BYTE startMarker) {
    if (startMarker == (PipeConstants::Marker::FLDSTART | PipeConstants::Marker::OBST)) { // Only supported fixed-length field is obstacles
        bool* obstaclesFlattened = new bool[(iMax + 2) * (jMax + 2)]();
        pipeManager.ReceiveObstacles(obstaclesFlattened, iMax + 2, jMax + 2);
        obstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);
        UnflattenArray(obstacles, obstaclesFlattened, (iMax + 2) * (jMax + 2), jMax + 2);
        //std::cout << "Address of obstacles after initialisation: " << obstacles << std::endl;
        /*for (int i = 0; i < iMax + 2; i++) {
            for (int j = 0; j < jMax + 2; j++) {
                std::cout << obstacles[i][j];
            }
            std::cout << std::endl;
        }*/
        delete[] obstaclesFlattened;
    }
    else if ((startMarker & ~PipeConstants::Marker::PRMMASK) == PipeConstants::Marker::PRMSTART) { // Check if startMarker is a PRMSTART by ANDing it with the inverse of the parameter mask
        BYTE parameterBits = startMarker & PipeConstants::Marker::PRMMASK;
        if (parameterBits == PipeConstants::Marker::ITERMAX) {
            parameters.pressureMaxIterations = pipeManager.ReadInt();
            std::cout << "IterMax changed" << std::endl;
        }
        else {
            REAL parameterValue = pipeManager.ReadReal(); // All of the other possible parameters have the data type REAL, so read the pipe and convert it to a REAL beforehand
            bool wasNotDefault = true; // DEBUGGING
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
                wasNotDefault = false; // DEBUGGING
                break;
            }
            if (wasNotDefault) {
                std::cout << "Parameter changed" << std::endl;
            }
        }
        if (pipeManager.ReadByte() != (PipeConstants::Marker::PRMEND | parameterBits)) { // Need to receive the corresponding PRMEND
            std::cerr << "Server sent malformed data" << std::endl;
            pipeManager.SendByte(PipeConstants::Error::BADREQ);
        }
        pipeManager.SendByte(PipeConstants::Status::OK); // Send an OK to say parameters were received correctly
    }
    else {
        std::cerr << "Server sent unsupported data" << std::endl;
        pipeManager.SendByte(PipeConstants::Error::BADREQ); // All others not supported at the moment
    }
}

void BackendCoordinator::SetParameters() {
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
    : iMax(iMax), jMax(jMax), fieldSize(iMax * jMax), pipeManager(pipeName), obstacles(nullptr) // Set obstacles to null pointer to represent unallcoated
{
    parameters = SimulationParameters();
    SetParameters();
}

BackendCoordinator::~BackendCoordinator() {
    FreeMatrix(obstacles, iMax + 2);
}

int BackendCoordinator::Run() {
    pipeManager.Handshake(iMax, jMax);
    std::cout << "Handshake completed ok" << std::endl;
    
    std::cout << "Enter a character and press enter: ";
    char nonsense;
    std::cin >> nonsense;

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