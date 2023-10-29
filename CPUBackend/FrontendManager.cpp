#include "FrontendManager.h"
#include "PipeConstants.h"
#include <iostream>
#include "PipeManager.h"

void FrontendManager::HandleRequest(BYTE requestByte) {

}

void FrontendManager::ReceiveData(BYTE startMarker) {
    if (startMarker == (PipeConstants::Marker::FLDSTART | PipeConstants::Marker::OBST)) {

    }
}

FrontendManager::FrontendManager(int iMax, int jMax, std::string pipeName)
    : iMax(iMax), jMax(jMax), fieldSize(iMax * jMax), pipeManager(pipeName)
{}

int FrontendManager::Run() {
    pipeManager.Handshake(fieldSize);

    //bool* obstaclesFlattened = new bool[fieldSize];
    //bool** obstacles = ObstacleMatrixMAlloc(iMax, jMax);
    //UnflattenArray(obstacles, obstaclesFlattened, fieldSize, jMax);
    // Obstacles not working at the moment so don't use obstacles

    bool stopRequested = false;

    while (!stopRequested) {
        BYTE receivedByte = pipeManager.ReadByte();
        switch (receivedByte & PipeConstants::CATEGORYMASK) {
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
                stopRequested = true;
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
        default:
            break;
        }
    }

    //delete[] obstaclesFlattened;
    return 0;
}