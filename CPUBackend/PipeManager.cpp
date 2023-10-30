#include "PipeManager.h"
#include <iostream>
#include "PipeConstants.h"

std::wstring PipeManager::WidenString(std::string input) {
	return std::wstring(input.begin(), input.end());
}

void PipeManager::ReadToNull(BYTE* outBuffer) {
	DWORD read = 0; // Number of bytes read in each ReadFile() call
	int index = 0;
	do {
		if (!ReadFile(pipeHandle, outBuffer + index, 1, &read, NULL)) {
			std::cerr << "Failed to read from the named pipe, error code " << GetLastError() << std::endl;
			break;
		}
		index++;
	} while (outBuffer[index - 1] != 0); // Stop if the most recent byte was null-termination
}

bool PipeManager::Read(BYTE* outBuffer, int bytesToRead) {
	DWORD bytesRead;
	return ReadFile(pipeHandle, outBuffer, bytesToRead, &bytesRead, NULL) && bytesRead == bytesToRead; // Success if bytes were read, and enough bytes were read
}

BYTE PipeManager::Read() {
	BYTE outputByte;
	if (!ReadFile(pipeHandle, &outputByte, 1, nullptr, NULL)) {
		std::cerr << "Failed to read from the named pipe, error code " << GetLastError() << std::endl;
	}
	return outputByte;
}

void PipeManager::Write(const BYTE* buffer, int bufferLength)
{
	if (!WriteFile(pipeHandle, buffer, bufferLength, nullptr, NULL)) {
		std::cerr << "Failed to write to the named pipe, error code " << GetLastError() << std::endl;
	}
}

void PipeManager::Write(BYTE byte) {
	if (!WriteFile(pipeHandle, &byte, 1, nullptr, NULL)) {
		std::cerr << "Failed to write to the named pipe, error code " << GetLastError() << std::endl;
	}
}

// Constructor for a named pipe, yet to be connected to
PipeManager::PipeManager(std::string pipeName) {
	pipeHandle = CreateFile(WidenString("\\\\.\\pipe\\" + pipeName).c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
}

// Constructor for if the named pipe has already been connected to
PipeManager::PipeManager(HANDLE existingHandle) : pipeHandle(existingHandle) {} // Pass the handle into the local handle 

PipeManager::~PipeManager() {
	CloseHandle(pipeHandle);
}

bool PipeManager::Handshake(int fieldSize) {
	BYTE receivedByte = Read();
	if ((receivedByte & PipeConstants::Status::GENERIC) != PipeConstants::Status::HELLO) { // We need a HELLO byte
		std::cerr << "Handshake not completed - server sent malformed request";
		Write(PipeConstants::Error::BADREQ);
		return false;
	}
	if ((receivedByte & PipeConstants::Status::PARAMMASK) != 0) { // With no parameter (0 for x bits)
		std::cerr << "Handshake not completed - server sent malformed request";
		Write(PipeConstants::Error::BADPARAM);
		return false;
	}
	
	BYTE buffer[5];
	buffer[0] = PipeConstants::Status::HELLO | 4; // HELLO byte with next 4 bytes as parameter
	buffer[1] = *reinterpret_cast<BYTE*>(&fieldSize); // Reinterpret the uint into bytes and add it to the buffer (strange intentional buffer overflow) -- NEEDS TESTING
	Write(buffer, 5);
	return true;
}

int PipeManager::Handshake() {
	BYTE receivedByte = Read();

	if ((receivedByte & PipeConstants::Status::GENERIC) != PipeConstants::Status::HELLO) { // We need a HELLO byte
		std::cerr << "Handshake not completed - server sent malformed request";
		Write(PipeConstants::Error::BADREQ);
		return false;
	}
	if ((receivedByte & PipeConstants::Status::PARAMMASK) != 4) { // With a parameter of length 4
		std::cerr << "Handshake not completed - server sent malformed request";
		Write(PipeConstants::Error::BADPARAM);
		return false;
	}

	BYTE buffer[4];
	Read(buffer, 4);
	return *reinterpret_cast<int*>(buffer); // Reinterpret the received bytes into an int
}

bool PipeManager::ReceiveObstacles(bool* obstacles, int fieldLength) {
	int bufferLength = fieldLength / 8 + (fieldLength % 8 == 0 ? 0 : 1);
	BYTE* buffer = new BYTE[bufferLength]; // Have to use new keyword because length of array is not a constant expression

	// Assume there has been a FLDSTART before

	Read(buffer, (int)bufferLength); // Read the bool obstacle data

	int byteNumber = 0;
	for (int i = 0; i < fieldLength; i++) {
		if (byteNumber <= fieldLength / 8) { // If there are more than a byte's worth of data to read, shift all 8 bits of the byte
			obstacles[byteNumber * 8 + 7 - (i % 8)] = ((buffer[byteNumber] >> i) == 0) ? false : true; // Due to the way bits are shifted into the bytes by the server, they must be shifted off in the opposite order hence the complicated expression for obstacles[...]
		}
		else {
			int remainingBits = fieldLength - byteNumber * 8; // This needs testing - I was very tired when I wrote this
			obstacles[byteNumber * 8 + (remainingBits - 1) - (i % 8)] = ((buffer[byteNumber] >> i) == 0) ? false : true;
		}

		if (i % 8 == 7) {
			byteNumber++;
		}
	}

	if (Read() != (PipeConstants::Marker::FLDEND | PipeConstants::Marker::OBST)) { // Ensure there is a FLDEND after
		std::cerr << "Cannot read obstacles - server sent malformed data";
		Write(PipeConstants::Error::BADPARAM);
		return false;
	}

	delete[] buffer;
	Write(PipeConstants::Status::OK); // Send an OK message to server to tell it the data was understood
	return true;
}

BYTE PipeManager::ReadByte() {
	return Read();
}

void PipeManager::SendByte(BYTE byte) {
	Write(byte);
}
