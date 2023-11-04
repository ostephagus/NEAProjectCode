#include "PipeManager.h"
#include <iostream>
#include "PipeConstants.h"
#include <algorithm>

#pragma region Private Methods
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

void PipeManager::SerialiseField(BYTE* buffer, REAL** field, int xLength, int yLength, int xOffset, int yOffset) {
	/*
	* The thinking is thus:
	* Each "row" of the field will be stored contiguously
	* The relevant part of these rows will span from (yOffset) to (yOffset + yLength)
	* Therefore each row can be copied directly into the buffer
	* The location in the buffer will have to increment by yLength * sizeof(REAL) each time.
	*/
	for (int i = 0; i < xLength; i++) { // Copy one row at a time (rows are not guaranteed to be contiguously stored)
		std::memcpy(
			buffer + i * yLength * sizeof(REAL), // Start index of destination, buffer + i * column length * 8
			field[i + xOffset] + yOffset, // Start index of source, start index of the column + y offset
			yLength * sizeof(REAL) // Number of bytes to copy, column size * 8
		);
	}
}
#pragma endregion

#pragma region Constructors/Destructors
// Constructor for a named pipe, yet to be connected to
PipeManager::PipeManager(std::string pipeName) {
	pipeHandle = CreateFile(WidenString("\\\\.\\pipe\\" + pipeName).c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
	std::cout << "File opened" << std::endl;
}

// Constructor for if the named pipe has already been connected to
PipeManager::PipeManager(HANDLE existingHandle) : pipeHandle(existingHandle) {} // Pass the handle into the local handle 

PipeManager::~PipeManager() {
	CloseHandle(pipeHandle);
}
#pragma endregion

#pragma region Public Methods
bool PipeManager::Handshake(int iMax, int jMax) {
	BYTE receivedByte = Read();
	if (receivedByte != PipeConstants::Status::HELLO) { // We need a HELLO byte
		std::cerr << "Handshake not completed - server sent malformed request";
		Write(PipeConstants::Error::BADREQ);
		return false;
	}

	
	BYTE buffer[13];
	buffer[0] = PipeConstants::Status::HELLO; // Reply with HELLO byte

	buffer[1] = PipeConstants::Marker::PRMSTART | PipeConstants::Marker::IMAX; // Send iMax, demarked with PRMSTART and PRMEND
	for (int i = 0; i < 4; i++) {
		buffer[i + 2] = iMax >> (i * 8);
	}
	buffer[6] = PipeConstants::Marker::PRMEND | PipeConstants::Marker::IMAX;

	buffer[7] = PipeConstants::Marker::PRMSTART | PipeConstants::Marker::JMAX; // Send jMax, demarked with PRMSTART and PRMEND
	for (int i = 0; i < 4; i++) {
		buffer[i + 8] = jMax >> (i * 8);
	}
	buffer[12] = PipeConstants::Marker::PRMEND | PipeConstants::Marker::JMAX;

	Write(buffer, 13);

	return Read() == PipeConstants::Status::OK; // Success if an OK byte is received
}

std::pair<int,int> PipeManager::Handshake() {
	BYTE receivedByte = Read();

	if (receivedByte != PipeConstants::Status::HELLO) { return std::pair<int, int>(0, 0); } // We need a HELLO byte, (0,0) is the error case

	BYTE buffer[13];
	Read(buffer, 12);

	if (buffer[1] != (PipeConstants::Marker::PRMSTART | PipeConstants::Marker::IMAX)) { return std::pair<int, int>(0, 0); } // Should start with PRMSTART
	int iMax = *reinterpret_cast<int*>(buffer + 2);
	if (buffer[6] != (PipeConstants::Marker::PRMEND | PipeConstants::Marker::IMAX)) { return std::pair<int, int>(0, 0); } // Should end with PRMEND

	if (buffer[7] != (PipeConstants::Marker::PRMSTART | PipeConstants::Marker::JMAX)) { return std::pair<int, int>(0, 0); }
	int jMax = *reinterpret_cast<int*>(buffer + 8);
	if (buffer[12] != (PipeConstants::Marker::PRMEND | PipeConstants::Marker::JMAX)) { return std::pair<int, int>(0, 0); }

	Write(PipeConstants::Status::OK); // Send an OK byte to show the transmission was successful

	return std::pair<int, int>(iMax, jMax);
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

void PipeManager::SendField(REAL** field, int xLength, int yLength, int xOffset, int yOffset)
{
	BYTE* buffer = new BYTE[xLength * yLength * sizeof(REAL)];

	SerialiseField(buffer, field, xLength, yLength, xOffset, yOffset);

	Write(buffer, xLength * yLength * sizeof(REAL));

	delete[] buffer;
}
#pragma endregion