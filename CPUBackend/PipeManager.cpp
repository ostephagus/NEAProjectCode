#include "PipeManager.h"
#include <iostream>

std::wstring PipeManager::WidenString(std::string input) {
	return std::wstring(input.begin(), input.end());
}

void PipeManager::PipeReceive(BYTE* outBuffer) {
	DWORD read = 0; // Number of bytes read in each ReadFile() call
	int index = 0;
	do {
		if (!ReadFile(pipeHandle, outBuffer + index, 1, &read, NULL)) {
			std::cerr << "Failed to read from the named pipe, error code " << GetLastError() << std::endl;
			break;
		}
		index++;
	} while (outBuffer[index - 1] != 0); // Stop if the most recent byte was null-termination.
}

void PipeManager::PipeSend(const BYTE* buffer, int bufferLength)
{
	if (!WriteFile(pipeHandle, buffer, bufferLength, nullptr, NULL)) {
		std::cerr << "Failed to write to the named pipe, error code " << GetLastError() << std::endl;
	}
}

// Constructor for a named pipe, yet to be connected to
PipeManager::PipeManager(std::string pipeName) : pipeHandle(CreateFile(L"\\\\.\\pipe\\TestingPipe", GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL)) {} // Create the handle as a const parameter (one-time initialisation).

// Constructor for if the named pipe has already been connected to
PipeManager::PipeManager(HANDLE existingHandle) : pipeHandle(existingHandle) {} // Pass the handle into the local handle 

PipeManager::~PipeManager() {
	CloseHandle(pipeHandle);
}

void PipeManager::Testing() {
	BYTE* buffer = new BYTE[1024];
	PipeReceive(buffer);
	std::cout << "Received: " << buffer << std::endl;

	const char* toSend = "Hello from c++";
	PipeSend(reinterpret_cast<const BYTE*>(toSend), strlen(toSend));
}