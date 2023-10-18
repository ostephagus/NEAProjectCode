#include "PipeManager.h"
#include <iostream>

void PipeManager::PipeConnect(LPCWSTR pipeFullName) {
	const int ATTEMPT_LIMIT = 10;
	int attempts = 0;
	
	do {
		if (attempts > ATTEMPT_LIMIT) {
			break;
		}
		pipeHandle = CreateFile(pipeFullName, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
		if (pipeHandle == INVALID_HANDLE_VALUE) {
			std::cerr << "Failed to connect to the named pipe, error code " << GetLastError() << ". Trying again." << std::endl;
			attempts++;
		}
	} while (pipeHandle == INVALID_HANDLE_VALUE);
}

void PipeManager::PipeReceive(BYTE* outBuffer) {
	DWORD read = 0; // Number of bytes read in each ReadFile() call
	int index = 0;
	do {
		if (!ReadFile(pipeHandle, outBuffer + index, 1, &read, NULL)) {
			std::cerr << "Failed to read from the named pipe, error code " << GetLastError() << std::endl;
			break;
		}
	} while (outBuffer[index - 1] != 0); // Stop if the most recent byte was null-termination.
}

void PipeManager::PipeSend(BYTE* buffer, int bufferLength)
{
	if (!WriteFile(pipeHandle, buffer, bufferLength, nullptr, NULL)) {
		std::cerr << "Failed to write to the named pipe, error code " << GetLastError() << std::endl;
	}
}

void PipeManager::PipeSend(const BYTE* buffer, int bufferLength)
{
	if (!WriteFile(pipeHandle, buffer, bufferLength, nullptr, NULL)) {
		std::cerr << "Failed to write to the named pipe, error code " << GetLastError() << std::endl;
	}
}

// Constructor for a named pipe, yet to be connected to
PipeManager::PipeManager(std::string pipeName) {
	std::string fullName = "\\\\.\\pipe\\" + pipeName;
	PipeConnect(std::wstring(fullName.begin(), fullName.end()).c_str());
}

// Constructor for if the named pipe has already been connected to
PipeManager::PipeManager(HANDLE pipeHandle) {
	this->pipeHandle = pipeHandle; // Pass the parameter into the pipeHandle field
}


void PipeManager::Testing() {
	BYTE* buffer = new BYTE[100];
	memset(buffer, 0, 100); //Set all the bytes to 0x00
	PipeReceive(buffer);
	std::cout << "Received: " << buffer << std::endl;

	const char* toSend = "Hello from c++";
	PipeSend(reinterpret_cast<const BYTE*>(toSend), strlen(toSend));
}