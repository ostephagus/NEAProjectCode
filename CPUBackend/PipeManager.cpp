#include "PipeManager.h"

class PipeManager
{
private:
	HANDLE pipeHandle;

	void PipeSend(BYTE* buffer, int bufferLength) {
		WriteFile(pipeHandle, buffer, bufferLength, nullptr, NULL);
	}

public:

	PipeManager(std::string pipeName) {
		std::string fullName = "\\\\.\\pipe\\" + pipeName;
		std::wstring WFullName = std::wstring(fullName.begin(), fullName.end()); // Convert to a "wide string"
		pipeHandle = CreateFileW(WFullName.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL); // Create a handle for the pipe, with 2-way access
	}
	PipeManager(HANDLE pipeHandle) {
		this->pipeHandle = pipeHandle; // Pass the parameter into the pipeHandle field
	}

};