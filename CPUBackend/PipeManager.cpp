#include "PipeManager.h"


PipeManager::PipeManager(std::string pipeName) {
	std::string fullName = "\\\\.\\pipe\\" + pipeName;
	std::wstring WFullName = std::wstring(fullName.begin(), fullName.end()); // Convert to a "wide string"
	pipeHandle = CreateFileW(WFullName.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL); // Create a handle for the pipe, with 2-way access
}
PipeManager::PipeManager(HANDLE pipeHandle) {
	this->pipeHandle = pipeHandle; // Pass the parameter into the pipeHandle field
}
void PipeManager::PipeReceive(BYTE* outBuffer) {
	ULONG read = 0; // Read as past tense - this is number of bytes read already
	int index = 0;
	do {
		ReadFile(pipeHandle, outBuffer + index++, 1, &read, NULL);
	} while (read > 0 && *(outBuffer + index - 1) != 0);
}

void PipeManager::PipeSend(BYTE* buffer, int bufferLength)
{
	WriteFile(pipeHandle, buffer, bufferLength, nullptr, NULL);
}

void Testing() {

}