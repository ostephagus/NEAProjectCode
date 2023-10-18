#ifndef PIPE_MANAGER_H
#define PIPE_MANAGER_H

#include <windows.h>
#include <string>

class PipeManager {
private:
	HANDLE pipeHandle;
	void PipeConnect(LPCWSTR pipeFullName);
	void PipeReceive(BYTE* outBuffer);
	void PipeSend(BYTE* buffer, int bufferLength);
	void PipeSend(const BYTE* buffer, int bufferLength);

public:
	PipeManager(std::string pipeName);
	PipeManager(HANDLE pipeHandle);
	void Testing();
};

#endif
