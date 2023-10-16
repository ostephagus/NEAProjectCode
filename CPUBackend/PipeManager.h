#ifndef PIPE_MANAGER_H
#define PIPE_MANAGER_H

#include <windows.h>
#include <iostream>

class PipeManager {
private:
	HANDLE pipeHandle;
	void PipeSend(BYTE* buffer, int bufferLength);
	void PipeReceive(BYTE* outBuffer);

public:
	PipeManager(std::string pipeName);
	PipeManager(HANDLE pipeHandle);
};

#endif
