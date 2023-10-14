#ifndef PIPE_MANAGER_H
#define PIPE_MANAGER_H

#include <windows.h>
#include <iostream>

class PipeManager {
private:
	HANDLE pipeHandle;

public:
	PipeManager(std::string pipeName);
	PipeManager(HANDLE pipeHandle);
};

#endif
