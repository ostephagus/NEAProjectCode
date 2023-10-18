#ifndef PIPE_MANAGER_H
#define PIPE_MANAGER_H

#include <windows.h>
#include <string>

class PipeManager {
private:
	const HANDLE pipeHandle;
	std::wstring WidenString(std::string input);
	void PipeReceive(BYTE* outBuffer);
	void PipeSend(const BYTE* buffer, int bufferLength);

public:
	PipeManager(std::string pipeName);
	PipeManager(HANDLE pipeHandle);
	~PipeManager();
	void Testing();
};

#endif // !PIPE_MANAGER_H