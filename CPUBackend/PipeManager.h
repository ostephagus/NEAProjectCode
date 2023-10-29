#ifndef PIPE_MANAGER_H
#define PIPE_MANAGER_H

#include <windows.h>
#include <string>

class PipeManager {
private:
	HANDLE pipeHandle;
	std::wstring WidenString(std::string input);
	void ReadToNull(BYTE* outBuffer);
	bool Read(BYTE* outBuffer, int bytesToRead);
	BYTE Read();
	void Write(const BYTE* buffer, int bufferLength);
	void Write(BYTE byte);

public:
	PipeManager(std::string pipeName);
	PipeManager(HANDLE pipeHandle);
	~PipeManager();
	bool Handshake(int fieldSize);
	int Handshake();
	bool ReceiveObstacles(bool* obstacles, int fieldLength);
	BYTE ReadByte();
	void SendByte(BYTE byte);
};

#endif // !PIPE_MANAGER_H