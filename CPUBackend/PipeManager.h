#ifndef PIPE_MANAGER_H
#define PIPE_MANAGER_H

#include "Definitions.h"
#include <windows.h>
#include <string>
#include <utility>

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
	/// <summary>
	/// Constructor to connect to the named pipe
	/// </summary>
	/// <param name="pipeName">The name of the named pipe for communication with the frontend</param>
	PipeManager(std::string pipeName);
	/// <summary>
	/// Constructor accepting an already connected pipe's handle
	/// </summary>
	/// <param name="pipeHandle">The handle of a connected pipe</param>
	PipeManager(HANDLE pipeHandle);
	/// <summary>
	/// Pipe manager destructor - disconnects from the named pipe then closes
	/// </summary>
	~PipeManager();
	/// <summary>
	/// Performs a handshake with the frontend.
	/// </summary>
	/// <returns>
	bool Handshake(int iMax, int jMax);
	std::pair<int, int> Handshake();
	bool ReceiveObstacles(bool* obstacles, int fieldLength);
	BYTE ReadByte();
	void SendByte(BYTE byte);
	void SendField(REAL** field, int xLength, int yLength, int xOffset, int yOffset);
};

#endif // !PIPE_MANAGER_H