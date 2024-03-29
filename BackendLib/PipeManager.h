#ifndef PIPE_MANAGER_H
#define PIPE_MANAGER_H

#include "pch.h"
#include <windows.h>
#include <string>

class PipeManager
{
private:
	HANDLE pipeHandle;
	std::wstring WidenString(std::string input);
	void ReadToNull(BYTE* outBuffer);
	bool Read(BYTE* outBuffer, int bytesToRead);
	BYTE Read();
	void Write(const BYTE* buffer, DWORD bufferLength);
	void Write(BYTE byte);

	/// <summary>
	/// A method to convert a 2D array of REALs (field) into a flat array of BYTEs for transmission over the pipe.
	/// </summary>
	/// <param name="buffer">An array of BYTEs, with length <c>sizeof(REAL) * fieldSize</c>.</param>
	/// <param name="field">The 2D array of REALs to serialise.</param>
	/// <param name="xLength">The number of REALs to serialise in the x direction.</param>
	/// <param name="yLength">The number of REALs to serialise in the y direction.</param>
	/// <param name="xOffset">The x-index of the first REAL to be serialised.</param>
	/// <param name="yOffset">The y-index of the first REAL to be serialised.</param>
	void SerialiseField(BYTE* buffer, REAL** field, int xLength, int yLength, int xOffset, int yOffset);

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
	/// <returns>A <c>bool</c> indicating whether the handshake completed successfully.</returns>
	bool Handshake(int iMax, int jMax);

	/// <summary>
	/// Performs a handshake with the frontend.
	/// </summary>
	/// <returns>A std::pair, with the values of iMax and jMax (the simulation domain's dimensions).</returns>
	std::pair<int, int> Handshake();

	/// <summary>
	/// A subroutine to receive obstacles through the pipe, and convert them to a bool array.
	/// </summary>
	/// <param name="obstacles">The obstacles array to output to.</param>
	/// <param name="xLength">The number of cells in the x direction</param>
	/// <param name="yLength">The number of cells in the y direction</param>
	/// <returns>A <c>bool</c> indicating whether the action was successful.</returns>
	bool ReceiveObstacles(bool* obstacles, int xLength, int yLength);

	/// <summary>
	/// Reads a byte from the pipe, and returns it
	/// </summary>
	/// <returns>The single byte read from the pipe</returns>
	BYTE ReadByte();

	/// <summary>
	/// Writes a single byte to the pipe
	/// </summary>
	/// <param name="byte">The byte to write</param>
	void SendByte(BYTE byte);

	/// <summary>
	/// Reads a <c>REAL</c> data type from the pipe, assuming one has been sent.
	/// </summary>
	/// <returns>The converted <c>REAL</c> read from the pipe.</returns>
	REAL ReadReal();

	/// <summary>
	/// Sends one REAL datum down the pipe.
	/// </summary>
	/// <param name="data">The REAL to send down the pipe.</param>
	void SendReal(REAL data);

	/// <summary>
	/// Reads a <c>int</c> data type from the pipe, assuming one has been sent.
	/// </summary>
	/// <returns>The converted <c>int</c> read from the pipe.</returns>
	int ReadInt();

	/// <summary>
	/// Sends the contents of a field through the pipe.
	/// </summary>
	/// <param name="field">An array of pointers to the rows of the field.</param>
	/// <param name="xLength">The length in the x direction that will be transmitted.</param>
	/// <param name="yLength">The length in the y direction that will be transmitted.</param>
	/// <param name="xOffset">The x-index of the first value to be transmitted.</param>
	/// <param name="yOffset">The y-index of the first value to be transmitted.</param>
	void SendField(REAL** field, int xLength, int yLength, int xOffset, int yOffset);

	/// <summary>
	/// Sends the contents of a field through the pipe.
	/// </summary>
	/// <param name="field">The field to transmit as a flattened array.</param>
	/// <param name="numElements">The number of elements in the field, <c>height * width</c>.</param>
	void SendField(REAL* field, int numElements);
};

#endif // !PIPE_MANAGER_H