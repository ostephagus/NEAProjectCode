#ifndef FRONTEND_MANAGER_H
#define FRONTEND_MANAGER_H

#include "Definitions.h"
#include "PipeManager.h"

class FrontendManager
{
private:
	const int iMax;
	const int jMax;
	const int fieldSize;
	PipeManager pipeManager;

	void UnflattenArray(bool** pointerArray, bool* flattenedArray, int length, int divisions);
	void SetupParameters(REAL** horizontalVelocity, REAL** verticalVelocity, REAL** pressure, SimulationParameters& parameters, DoubleReal stepSizes);
	void TimeStep(DoubleField velocities, DoubleField FG, REAL** pressure, REAL** nextPressure, REAL** RHS, REAL** streamFunction, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int numFluidCells, const SimulationParameters& parameters, DoubleReal stepSizes);
	void HandleRequest(BYTE requestByte);
	void ReceiveData(BYTE startMarker);

public:
	/// <summary>
	/// Constructor - sets up field dimensions and pipe name.
	/// </summary>
	/// <param name="iMax">The width, in cells, of the simulation domain excluding boundary cells.</param>
	/// <param name="jMax">The height, in cells, of the simulation domain excloding boundary cells.</param>
	/// <param name="pipeName">The name of the named pipe to use for communication with the frontend.</param>
	FrontendManager(int iMax, int jMax, std::string pipeName);
	/// <summary>
	/// Main method for FrontendManager class, which handles all the data flow and computation.
	/// </summary>
	/// <returns>An exit code to be directly returned by the program</returns>
	int Run();
};

#endif // !FRONTEND_MANAGER_H
