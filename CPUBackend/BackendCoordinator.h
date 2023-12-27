#ifndef BACKEND_COORDINATOR_H
#define BACKEND_COORDINATOR_H

#include "Definitions.h"
#include "PipeManager.h"
#include "Solver.h"

class BackendCoordinator
{
private:
	PipeManager pipeManager;
	Solver* solver;

	void UnflattenArray(bool** pointerArray, bool* flattenedArray, int length, int divisions);
	void PrintFlagsArrows(BYTE** flags, int xLength, int yLength);
	void HandleRequest(BYTE requestByte);
	void ReceiveObstacles();
	void ReceiveParameters(const BYTE parameterBits, SimulationParameters& parameters);
	void ReceiveData(BYTE startMarker);
	void SetDefaultParameters(SimulationParameters& parameters);

public:
	/// <summary>
	/// Constructor - sets up field dimensions and pipe name.
	/// </summary>
	/// <param name="iMax">The width, in cells, of the simulation domain excluding boundary cells.</param>
	/// <param name="jMax">The height, in cells, of the simulation domain excluding boundary cells.</param>
	/// <param name="pipeName">The name of the named pipe to use for communication with the frontend.</param>
	BackendCoordinator(int iMax, int jMax, std::string pipeName);

	/// <summary>
	/// Main method for BackendCoordinator class, which handles all the data flow and computation.
	/// </summary>
	/// <returns>An exit code to be directly returned by the program</returns>
	int Run();
};

#endif // !BACKEND_COORDINATOR_H
