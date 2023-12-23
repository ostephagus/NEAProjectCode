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
	bool** obstacles;
	SimulationParameters parameters;

	void UnflattenArray(bool** pointerArray, bool* flattenedArray, int length, int divisions);
	void PrintFlagsArrows(BYTE** flags, int xLength, int yLength);
	void Timestep(REAL& timestep, const DoubleReal& stepSizes, const DoubleField& velocities, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, const DoubleField& FG, REAL** RHS, REAL** pressure, REAL** streamFunction, int numFluidCells, REAL& pressureResidualNorm);
	void HandleRequest(BYTE requestByte);
	void PerformInitialisation(DoubleField& velocities, REAL**& pressure, REAL**& RHS, REAL**& streamFunction, DoubleField& FG, BYTE**& flags, std::pair<int, int>*& coordinates, int& coordinatesLength, int& numFluidCells, DoubleReal& stepSizes);
	void ReceiveData(BYTE startMarker);
	void SetParameters();

public:
	/// <summary>
	/// Constructor - sets up field dimensions and pipe name.
	/// </summary>
	/// <param name="iMax">The width, in cells, of the simulation domain excluding boundary cells.</param>
	/// <param name="jMax">The height, in cells, of the simulation domain excloding boundary cells.</param>
	/// <param name="pipeName">The name of the named pipe to use for communication with the frontend.</param>
	FrontendManager(int iMax, int jMax, std::string pipeName);

	~FrontendManager();

	/// <summary>
	/// Main method for FrontendManager class, which handles all the data flow and computation.
	/// </summary>
	/// <returns>An exit code to be directly returned by the program</returns>
	int Run();
};

#endif // !FRONTEND_MANAGER_H
