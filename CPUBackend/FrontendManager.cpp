#include "FrontendManager.h"
#include "PipeConstants.h"
#include <iostream>
#include "PipeManager.h"
#include "Computation.h"
#include "Init.h"
#include "Boundary.h"
#define OBSTACLES

void FrontendManager::UnflattenArray(bool** pointerArray, bool* flattenedArray, int length, int divisions) {
	for (int i = 0; i < length / divisions; i++) {
		pointerArray[i] = flattenedArray + i * divisions;
	}
}

void FrontendManager::Timestep(REAL& timestep, const DoubleReal& stepSizes, const DoubleField& velocities, SimulationParameters& parameters, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, const DoubleField& FG, REAL** RHS, REAL** pressure, REAL** streamFunction, int numFluidCells, REAL& pressureResidualNorm)
{
	SetBoundaryConditions(velocities, flags, coordinates, coordinatesLength, iMax, jMax, parameters.inflowVelocity, parameters.surfaceFrictionalPermissibility);
	ComputeTimestep(timestep, iMax, jMax, stepSizes, velocities, parameters.reynoldsNo, parameters.timeStepSafetyFactor);
	REAL gamma = ComputeGamma(velocities, iMax, jMax, timestep, stepSizes);
	ComputeFG(velocities, FG, flags, iMax, jMax, timestep, stepSizes, parameters.bodyForces, gamma, parameters.reynoldsNo);
	ComputeRHS(FG, RHS, flags, iMax, jMax, timestep, stepSizes);
	int pressureIterations = PoissonMultiThreaded(pressure, RHS, flags, coordinates, coordinatesLength, numFluidCells, iMax, jMax, stepSizes, parameters.pressureResidualTolerance, parameters.pressureMinIterations, parameters.pressureMaxIterations, parameters.relaxationParameter, pressureResidualNorm);
	ComputeVelocities(velocities, FG, pressure, flags, iMax, jMax, timestep, stepSizes);
	ComputeStream(velocities, streamFunction, flags, iMax, jMax, stepSizes);
	std::cout << "Pressure iterations: " << pressureIterations << ", residual norm " << pressureResidualNorm << std::endl;
}

void FrontendManager::HandleRequest(BYTE requestByte) {
	std::cout << "Starting execution of timestepping loop" << std::endl;
	if ((requestByte & ~PipeConstants::Request::PARAMMASK) == PipeConstants::Request::CONTREQ) {
		if (requestByte == PipeConstants::Request::CONTREQ) {
			pipeManager.SendByte(PipeConstants::Error::BADPARAM);
			std::cerr << "Server sent a blank request, exiting";
			return;
		}
		bool hVelWanted = requestByte & PipeConstants::Request::HVEL;
		bool vVelWanted = requestByte & PipeConstants::Request::VVEL;
		bool pressureWanted = requestByte & PipeConstants::Request::PRES;
		bool streamWanted = requestByte & PipeConstants::Request::STRM;

		SimulationParameters parameters;
		DoubleField velocities;
		REAL** pressure;
		REAL** RHS;
		REAL** streamFunction;
		DoubleField FG;
		BYTE** flags;
		std::pair<int, int>* coordinates;
		int coordinatesLength;
		int numFluidCells;
		REAL timestep;
		DoubleReal stepSizes;

		SetParameters(velocities, pressure, RHS, streamFunction, FG, flags, obstacles, coordinates, coordinatesLength, numFluidCells, parameters, stepSizes);

		REAL pressureResidualNorm = 0;
		bool stopRequested = false;
		pipeManager.SendByte(PipeConstants::Status::OK); // Send OK to say backend is set up and about to start executing

		int iteration = 0;
		REAL cumulativeTimestep = 0;
		while (!stopRequested) {
			std::cout << "Iteration " << iteration << ", " << cumulativeTimestep << " seconds passed. ";
			pipeManager.SendByte(PipeConstants::Marker::ITERSTART);

			Timestep(timestep, stepSizes, velocities, parameters, flags, coordinates, coordinatesLength, FG, RHS, pressure, streamFunction, numFluidCells, pressureResidualNorm);
			cumulativeTimestep += timestep;

			if (hVelWanted) {
				pipeManager.SendByte(PipeConstants::Marker::FLDSTART | PipeConstants::Marker::HVEL);
				pipeManager.SendField(velocities.x, iMax, jMax, 1, 1); // Set the offsets to 1 and the length to iMax / jMax to exclude boundary cells at cells 0 and max
				pipeManager.SendByte(PipeConstants::Marker::FLDEND | PipeConstants::Marker::HVEL);
			}
			if (vVelWanted) {
				pipeManager.SendByte(PipeConstants::Marker::FLDSTART | PipeConstants::Marker::VVEL);
				pipeManager.SendField(velocities.y, iMax, jMax, 1, 1);
				pipeManager.SendByte(PipeConstants::Marker::FLDEND | PipeConstants::Marker::VVEL);
			}
			if (pressureWanted) {
				pipeManager.SendByte(PipeConstants::Marker::FLDSTART | PipeConstants::Marker::PRES);
				pipeManager.SendField(pressure, iMax, jMax, 1, 1);
				pipeManager.SendByte(PipeConstants::Marker::FLDEND | PipeConstants::Marker::PRES);
			}
			if (streamWanted) {
				pipeManager.SendByte(PipeConstants::Marker::FLDSTART | PipeConstants::Marker::STRM);
				pipeManager.SendField(streamFunction, iMax, jMax, 0, 0);
				pipeManager.SendByte(PipeConstants::Marker::FLDEND | PipeConstants::Marker::STRM);
			}

			pipeManager.SendByte(PipeConstants::Marker::ITEREND);

			BYTE receivedByte = pipeManager.ReadByte();
			stopRequested = receivedByte == PipeConstants::Status::STOP || receivedByte == PipeConstants::Error::INTERNAL; // Stop if requested or the frontend fatally errors
			iteration++;
		}

		FreeMatrix(obstacles, iMax + 2);
		FreeMatrix(velocities.x, iMax + 2);
		FreeMatrix(velocities.y, iMax + 2);
		FreeMatrix(pressure, iMax + 2);
		FreeMatrix(RHS, iMax + 2);
		FreeMatrix(FG.x, iMax + 2);
		FreeMatrix(FG.y, iMax + 2);

		std::cout << "Backend stopped." << std::endl;

		pipeManager.SendByte(PipeConstants::Status::OK); // Send OK then stop executing

	}
	else { // Only continuous requests are supported
		std::cerr << "Server sent an unsupported request" << std::endl;
		pipeManager.SendByte(PipeConstants::Error::BADREQ);
	}
}

void FrontendManager::SetParameters(DoubleField& velocities, REAL**& pressure, REAL**& RHS, REAL**& streamFunction, DoubleField& FG, BYTE**& flags, bool**& obstacles, std::pair<int, int>*& coordinates, int& coordinatesLength, int& numFluidCells, SimulationParameters& parameters, DoubleReal& stepSizes)
{
	velocities.x = MatrixMAlloc(iMax + 2, jMax + 2);
	velocities.y = MatrixMAlloc(iMax + 2, jMax + 2);

	pressure = MatrixMAlloc(iMax + 2, jMax + 2);
	RHS = MatrixMAlloc(iMax + 2, jMax + 2);
	streamFunction = MatrixMAlloc(iMax + 1, jMax + 1);

	FG.x = MatrixMAlloc(iMax + 2, jMax + 2);
	FG.y = MatrixMAlloc(iMax + 2, jMax + 2);

	flags = FlagMatrixMAlloc(iMax + 2, jMax + 2);
	if (obstacles == nullptr) { // Perform default initialisation if not already initialised
		obstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);
		for (int i = 1; i <= iMax; i++) { for (int j = 1; j <= jMax; j++) { obstacles[i][j] = 1; } } //Set all the cells to fluid
#ifdef OBSTACLES
		int boundaryLeft = (int)(0.25 * iMax);
		int boundaryRight = (int)(0.35 * iMax);
		int boundaryBottom = (int)(0.45 * jMax);
		int boundaryTop = (int)(0.55 * jMax);

		for (int i = boundaryLeft; i < boundaryRight; i++) { // Create a square of boundary cells
			for (int j = boundaryBottom; j < boundaryTop; j++) {
				obstacles[i][j] = 0;
			}
		}
#endif // OBSTACLES
	}

	//SetObstacles(obstacles);
	SetFlags(obstacles, flags, iMax + 2, jMax + 2);

	std::pair<std::pair<int, int>*, int> coordinatesWithLength = FindBoundaryCells(flags, iMax, jMax);
	coordinates = coordinatesWithLength.first;
	coordinatesLength = coordinatesWithLength.second;

	numFluidCells = CountFluidCells(flags, iMax, jMax);

	parameters.width = 1;
	parameters.height = 1;
	parameters.timeStepSafetyFactor = 0.5;
	parameters.relaxationParameter = 1.7;
	parameters.pressureResidualTolerance = 2;
	parameters.pressureMinIterations = 1;
	parameters.pressureMaxIterations = 1000;
	parameters.reynoldsNo = 1000;
	parameters.inflowVelocity = 1;
	parameters.surfaceFrictionalPermissibility = 0;
	parameters.bodyForces.x = 0;
	parameters.bodyForces.y = 0;

	stepSizes.x = parameters.width / iMax;
	stepSizes.y = parameters.height / jMax;

	/*for (int i = 0; i <= iMax + 1; i++) {
		for (int j = 0; j <= jMax + 1; j++) {
			pressure[i][j] = 1000;
		}
	}*/
}

void FrontendManager::ReceiveData(BYTE startMarker) {
	if (startMarker == (PipeConstants::Marker::FLDSTART | PipeConstants::Marker::OBST)) { // Only supported use is obstacle send
		bool* obstaclesFlattened = new bool[(iMax + 2) * (jMax + 2)]();
		pipeManager.ReceiveObstacles(obstaclesFlattened, iMax + 2, jMax + 2);
		obstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);
		UnflattenArray(obstacles, obstaclesFlattened, fieldSize, jMax + 2);
		delete[] obstaclesFlattened;
	}
	else {
		std::cerr << "Server sent unsupported data" << std::endl;
		pipeManager.SendByte(PipeConstants::Error::BADREQ); // All others not supported at the moment
	}
}

FrontendManager::FrontendManager(int iMax, int jMax, std::string pipeName)
	: iMax(iMax), jMax(jMax), fieldSize(iMax * jMax), pipeManager(pipeName), obstacles(nullptr) // Set obstacles to null pointer to represent unallcoated
{}

int FrontendManager::Run() {
	pipeManager.Handshake(iMax, jMax);
	std::cout << "Handshake completed ok" << std::endl;

	bool closeRequested = false;

	while (!closeRequested) {
		std::cout << "In read loop" << std::endl;
		BYTE receivedByte = pipeManager.ReadByte();
		switch (receivedByte & PipeConstants::CATEGORYMASK) { // Gets the category of control byte
		case PipeConstants::Status::GENERIC: // Status bytes
			switch (receivedByte & ~PipeConstants::Status::PARAMMASK) {
			case PipeConstants::Status::HELLO:
			case PipeConstants::Status::BUSY:
			case PipeConstants::Status::OK:
			case PipeConstants::Status::STOP:
				std::cerr << "Server sent a status byte out of sequence, request not understood" << std::endl;
				pipeManager.SendByte(PipeConstants::Error::BADREQ);
				break;
			case PipeConstants::Status::CLOSE:
				closeRequested = true;
				pipeManager.SendByte(PipeConstants::Status::OK);
				std::cout << "Backend closing..." << std::endl;
				break;
			default:
				std::cerr << "Server sent a malformed status byte, request not understood" << std::endl;
				pipeManager.SendByte(PipeConstants::Error::BADREQ);
				break;
			}
			break;
		case PipeConstants::Request::GENERIC: // Request bytes have a separate handler
			HandleRequest(receivedByte);
			break;
		case PipeConstants::Marker::GENERIC: // So do marker bytes
			ReceiveData(receivedByte);
			break;
		default: // Error bytes
			break;
		}
	}
	return 0;
}