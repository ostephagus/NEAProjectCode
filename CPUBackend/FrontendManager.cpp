#include "FrontendManager.h"
#include "PipeConstants.h"
#include <iostream>
#include "PipeManager.h"
#include "Computation.h"
#include "Init.h"
#include "Boundary.h"

void FrontendManager::UnflattenArray(bool** pointerArray, bool* flattenedArray, int length, int divisions) {
	for (int i = 0; i < length / divisions; i++) {
		pointerArray[i] = flattenedArray + i * divisions;
	}
}

void FrontendManager::SetupParameters(REAL** horizontalVelocity, REAL** verticalVelocity, REAL** pressure, SimulationParameters& parameters, DoubleReal stepSizes) {
	parameters.width = 1;
	parameters.height = 1;
	parameters.timeStepSafetyFactor = 0.8;
	parameters.relaxationParameter = 1.2;
	parameters.pressureResidualTolerance = 1;
	parameters.pressureMaxIterations = 1000;
	parameters.reynoldsNo = 2000;
	parameters.inflowVelocity = 5;
	parameters.surfaceFrictionalPermissibility = 0;
	parameters.bodyForces.x = 0;
	parameters.bodyForces.y = 0;
	stepSizes.x = parameters.width / iMax;
	stepSizes.y = parameters.height / jMax;

	for (int i = 0; i <= iMax + 1; i++) {
		for (int j = 0; j <= jMax + 1; j++) {
			pressure[i][j] = 10;
		}
	}

	for (int i = 1; i <= iMax; i++) {
		for (int j = 1; j <= jMax; j++) {
			horizontalVelocity[i][j] = 4;
			verticalVelocity[i][j] = 0;
		}
	}
}

void FrontendManager::TimeStep(DoubleField velocities, DoubleField FG, REAL** pressure, REAL** nextPressure, REAL** RHS, REAL** streamFunction, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int numFluidCells, const SimulationParameters& parameters, DoubleReal stepSizes) {
	
	REAL timestep = 0;
	REAL pressureResidualNorm = 0;
	ComputeTimestep(timestep, iMax, jMax, stepSizes, velocities, parameters.reynoldsNo, parameters.timeStepSafetyFactor);
	SetBoundaryConditions(velocities, flags, coordinates, coordinatesLength, iMax, jMax, parameters.inflowVelocity, parameters.surfaceFrictionalPermissibility);
	REAL gamma = ComputeGamma(velocities, iMax, jMax, timestep, stepSizes);
	ComputeFG(velocities, FG, flags, iMax, jMax, timestep, stepSizes, parameters.bodyForces, gamma, parameters.reynoldsNo);
	ComputeRHS(FG, RHS, flags, iMax, jMax, timestep, stepSizes);
	Poisson(pressure, nextPressure, RHS, flags, coordinates, coordinatesLength, numFluidCells, iMax, jMax, stepSizes, parameters.pressureResidualTolerance, parameters.pressureMaxIterations, parameters.pressureMaxIterations, parameters.relaxationParameter, pressureResidualNorm);
	ComputeVelocities(velocities, FG, pressure, flags, iMax, jMax, timestep, stepSizes);
	ComputeStream(velocities, streamFunction, flags, iMax, jMax, stepSizes);
	std::cerr << "Timestep" << std::endl;
}

void FrontendManager::HandleRequest(BYTE requestByte) {
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

		DoubleField velocities;
		velocities.x = MatrixMAlloc(iMax + 2, jMax + 2);
		velocities.y = MatrixMAlloc(iMax + 2, jMax + 2);

		REAL** pressure = MatrixMAlloc(iMax + 2, jMax + 2);
		REAL** nextPressure = MatrixMAlloc(iMax + 2, jMax + 2);
		REAL** RHS = MatrixMAlloc(iMax + 2, jMax + 2);

		DoubleField FG;
		FG.x = MatrixMAlloc(iMax + 2, jMax + 2);
		FG.y = MatrixMAlloc(iMax + 2, jMax + 2);

		BYTE** flags = FlagMatrixMAlloc(iMax + 2, jMax + 2);
		bool** obstacles = ObstacleMatrixMAlloc(iMax + 2, jMax + 2);
		for (int i = 1; i <= iMax; i++) { for (int j = 1; j <= jMax; j++) { obstacles[i][j] = 1; } } //Set all the cells to fluid
		//SetObstacles(obstacles);
		SetFlags(obstacles, flags, iMax + 2, jMax + 2);
		//PrintField(flags, iMax + 2, jMax + 2, "flags");

		std::pair<std::pair<int, int>*, int> coordinatesWithLength = FindBoundaryCells(flags, iMax, jMax);
		std::pair<int, int>* coordinates = coordinatesWithLength.first;
		int coordinatesLength = coordinatesWithLength.second;

		int numFluidCells = CountFluidCells(flags, iMax, jMax);

		const REAL width = 1;
		const REAL height = 1;
		const REAL timeStepSafetyFactor = 0.8;
		const REAL relaxationParameter = 1.2;
		const REAL pressureResidualTolerance = 5; //Needs experimentation
		const int pressureMinIterations = 10;
		const int pressureMaxIterations = 1000; //Needs experimentation
		const REAL reynoldsNo = 2000;
		const REAL inflowVelocity = 5;
		const REAL surfaceFrictionalPermissibility = 0;
		REAL pressureResidualNorm = 0;

		DoubleReal bodyForces;
		bodyForces.x = 0;
		bodyForces.y = 0;

		REAL timestep;
		DoubleReal stepSizes;
		stepSizes.x = width / iMax;
		stepSizes.y = height / jMax;

		for (int i = 0; i <= iMax + 1; i++) {
			for (int j = 0; j <= jMax + 1; j++) {
				pressure[i][j] = 1000;
			}
		}
		//PrintField(pressure, iMax+2, jMax+2, "Pressure");
		for (int i = 1; i <= iMax; i++) {
			for (int j = 1; j <= jMax; j++) {
				velocities.x[i][j] = 4;
				velocities.y[i][j] = 0;
			}
		}

		bool stopRequested = false;

		pipeManager.SendByte(PipeConstants::Status::OK); // Send OK to say backend is set up and about to start executing

		while (!stopRequested) {
			pipeManager.SendByte(PipeConstants::Marker::ITERSTART);
			ComputeTimestep(timestep, iMax, jMax, stepSizes, velocities, reynoldsNo, timeStepSafetyFactor);
			SetBoundaryConditions(velocities, flags, coordinates, coordinatesLength, iMax, jMax, inflowVelocity, surfaceFrictionalPermissibility);
			REAL gamma = ComputeGamma(velocities, iMax, jMax, timestep, stepSizes);
			ComputeFG(velocities, FG, flags, iMax, jMax, timestep, stepSizes, bodyForces, gamma, reynoldsNo);
			ComputeRHS(FG, RHS, flags, iMax, jMax, timestep, stepSizes);
			Poisson(pressure, nextPressure, RHS, flags, coordinates, coordinatesLength, numFluidCells, iMax, jMax, stepSizes, pressureResidualTolerance, pressureMinIterations, pressureMaxIterations, relaxationParameter, pressureResidualNorm);
			ComputeVelocities(velocities, FG, pressure, flags, iMax, jMax, timestep, stepSizes);

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
			/*if (streamWanted) {
				pipeManager.SendByte(PipeConstants::Marker::FLDSTART | PipeConstants::Marker::STRM);
				pipeManager.SendField(streamFunction, iMax, jMax, 0, 0);
				pipeManager.SendByte(PipeConstants::Marker::FLDEND | PipeConstants::Marker::STRM);
			}*/

			pipeManager.SendByte(PipeConstants::Marker::ITEREND);

			BYTE receivedByte = pipeManager.ReadByte(); // This may require duplex communication

			stopRequested = receivedByte == PipeConstants::Status::STOP || receivedByte == PipeConstants::Error::INTERNAL; // Stop if requested or the frontend fatally errors
		}

		std::cout << "Stopping..." << std::endl;

		FreeMatrix(velocities.x, iMax + 2);
		FreeMatrix(velocities.y, iMax + 2);
		FreeMatrix(pressure, iMax + 2);
		FreeMatrix(nextPressure, iMax + 2);
		FreeMatrix(RHS, iMax + 2);
		FreeMatrix(FG.x, iMax + 2);
		FreeMatrix(FG.y, iMax + 2);

		std::cout << "Stopped." << std::endl;

		pipeManager.SendByte(PipeConstants::Status::OK); // Send OK then stop executing

	}
	else { // Only continuous requests are supported
		std::cerr << "Server sent an unsupported request" << std::endl;
		pipeManager.SendByte(PipeConstants::Error::BADREQ);
	}
}

void FrontendManager::ReceiveData(BYTE startMarker) {
	if (startMarker == (PipeConstants::Marker::FLDSTART | PipeConstants::Marker::OBST)) { // Only supported use is obstacle send
		bool* obstaclesFlattened = new bool[fieldSize];
		bool** obstacles = ObstacleMatrixMAlloc(iMax, jMax);
		UnflattenArray(obstacles, obstaclesFlattened, fieldSize, jMax);
	}
	else {
		std::cerr << "Server sent unsupported data" << std::endl;
		pipeManager.SendByte(PipeConstants::Error::BADREQ); // All others not supported at the moment
	}
}

FrontendManager::FrontendManager(int iMax, int jMax, std::string pipeName)
	: iMax(iMax), jMax(jMax), fieldSize(iMax * jMax), pipeManager(pipeName)
{}

int FrontendManager::Run() {
	pipeManager.Handshake(iMax, jMax);
	std::cerr << "Handshake completed ok" << std::endl;

	bool closeRequested = false;

	while (!closeRequested) {
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

	//delete[] obstaclesFlattened;
	return 0;
}