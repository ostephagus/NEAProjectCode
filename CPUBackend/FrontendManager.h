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

	void HandleRequest(BYTE requestByte);
	void ReceiveData(BYTE startMarker);

public:
	FrontendManager(int iMax, int jMax, std::string pipeName);
	int Run();
};

#endif // !FRONTEND_MANAGER_H
