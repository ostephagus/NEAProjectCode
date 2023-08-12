#include "Definitions.h"
#include "Boundary.h"

void SetBoundaryConditions(DoubleField velocities, int iMax, int jMax, REAL inflowVelocity) {
	//Top and bottom: free-slip
	for (int i = 1; i <= iMax; i++) {
		velocities.y[i][0] = 0; // No mass crossing the boundary - velocity is 0
		velocities.y[i][jMax] = 0;

		velocities.x[i][0] = velocities.x[i][1]; // Speed outside the boundary is the same as the speed inside
		velocities.x[i][jMax + 1] = velocities.x[i][jMax];
	}

	for (int j = 1; j <= jMax; j++) {
		// Left: inflow
		velocities.x[0][j] = inflowVelocity; // Fluid flows in the x direction at a set velocity...
		velocities.y[0][j] = 0; // ...and there should be no movement in the y direction

		// Right: outflow
		velocities.x[iMax][j] = velocities.x[iMax - 1][j]; // Copy the velocity values from the previous cell (mass flows out at the boundary)
		velocities.y[iMax + 1][j] = velocities.y[iMax][j];
	}
}