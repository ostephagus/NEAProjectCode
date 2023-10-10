#include "Definitions.h"
#include "Boundary.h"
#include <utility>
#include <bitset>

constexpr BYTE TOPMASK =    0b00001000;
constexpr BYTE RIGHTMASK =  0b00000100;
constexpr BYTE BOTTOMMASK = 0b00000010;
constexpr BYTE LEFTMASK =   0b00000001;
constexpr int TOPSHIFT = 3;
constexpr int RIGHTSHIFT = 2;
constexpr int BOTTOMSHIFT = 1;

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

void CopyBoundaryPressures(REAL** pressure, std::pair<int,int>* coordinates, BYTE** flags, int iMax, int jMax) {
	for (int i = 1; i <= iMax; i++) {
		pressure[i][0] = pressure[i][1];
		pressure[i][jMax + 1] = pressure[i][jMax];
	}
	for (int j = 1; j <= jMax; j++) {
		pressure[0][j] = pressure[1][j];
		pressure[iMax + 1][j] = pressure[iMax][j];
	}
	for (int coord = 0; coord < *(&coordinates + 1) - coordinates; coord++) {
		BYTE relevantFlag = flags[coordinates[coord].first][coordinates[coord].second];
		if (std::bitset<8>(relevantFlag).count() == 1) { // Only boundary cells with one edge
			pressure[coordinates[coord].first][coordinates[coord].second] = pressure[coordinates[coord].first + ((relevantFlag && TOPMASK) >> TOPSHIFT) - ((relevantFlag && BOTTOMMASK) >> BOTTOMSHIFT)][coordinates[coord].second + ((relevantFlag && RIGHTMASK) >> RIGHTSHIFT) - (relevantFlag && LEFTMASK)]; // Copying pressure from the relevant cell. Using anding with bit masks to do things like [i+1][j] using single bits
		}
		else { // These are boundary cells with 2 edges
			pressure[coordinates[coord].first][coordinates[coord].second] = (pressure[coordinates[coord].first + ((relevantFlag && TOPMASK) >> TOPSHIFT) - ((relevantFlag && BOTTOMMASK) >> BOTTOMSHIFT)][coordinates[coord].second] + pressure[coordinates[coord].first][coordinates[coord].second + ((relevantFlag && RIGHTMASK) >> RIGHTSHIFT) - (relevantFlag && LEFTMASK)]) / 2; //Take the average of the one above/below and the one left/right by keeping j constant for the first one, and I constant for the second one.
		}
	}
}