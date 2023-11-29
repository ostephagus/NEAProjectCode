#include "Boundary.h"
#include <bitset>
#include <vector>
#include <iterator>

#define XVEL velocities.x[coordinates[coord].first][coordinates[coord].second]
#define YVEL velocities.y[coordinates[coord].first][coordinates[coord].second]

constexpr BYTE TOPMASK =    0b00001000;
constexpr BYTE RIGHTMASK =  0b00000100;
constexpr BYTE BOTTOMMASK = 0b00000010;
constexpr BYTE LEFTMASK =   0b00000001;
constexpr int TOPSHIFT = 3;
constexpr int RIGHTSHIFT = 2;
constexpr int BOTTOMSHIFT = 1;

void SetBoundaryConditions(DoubleField velocities, BYTE** flags, std::pair<int, int>* coordinates, int coordinatesLength, int iMax, int jMax, REAL inflowVelocity, REAL chi) {
	REAL velocityModifier = 2 * chi - 1; // This converts chi from chi in [0,1] to in [-1,1]

	//Top and bottom: free-slip
	for (int i = 1; i <= iMax; i++) {
		velocities.y[i][0] = 0; // No mass crossing the boundary - velocity is 0
		velocities.y[i][jMax] = 0;

		velocities.x[i][0] = velocities.x[i][1]; // Speed outside the boundary is the same as the speed inside
		velocities.x[i][jMax + 1] = velocities.x[i][jMax];
	}

	for (int j = 1; j <= jMax; j++) {
		// Left: inflow for bottom quarter
		if (j < jMax / 4) {
			velocities.x[0][j] = inflowVelocity; // Fluid flows in the x direction at a set velocity for bottom quarter ...
		}
		else {
			velocities.x[0][j] = 0; // ... but has no flow across the boundary in the other 75%
		}

		velocities.y[0][j] = 0; // There should be no movement in the y direction	
		// Right: outflow
		velocities.x[iMax][j] = velocities.x[iMax - 1][j]; // Copy the velocity values from the previous cell (mass flows out at the boundary)
		velocities.y[iMax + 1][j] = velocities.y[iMax][j];
	}

	// Obstacle boundary cells: partial-slip
	for (int coord = 0; coord < coordinatesLength; coord++) {
		BYTE relevantFlag = flags[coordinates[coord].first][coordinates[coord].second];
		switch (relevantFlag) {
		case B_N:
			XVEL = velocityModifier * velocities.x[coordinates[coord].first][coordinates[coord].second + 1]; //Tangential velocity: friction
			YVEL = 0; //Normal velocity = 0
			break;
		case B_NE:
			XVEL = 0; //Both velocities owned by a B_NE are normal, so set to 0.
			YVEL = 0; 
			break;
		case B_E:
			XVEL = 0; //Normal velocity = 0
			YVEL = velocityModifier * velocities.x[coordinates[coord].first + 1][coordinates[coord].second]; //Tangential velocity: friction
			break;
		case B_SE:
			XVEL = 0;
			YVEL = velocityModifier * velocities.x[coordinates[coord].first + 1][coordinates[coord].second]; //Tangential velocity: friction
			velocities.y[coordinates[coord].first][coordinates[coord].second - 1] = 0; // y velocity south of a B_SE must be set to 0
			break;
		case B_S:
			XVEL = velocityModifier * velocities.x[coordinates[coord].first][coordinates[coord].second - 1]; //Tangential velocity: friction
			velocities.y[coordinates[coord].first][coordinates[coord].second - 1] = 0; // y velocity south of a B_S must be set to 0
			break;
		case B_SW:
			XVEL = velocityModifier * velocities.x[coordinates[coord].first][coordinates[coord].second - 1]; //Tangential velocity: friction
			YVEL = velocityModifier * velocities.x[coordinates[coord].first - 1][coordinates[coord].second]; //Tangential velocity: friction
			velocities.x[coordinates[coord].first - 1][coordinates[coord].second] = 0; // x velocity west of a B_SW must be set to 0
			velocities.y[coordinates[coord].first][coordinates[coord].second - 1] = 0; // y velocity south of a B_SW must be set to 0
			break;
		case B_W:
			YVEL = velocityModifier * velocities.x[coordinates[coord].first - 1][coordinates[coord].second]; //Tangential velocity: friction
			velocities.x[coordinates[coord].first - 1][coordinates[coord].second] = 0; // x velocity west of a B_W must be set to 0
			break;
		case B_NW:
			XVEL = velocityModifier * velocities.x[coordinates[coord].first][coordinates[coord].second + 1]; //Tangential velocity: friction
			YVEL = 0; //Normal velocity = 0
			velocities.x[coordinates[coord].first - 1][coordinates[coord].second] = 0; // x velocity west of a B_NW must be set to 0
			break;
		}
		// Any velocities for a cell with a north or east bit unset (referring to an obstacle in that direction) must be set to 0, i.e. cells south or west of a boundary.
	}
}

void CopyBoundaryPressures(REAL** pressure, std::pair<int,int>* coordinates, int numCoords, BYTE** flags, int iMax, int jMax) {
	for (int i = 1; i <= iMax; i++) {
		pressure[i][0] = pressure[i][1];
		pressure[i][jMax + 1] = pressure[i][jMax];
	}
	for (int j = 1; j <= jMax; j++) {
		pressure[0][j] = pressure[1][j];
		pressure[iMax + 1][j] = pressure[iMax][j];
	}
	for (int coord = 0; coord < numCoords; coord++) {
		BYTE relevantFlag = flags[coordinates[coord].first][coordinates[coord].second];
		if (std::bitset<8>(relevantFlag).count() == 1) { // Only boundary cells with one edge
			pressure[coordinates[coord].first][coordinates[coord].second] = pressure[coordinates[coord].first + ((relevantFlag & RIGHTMASK) >> RIGHTSHIFT) - (relevantFlag & LEFTMASK)][coordinates[coord].second + ((relevantFlag & TOPMASK) >> TOPSHIFT) - ((relevantFlag & BOTTOMMASK) >> BOTTOMSHIFT)]; // Copying pressure from the relevant cell. Using anding with bit masks to do things like [i+1][j] using single bits
		}
		else { // These are boundary cells with 2 edges
			pressure[coordinates[coord].first][coordinates[coord].second] = (pressure[coordinates[coord].first + ((relevantFlag & RIGHTMASK) >> RIGHTSHIFT) - (relevantFlag & LEFTMASK)][coordinates[coord].second] + pressure[coordinates[coord].first][coordinates[coord].second + ((relevantFlag & TOPMASK) >> TOPSHIFT) - ((relevantFlag & BOTTOMMASK) >> BOTTOMSHIFT)]) / (REAL)2; //Take the average of the one above/below and the one left/right by keeping j constant for the first one, and I constant for the second one.
		}
	}
}

//Counts number of fluid cells in the region [1,iMax]x[1,jMax]
int CountFluidCells(BYTE** flags, int iMax, int jMax) {
	int count = 0;
	for (int i = 0; i <= iMax; i++) {
		for (int j = 0; j <= jMax; j++) {
			count += flags[i][j] >> 4; //This will include only the "self" bit, which is one for fluid cells and 0 for boundary and obstacle cells.
		}
	}
	return count;
}

std::pair<std::pair<int, int>*, int> FindBoundaryCells(BYTE** flags, int iMax, int jMax) { //Returns size of array rather than actual array
	std::vector<std::pair<int, int>> coordinates;
	for (int i = 1; i <= iMax; i++) {
		for (int j = 1; j <= jMax; j++) {
			if (flags[i][j] >= 0b00000001 && flags[i][j] <= 0b00001111) { // This defines boundary cells - all cells without the self bit set except when no bits are set. This could probably be optimised.
				coordinates.push_back(std::pair<int, int>(i, j));
			}
		}
	}
	std::pair<int, int>* coordinatesAsArray = new std::pair<int, int>[coordinates.size()]; // Allocate mem for array into already defined pointer
	std::copy(coordinates.begin(), coordinates.end(), coordinatesAsArray); // Copy the elements from the vector to the array
	return std::pair<std::pair<int, int>*, int>(coordinatesAsArray, coordinates.size()); // Return the array with values copied into it and the size
}