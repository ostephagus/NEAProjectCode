#pragma once
typedef double REAL;

enum BoundaryCondition {inflow, freeSlip, noSlip, outflow};

struct BoundaryConditions
{
	BoundaryCondition top;
	BoundaryCondition right;
	BoundaryCondition bottom;
	BoundaryCondition left;
};