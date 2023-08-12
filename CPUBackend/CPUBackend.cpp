#include <iostream>
#include "Boundary.h"
#include "Computation.h"
#include "Init.h"

int main()
{
    std::cout << "Hello World!\n";
    REAL** matrixX = MatrixMAlloc(5, 5);
    matrixX[4][4] = 4;

    REAL** matrixY = MatrixMAlloc(5, 5);
    matrixY[4][4] = 5;

    DoubleField matrices;
    matrices.x = matrixX;
    matrices.y = matrixY;

    REAL** matrixOne = matrices.x;
    REAL** matrixTwo = matrices.y;

    matrixOne[0][0] = 1;
    matrixTwo[0][0] = 2;

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << matrixOne[i][j] << "|" << matrices.x[i][j] << "\n";
        }
        std::cout << "\n";
    }

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << matrixTwo[i][j] << "|" << matrices.y[i][j] << "\n";
        }
        std::cout << "\n";
    }
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
