#include <iostream>
#include "Init.h"

int main()
{
    std::cout << "Hello World!\n";
    REAL** matrix = MatrixMAlloc(5, 5);
    matrix[4][4] = 4;

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << matrix[i][j] << "\n";
        }
        std::cout << "\n";
    }

    FreeMatrix(matrix, 5);
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
