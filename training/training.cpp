#include <fstream>
#include <iostream>
#include <algorithm>

int readFlippedInteger(std::istream &in) {
    char temp[sizeof(int)];
    in.read(temp, sizeof(int));
    std::reverse(temp, temp+sizeof(int));
    return *reinterpret_cast<int*>(temp);
}

int main() {
    std::ifstream fin("MNIST/train-images.idx3-ubyte", std::ios::binary);

    if (!fin) {
        std::cerr << "Could not open file\n";
        return -1;
    }

    // delcare function;
    int magicNumber = readFlippedInteger(fin);
    int numImages = readFlippedInteger(fin);
    int numRows = readFlippedInteger(fin);
    int numCols = readFlippedInteger(fin);

    std::cout << magicNumber << std::endl // 2051
              << numImages << std::endl   // 60000
              << numRows << std::endl    // 28
              << numCols << std::endl;  // 28
}