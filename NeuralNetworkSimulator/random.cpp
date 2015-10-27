#include "random.h"
#include <iostream>

bool Random::alreadySeeded;

Random::Random() : Random(-1, 1) {}

Random::Random(const double& min, const double& max) {
    Random::min = min;
    Random::max = max;
    if(!alreadySeeded) {
        srand(static_cast<unsigned int>(time(0)));
        alreadySeeded = true;
    }
}

double Random::getRandomDouble() const {
    return (static_cast<double>(rand()) / static_cast<double>(RAND_MAX) * (max - min)) + min;
}

int Random::getRandomInt() const {
    return rand() % static_cast<int>(max - min + 1) + min;
}
