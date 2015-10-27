#ifndef RANDOM_H
#define RANDOM_H

#include <stdlib.h>
#include <time.h>

class Random
{                                                       // It generates a random number in (min, max)
protected:
    double min;
    double max;
public:
    static bool alreadySeeded;

    Random();                                           // default values: min = -1, max = 1
    Random(const double& min, const double& max);
    double getRandomDouble() const;                     // between (min, max)
    int getRandomInt() const;
};

#endif // RANDOM_H
