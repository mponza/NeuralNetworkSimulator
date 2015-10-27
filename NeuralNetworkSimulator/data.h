#ifndef DATA_H
#define DATA_H

#include <iostream>

#include <vector>

using namespace std;

class Data
{    // Neural Network Data (one row of the file)
public:
    vector<double> targets;
    vector<double> features;

    Data();
    Data(const Data& data);
    Data (const vector<double>& features, const vector<double>& targets);
};

#endif // DATA_H
