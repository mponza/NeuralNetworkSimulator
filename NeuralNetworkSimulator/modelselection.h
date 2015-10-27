#ifndef MODELSELECTION_H
#define MODELSELECTION_H

#include <iostream>
#include <vector>
#include "neuralnetwork.h"

using namespace std;

class ModelSelection
{
protected:
    void printResults(const HyperParameters& parameters, const double& empiricalRisk) const;
public:
    ModelSelection();
};

#endif // MODELSELECTION_H
