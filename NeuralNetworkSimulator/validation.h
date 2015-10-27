#ifndef VALIDATION_H
#define VALIDATION_H

#include "modelselection.h"
#include "neuralnetwork.h"

using namespace std;

class Validation : public ModelSelection
{                                           // Simple Validation
public:
    vector<Data> trainingSet;
    vector<Data> validationSet;
    vector<Data> stoppingSet;

    Validation();
    Validation(const Validation& validation);
    Validation(const vector<Data>& trainingSet);

    HyperParameters validate(const vector<HyperParameters>& parameters) const;
};

#endif // VALIDATION_H
