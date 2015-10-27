#ifndef HYPERPARAMETERS_H
#define HYPERPARAMETERS_H

enum Task{monk, loc};

class HyperParameters
{   // Class which includes all configurable neural network parameters
public:
    // data parameter
    int nFeatures = 0;

    // neural network parameters
    Task task;
    int nNeurons;
    int nOutputs;
    int epochs;
    double eta;
    double lambda;
    double alpha;

    HyperParameters();
    HyperParameters(const HyperParameters& parameters);
    HyperParameters(const Task& task);
    HyperParameters(const Task& task, const int& nNeurons, const int& epochs, const double& eta, const double& lambda, const double& alpha);
};

#endif // HYPERPARAMETERS_H
