#include "hyperparameters.h"

HyperParameters::HyperParameters() {}

HyperParameters::HyperParameters(const HyperParameters& parameters)
    : HyperParameters(parameters.task, parameters.nNeurons, parameters.epochs, parameters.eta, parameters.lambda, parameters.alpha) {}

HyperParameters::HyperParameters(const Task& task)
    : HyperParameters(task, 0, 0, 0, 0, 0) {}


HyperParameters::HyperParameters(const Task& task, const int& nNeurons, const int& epochs, const double& eta, const double& lambda, const double& alpha) {
    HyperParameters::task = task;
    HyperParameters::nFeatures = (task == monk ? 17 : 5);
    HyperParameters::nNeurons = nNeurons;
    HyperParameters::nOutputs = (task == monk ? 1 : 2);
    HyperParameters::epochs = epochs;
    HyperParameters::eta = eta;
    HyperParameters::lambda = lambda;
    HyperParameters::alpha = alpha;
}
