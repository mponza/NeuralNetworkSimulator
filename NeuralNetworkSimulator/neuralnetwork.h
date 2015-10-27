#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "neuron.h"
#include "hyperparameters.h"

using namespace std;

class NeuralNetwork
{                   // Neural Network with one layer of hidden units
                    // and one layer of output units
protected:
    void updateWeights(vector<Neuron>& layer, const int& size);
    vector<vector<double>> computeClassification(const vector<Data>& dataSet);
    double computeMSE(const vector<vector<double>> outputs, const vector<Data>& dataSet);
public:
    HyperParameters parameters;
    vector<Neuron> hiddenLayer;
    vector<Neuron> outputLayer;

    NeuralNetwork();
    NeuralNetwork(const NeuralNetwork& nn);
    NeuralNetwork(const HyperParameters& parameters);

    // create and randomly initialize the hidden and output layers
    void reset();

    vector<vector<double>> computeOutput(const vector<Data>& inputs);

    void train(const vector<Data>& trainingSet);

    // training with early stopping, it returns the number of epochs needed
    int train(const vector<Data>& trainingSet, const vector<Data>& validationSet);

    vector<double> computeOutput(const Data& input);

    // compute the error function specified in parameters.task
    double computeError(const vector<Data>& dataSet);
};

#endif // NEURALNETWORK_H
