#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& nn) : NeuralNetwork(nn.parameters) {
    NeuralNetwork::hiddenLayer = nn.hiddenLayer;
    NeuralNetwork::outputLayer = nn.outputLayer;
}

NeuralNetwork::NeuralNetwork(const HyperParameters& parameters) {
    NeuralNetwork::parameters = parameters;
    reset();
}

void NeuralNetwork::reset() {
    Random r = Random(-0.7, 0.7);
    hiddenLayer.clear();
    outputLayer.clear();

    for (int i = 0; i < parameters.nNeurons; ++i) {
        hiddenLayer.push_back(Neuron(parameters.nFeatures, r));
    }
    for (int i = 0; i < parameters.nOutputs; ++i) {
        outputLayer.push_back(Neuron(parameters.nNeurons, r));
    }
}

vector<double> NeuralNetwork::computeOutput(const Data& input) {
    vector<double> hiddenOutputs;
    vector<double> outputs;

    for (auto hiddenNeuron = hiddenLayer.begin(); hiddenNeuron != hiddenLayer.end(); ++hiddenNeuron) {
        hiddenOutputs.push_back((*hiddenNeuron).computeOutput(input.features));
    }
    for (auto outputNeuron = outputLayer.begin(); outputNeuron != outputLayer.end(); ++outputNeuron) {
        outputs.push_back((*outputNeuron).computeOutput(hiddenOutputs));
    }
    return outputs;
}

vector<vector<double>> NeuralNetwork::computeOutput(const vector<Data> &inputs) {
    vector<vector<double>> outputs;

    for(auto input = inputs.begin(); input != inputs.end(); ++input) {
        outputs.push_back(computeOutput(*input));
    }
    return outputs;
}

void NeuralNetwork::train(const vector<Data>& trainingSet) {
    for (int e = 0; e < parameters.epochs; ++e) {
        // for every example in the trainingSet
        for (auto example = trainingSet.begin(); example != trainingSet.end(); ++example) {
            computeOutput(*example);     // after this we can compute the neural network error

            // find delta for every output neuron
            int i = 0;
            for (auto outputNeuron = outputLayer.begin(); outputNeuron != outputLayer.end(); ++outputNeuron) {
                (*outputNeuron).findDelta(example->targets[i]);
                ++i;
            }

            // find delta for every hidden neuron
            i = 1;  // because i = 0 is bias
            for (auto hiddenNeuron = hiddenLayer.begin(); hiddenNeuron != hiddenLayer.end(); ++hiddenNeuron) {
                // compute weights and deltas output neurons linked to the current hidden neuron (*iter)
                vector<double> weights;
                vector<double> deltas;
                for (auto outputNeuron = outputLayer.begin(); outputNeuron != outputLayer.end(); ++outputNeuron) {
                    weights.push_back((*outputNeuron).getWeights()[i]);
                    deltas.push_back((*outputNeuron).getDelta());
                }
                (*hiddenNeuron).findDelta(deltas, weights);
                ++i;
            }

            updateWeights(outputLayer, trainingSet.size());
            updateWeights(hiddenLayer, trainingSet.size());
        }
    }
}

void NeuralNetwork::updateWeights(vector<Neuron>& layer, const int& size) {
    for (auto neuron = layer.begin(); neuron != layer.end(); ++neuron) {
        neuron->updateWeights(parameters.eta, parameters.lambda, parameters.alpha, size);
    }
}

int NeuralNetwork::train(const vector<Data>& trainingSet, const vector<Data>& validationSet) {
    int maxEpochs = parameters.epochs;
    double lastError = computeError(validationSet);
    double currentError;
    int i;

    parameters.epochs = 100;
    for(i = 0; i < maxEpochs; i += 100) {
        train(trainingSet);
        currentError = computeError(validationSet);
        if(lastError < currentError || currentError == 0) { // early stopping
            parameters.epochs = maxEpochs;
            return i + 100;
        } else {
            lastError = currentError;
        }
    }
    parameters.epochs = maxEpochs;
    return i;
}

double NeuralNetwork::computeError(const vector<Data>& dataSet) {
    switch(parameters.task) {
        case(monk): return computeMSE(computeClassification(dataSet), dataSet);
        case(loc):
            return computeMSE(computeOutput(dataSet), dataSet);
    }
}

vector<vector<double>> NeuralNetwork::computeClassification(const vector<Data>& dataSet) {
    vector<vector<double>> outputs = computeOutput(dataSet);
    vector<vector<double>> classificatedOutputs;

    for(vector<vector<double>>::const_iterator cIter = outputs.begin(); cIter != outputs.end(); ++cIter) {
        vector<double> classification;
        if((*cIter)[0] >= 0.5) {
            classification.push_back(1);
            classificatedOutputs.push_back(classification);
        }
        else {
            classification.push_back(0);
            classificatedOutputs.push_back(classification);
        }
    }
    return classificatedOutputs;
}

double NeuralNetwork::computeMSE(const vector<vector<double>> outputs, const vector<Data>& dataSet) {
    double mse = 0;

    for(int i = 0; i < static_cast<int>(outputs.size()); ++i) {
        for(int j = 0; j < static_cast<int>(outputs[i].size()); ++j) {
            double out = outputs[i][j];
            double target = dataSet[i].targets[j];
            mse += pow(out - target, 2.0);
        }
    }
    return (mse/outputs.size());
}
