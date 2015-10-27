#include "neuron.h"
#include "random.h"
#include "data.h"

Neuron::Neuron() { }

Neuron::Neuron(int nFeatures, Random r) {
    length = nFeatures + 1;     // because weigth[0] is the bias
    weights = vector<double>(length, 0);
    for (vector<double>::iterator iter = weights.begin(); iter != weights.end(); ++iter) {
        *iter = r.getRandomDouble();
    }
    deltaWeights = vector<double>(length, 0);
    output = 0;
    delta = 0;
}

vector<double> Neuron::getWeights() {
    return weights;
}

double Neuron::computeNet(const vector<double>& input) {
    net = weights[0]; // bias

    if (static_cast<int>(input.size()) != length - 1) throw string("Wrong input size in computeNet.");

    for (int i = 1; i < length; ++i) {
        net += input[i - 1] * weights[i];
    }
    return net;
}

double Neuron::sigmoid(const double& x) const {
    return 1 / (1 + exp(-x));
}

double Neuron::derivedSigmoid (const double& x) const {
    return sigmoid(x) * (1 - sigmoid(x));
}

double Neuron::computeOutput(const vector<double>& input) {
    lastInput = input;
    return output = sigmoid(computeNet(input));
}

double Neuron::getDelta() {
    return delta;
}

void Neuron::updateWeights(const double& eta, const double& lambda, const double& alpha, const int& size) {
    double in;
    double updating;

    for (int i = 0; i < length; ++i) {
        if (i == 0) {   // because bias
            in = 1;
        } else {
            in = lastInput[i - 1];
        }
        // Updating with momentum and weight decay
        updating = eta * (delta * in - 2 * lambda * weights[i] / size) + alpha * deltaWeights[i];
        weights[i] += updating;
        deltaWeights[i] = updating;
    }
}

double Neuron::findDelta(const vector<double>& outputNeuronDeltas, const vector<double>& outputWeights) {
    if (outputNeuronDeltas.size() != outputWeights.size()) {
        throw string("outputNeuronDeltas and outputWeights must have the same size");
    }
    double tmpDelta = 0;
    for (int i = 0; i < static_cast<int>(outputWeights.size()); ++i) {
        tmpDelta += outputNeuronDeltas[i] * outputWeights[i];
    }
    return  delta = tmpDelta * derivedSigmoid(net);
}

double Neuron::findDelta(const double& target) {
    return delta = (target - output) * derivedSigmoid(net);
}
