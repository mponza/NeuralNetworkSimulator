#include "validation.h"

Validation::Validation() {}

Validation::Validation(const Validation& validation) : Validation(validation.trainingSet) {}

Validation::Validation(const vector<Data>& trainingSet) {
    Validation::trainingSet = vector<Data>(trainingSet.begin(), trainingSet.begin() + trainingSet.size() * 0.7);
    Validation::validationSet = vector<Data>(trainingSet.begin() + trainingSet.size() * 0.7, trainingSet.begin() + trainingSet.size() * 0.85);
    Validation::stoppingSet = vector<Data>(trainingSet.begin() + trainingSet.size() * 0.85, trainingSet.end());
}

HyperParameters Validation::validate(const vector<HyperParameters>& parameters) const {
    HyperParameters bestParameters;
    bool firstBest = true;
    double bestError;

    cout << "Running the Simple Validation..." << endl;

    for(auto parameter = parameters.begin(); parameter != parameters.end(); ++parameter) {
        NeuralNetwork nn(*parameter);
        double averageMSE = 0;

        // train the neural network and keep the average of the empirical risk on 10 runs
        for(int i = 0; i < 10; ++i) {
            nn.train(trainingSet, validationSet);
            averageMSE += nn.computeError(validationSet);
            nn.reset();
        }
        averageMSE /= 10;
        printResults(*parameter, averageMSE);

        // first time comparison or better empirical risk
        if(firstBest == true || averageMSE < bestError) {
            firstBest = false;
            bestError = averageMSE;
            bestParameters = *parameter;
        }
    }

    // compute the maximum number of epochs with the best hyper-parameters found on the stoppingSet
    vector<Data> newTrainingSet;
    newTrainingSet.insert(newTrainingSet.end(), Validation::trainingSet.begin(), Validation::trainingSet.end());
    newTrainingSet.insert(newTrainingSet.end(), Validation::validationSet.begin(), Validation::validationSet.end());
    bestParameters.epochs = NeuralNetwork(bestParameters).train(newTrainingSet, stoppingSet);

    cout << "Best hyper-parameters founded:" << endl;
    printResults(bestParameters, bestError);
    cout << "...end of the Simple Validation." << endl;
    return bestParameters;
}
