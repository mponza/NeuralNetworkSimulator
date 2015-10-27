#include "crossvalidation.h"

CrossValidation::CrossValidation() {}

CrossValidation::CrossValidation(const CrossValidation& crossValidation) {
    CrossValidation::earlyStopping = crossValidation.earlyStopping;
    CrossValidation::folds = crossValidation.folds;
    CrossValidation::n = crossValidation.n;
}

CrossValidation::CrossValidation(const vector<Data>& dataSet, const int& k, const int& n, const bool& earlyStopping) {
    CrossValidation::earlyStopping = earlyStopping;
    CrossValidation::folds = Folds(dataSet, (earlyStopping ? k + 1 : k));
    CrossValidation::n = n;
}

vector<HyperParameters> CrossValidation::validate(const vector<HyperParameters>& parameters) const {
    int k = (earlyStopping ? folds.k - 1 : folds.k);
    vector<pair<HyperParameters, double>> parametersMSE;

    cout << "Running the " << k << "-folds Cross-Validation..." << endl;
    cout.flush();

    for(auto parameter = parameters.begin(); parameter != parameters.end(); ++parameter) {
        double averageMSE = 0;
        NeuralNetwork nn(*parameter);

        for(int i = 0; i < k; ++i) {
            vector<Data> trainingSet = folds.getTrainingSet(i);
            vector<Data> validationSet = folds.folds[i];
            if(earlyStopping) {
                nn.train(trainingSet, validationSet);
            } else {
                nn.train(trainingSet);
            }
            averageMSE += nn.computeError(validationSet);;
            nn.reset();
        }
        averageMSE /= k;
        printResults(*parameter, averageMSE);
        // save the hyper-parameters and the corrisponding averageError
        parametersMSE = bestHyperParameters(parametersMSE, *parameter, averageMSE);
    }

    if (earlyStopping) {
        // compute the maximum number of epochs with the best hyper-parameters found on the last fold
        vector<Data> trainingSet = folds.getTrainingSet(k);
        vector<Data> stoppingSet = folds.folds[k];
        for(int i = 0; i < static_cast<int>(parametersMSE.size()); ++i) {
            parametersMSE[i].first.epochs = NeuralNetwork(parametersMSE[i].first).train(trainingSet, stoppingSet);
        }
    }

    cout << "Best hyper-parameters configurations founded" << endl;
    for(auto best = parametersMSE.begin(); best != parametersMSE.end(); ++best) {
        printResults(best->first, best->second);
    }
    cout << "...end of the " << k << "-folds Cross-Validation." << endl;

    vector<HyperParameters> bestParameters;
    for(auto best = parametersMSE.begin(); best != parametersMSE.end(); ++best) {
        bestParameters.push_back(best->first);
    }
    return bestParameters;
}

vector<pair<HyperParameters, double>> CrossValidation::bestHyperParameters(vector<pair<HyperParameters, double>> parametersMSE,
                                                             const HyperParameters& newParameters, const double& newMSE) const {
    if(parametersMSE.size() == 0) {
        parametersMSE.push_back(make_pair<HyperParameters, double>(newParameters, newMSE));
        return parametersMSE;
    }

    double highestMSE = parametersMSE[0].second;
    int highestPos = 0;

    // compute the highest MSE and the corrisponding position in parametersMSE
    int i = 1;
    for(auto best = parametersMSE.begin() + 1; best != parametersMSE.end(); ++best) {
        if(best->second >= highestMSE) {
            highestMSE = best->second;
            highestPos = i;
        }
        ++i;
    }

    if(newMSE <= highestMSE) {
        parametersMSE[highestPos].first = newParameters;
        parametersMSE[highestPos].second = newMSE;
    }

    return parametersMSE;
}

