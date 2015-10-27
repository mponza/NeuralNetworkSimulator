#include "neuralnetworksimulator.h"

NeuralNetworkSimulator::NeuralNetworkSimulator() {
    trainingSet = fileManager.getTrainingSet();
    testSet = fileManager.getTestSet();
}

NeuralNetworkSimulator::NeuralNetworkSimulator(const int& monkIndex) {
    NeuralNetworkSimulator::monkIndex = monkIndex;
    fileManager = FileManager(monkIndex);
    trainingSet = fileManager.getTrainingSet();
    testSet = fileManager.getTestSet();
}

void NeuralNetworkSimulator::simulate(HyperParameters parameters) {
    if(monkIndex >= 1) {
        cout << "Running Monk "<< monkIndex << " Simulation" << endl;
        generateMonkPerformance(parameters);
        cout << "...end of the Monk "<< monkIndex << " Simulation" << endl;
    } else {
        cout << "Running Loc Simulation" << endl;
        generateLocPerformance(parameters);
        generateCompetition(parameters);
        cout << "...end of the Loc Simulation." << endl;
    }
}

void NeuralNetworkSimulator::simulate() {
    if(monkIndex >= 1) {
        cout << "Running Monk "<< monkIndex << " Simulation..." << endl;
        HyperParameters bestParameters;
        bestParameters = Validation(trainingSet.get1OfKDataSet()).validate(getMonkParameters());
        generateMonkPerformance(bestParameters);
        cout << "...end of the Monk "<< monkIndex << " Simulation." << endl;
    } else {
        cout << "Running Loc Simulation..." << endl;
        vector<HyperParameters> bestParameters;
        bestParameters = CrossValidation(trainingSet.getScaledDataSet(), 5, 3, true).validate(getLocParameters());
        bestParameters = CrossValidation(trainingSet.getScaledDataSet(), 10, 1, false).validate(getLocParameters());
        generateLocPerformance(bestParameters[0]);
        generateCompetition(bestParameters[0]);
        cout << "...end of the Loc Simulation." << endl;
    }
}

/////////////////////////////////////////////////// Loc Functions //////////////////////////////////////////////////

vector<HyperParameters> NeuralNetworkSimulator::getLocParameters() const {

    vector<HyperParameters> parameters;

    vector<int> nNeurons; nNeurons.push_back(10); nNeurons.push_back(15); nNeurons.push_back(20);
    vector<double> etas; etas.push_back(0.005);
    vector<double> alphas; alphas.push_back(0); alphas.push_back(0.1); alphas.push_back(0.3); alphas.push_back(0.5); alphas.push_back(0.8);
    vector<double> lambdas; lambdas.push_back(0); lambdas.push_back(0.001); lambdas.push_back(0.003); lambdas.push_back(0.005); lambdas.push_back(0.008);

    for(auto n = nNeurons.begin(); n != nNeurons.end(); ++n) {
        for(auto eta = etas.begin(); eta != etas.end(); ++eta) {
            for(auto lambda = lambdas.begin(); lambda != lambdas.end(); ++lambda) {
                for(auto alpha = alphas.begin(); alpha != alphas.end(); ++alpha) {
                    parameters.push_back(HyperParameters(loc, *n, 10000, *eta, *lambda, *alpha));
                }
            }
        }
    }

    return parameters;
}

void NeuralNetworkSimulator::locRun(HyperParameters& parameters) {
    vector<Data> scaledTraining = trainingSet.getScaledDataSet();
    vector<Data> originalTraining = trainingSet.dataSet;
    vector<Data> trainingDataSet = vector<Data>(scaledTraining.begin(), scaledTraining.begin() + scaledTraining.size() * 0.7);
    vector<Data> validationDataSet = vector<Data>(scaledTraining.begin() + scaledTraining.size() * 0.7, scaledTraining.end());
    vector<Data> originalTrainingDataSet = vector<Data>(originalTraining.begin(), originalTraining.begin() + originalTraining.size() * 0.7);
    vector<Data> originalValidationDataSet = vector<Data>(originalTraining.begin() + originalTraining.size() * 0.7, originalTraining.end());


    vector<pair<double, double>> trainingMSE;
    vector<pair<double, double>> validationMSE;
    vector<pair<double, double>> trainingMEE;
    vector<pair<double, double>> validationMEE;
    int maxEpochs = parameters.epochs;

    cout << "Loc Running..." << endl;
    // save the mse and mee errors for every epoch
    NeuralNetwork nn(parameters);
    nn.parameters.epochs = 1;
    trainingMSE.push_back(make_pair<int, double>(0, nn.computeError(trainingDataSet)));
    validationMSE.push_back(make_pair<int, double>(0, nn.computeError(validationDataSet)));
    trainingMEE.push_back(make_pair<int, double>(0, computeMee(trainingSet.rescaleOutputs(nn.computeOutput(trainingDataSet)), originalTrainingDataSet)));
    validationMEE.push_back(make_pair<int, double>(0, computeMee(trainingSet.rescaleOutputs(nn.computeOutput(validationDataSet)), originalValidationDataSet)));
    for(int i = 1; i <= maxEpochs; ++i) {
        nn.train(trainingDataSet);
        if(i % 10 == 0) {
            trainingMSE.push_back(make_pair<int, double>(i, nn.computeError(trainingDataSet)));
            validationMSE.push_back(make_pair<int, double>(i, nn.computeError(validationDataSet)));
            trainingMEE.push_back(make_pair<int, double>(i, computeMee(trainingSet.rescaleOutputs(nn.computeOutput(trainingDataSet)), originalTrainingDataSet)));
            validationMEE.push_back(make_pair<int, double>(i, computeMee(trainingSet.rescaleOutputs(nn.computeOutput(validationDataSet)), originalValidationDataSet)));
            if((validationMSE[validationMSE.size() - 1].second == 0) ||
               (i % 100 == 0 && validationMSE[validationMSE.size() - 1].second > validationMSE[validationMSE.size() - 10].second))
                break;
        }
    }
    parameters.epochs = maxEpochs;

    cout << "Final Training MSE: " << trainingMSE[trainingMSE.size() - 1].second << endl;
    cout << "Final Validation MSE: " << validationMSE[validationMSE.size() - 1].second << endl;
    cout << "Final Training MEE: " << trainingMEE[trainingMEE.size() - 1].second << endl;
    cout << "Final Validation MEE: " << validationMEE[validationMEE.size() - 1].second << endl;

    cout << "Writing Loc performance on file...";
    // writing errors about training and test set
    fileManager.writeErrors(trainingMSE, "TrainingRunMSE");
    fileManager.writeErrors(validationMSE, "ValidationRunMSE");
    fileManager.writeErrors(trainingMEE, "TrainingRunMEE");
    fileManager.writeErrors(validationMEE, "ValidationRunMEE");
    cout << " end of the performance writing." << endl;
    cout << "end of the Loc Running." << endl;
}


void NeuralNetworkSimulator::locFoldRun(HyperParameters& parameters) {
    vector<Data> scaledTraining = trainingSet.getScaledDataSet();
    vector<Data> originalTraining = trainingSet.dataSet;
    Folds scaledFolds = Folds(scaledTraining, 10);
    Folds originalFolds = Folds(originalTraining, 10);
    int n = 5; // (2, 5 or 7)

    vector<pair<double, double>> trainingMSE;
    vector<pair<double, double>> validationMSE;
    vector<pair<double, double>> trainingMEE;
    vector<pair<double, double>> validationMEE;
    int maxEpochs = parameters.epochs;

    cout << "Loc Fold Running..." << endl;
    // save the mse and mee errors for every epoch
    NeuralNetwork nn(parameters);
    nn.parameters.epochs = 1;
    trainingMSE.push_back(make_pair<int, double>(0, nn.computeError(scaledFolds.getTrainingSet(n))));
    validationMSE.push_back(make_pair<int, double>(0, nn.computeError(scaledFolds.folds[n])));
    trainingMEE.push_back(make_pair<int, double>(0, computeMee(trainingSet.rescaleOutputs(nn.computeOutput(scaledFolds.getTrainingSet(n))), originalFolds.getTrainingSet(n))));
    validationMEE.push_back(make_pair<int, double>(0, computeMee(trainingSet.rescaleOutputs(nn.computeOutput(scaledFolds.folds[n])), originalFolds.folds[n])));
    for(int i = 1; i <= maxEpochs; ++i) {
        nn.train(scaledFolds.getTrainingSet(n));
        if(i % 10 == 0) {
            trainingMSE.push_back(make_pair<int, double>(i, nn.computeError(scaledFolds.getTrainingSet(n))));
            validationMSE.push_back(make_pair<int, double>(i, nn.computeError(scaledFolds.folds[n])));
            trainingMEE.push_back(make_pair<int, double>(i, computeMee(trainingSet.rescaleOutputs(nn.computeOutput(scaledFolds.getTrainingSet(n))), originalFolds.getTrainingSet(n))));
            validationMEE.push_back(make_pair<int, double>(i, computeMee(trainingSet.rescaleOutputs(nn.computeOutput(scaledFolds.folds[n])), originalFolds.folds[n])));
        }
    }
    parameters.epochs = maxEpochs;

    cout << "Final Training MSE: " << trainingMSE[trainingMSE.size() - 1].second << endl;
    cout << "Final Validation MSE: " << validationMSE[validationMSE.size() - 1].second << endl;
    cout << "Final Training MEE: " << trainingMEE[trainingMEE.size() - 1].second << endl;
    cout << "Final Validation MEE: " << validationMEE[validationMEE.size() - 1].second << endl;

    cout << "Writing Loc performance on file...";

    stringstream ss;
    ss << parameters.nNeurons;

    // writing errors about training and test set
    fileManager.writeErrors(trainingMSE, "TrainingFoldRunMSE_" + ss.str());
    fileManager.writeErrors(validationMSE, "ValidationFoldRunMSE_" + ss.str());
    fileManager.writeErrors(trainingMEE, "TrainingFoldRunMEE_" + ss.str());
    fileManager.writeErrors(validationMEE, "ValidationFoldRunMEE_" + ss.str());
    cout << " end of the performance writing." << endl;
    cout << "end of the Loc Running." << endl;
}

void NeuralNetworkSimulator::generateLocPerformance(HyperParameters parameters) {
    vector<pair<double, double>> trainingMSE;
    vector<pair<double, double>> testMSE;
    vector<pair<double, double>> trainingMEE;
    vector<pair<double, double>> testMEE;
    double finalMSE = 0;
    double finalMEE = 0;
    int maxEpochs = parameters.epochs;

    vector<Data> trainingDataSet = trainingSet.getScaledDataSet();
    vector<Data> testDataSet = trainingSet.scale(testSet.dataSet);

    cout << "Loc Testing..." << endl;

    // save the mse and mee errors for every epoch
    NeuralNetwork nn(parameters);
    nn.parameters.epochs = 1;
    trainingMSE.push_back(make_pair<int, double>(0, nn.computeError(trainingDataSet)));
    testMSE.push_back(make_pair<int, double>(0, nn.computeError(testDataSet)));
    trainingMEE.push_back(make_pair<int, double>(0, computeMee(trainingSet.rescaleOutputs(nn.computeOutput(trainingDataSet)), trainingSet.dataSet)));
    testMEE.push_back(make_pair<int, double>(0, computeMee(trainingSet.rescaleOutputs(nn.computeOutput(testDataSet)), testSet.dataSet)));

    for(int i = 1; i <= maxEpochs; ++i) {
        nn.train(trainingDataSet);
        if(i % 10 == 0) {
            trainingMSE.push_back(make_pair<int, double>(i, nn.computeError(trainingDataSet)));
            testMSE.push_back(make_pair<int, double>(i, nn.computeError(testDataSet)));
            trainingMEE.push_back(make_pair<int, double>(i, computeMee(trainingSet.rescaleOutputs(nn.computeOutput(trainingDataSet)), trainingSet.dataSet)));
            testMEE.push_back(make_pair<int, double>(i, computeMee(trainingSet.rescaleOutputs(nn.computeOutput(testDataSet)), testSet.dataSet)));
        }
    }
    nn.parameters.epochs = maxEpochs;

    cout << "Writing Loc performance on file..." << endl;
    // writing errors about training and test set
    fileManager.writeErrors(trainingMSE, "TrainingMSE");
    fileManager.writeErrors(testMSE, "TestMSE");
    fileManager.writeErrors(trainingMEE, "TrainingMEE");
    fileManager.writeErrors(testMEE, "TestMEE");
    cout << "...end of the performance writing." << endl;

    nn.train(trainingDataSet);
    cout << "Writing points on file..." << endl;
    vector<vector<double>> points = trainingSet.rescaleOutputs(nn.computeOutput(testDataSet));
    fileManager.writePoints(points, testSet.dataSet);
    cout << "End of writing points on file." << endl;

    cout << "Computing the average errors on 10 runs..." << endl;
    for(int i = 0; i < 10; ++i) {
        nn.train(trainingDataSet);
        finalMSE += nn.computeError(testDataSet);
        finalMEE += computeMee(trainingSet.rescaleOutputs(nn.computeOutput(testDataSet)), testSet.dataSet);
        nn.reset();
    }
    finalMSE /= 10;
    finalMEE /= 10;

    cout << "The final MSE on the test set is " << finalMSE << endl;
    cout << "The final MEE on the test set is " << finalMEE << endl;

    cout << "end of the Loc Testing." << endl;
}

double NeuralNetworkSimulator::computeMee(const vector<vector<double>>& outputs, const vector<Data>& dataSet) const {
    double mee = 0;

    for(int i = 0; i < static_cast<int>(outputs.size()); ++i) {
        double outputX = outputs[i][0];
        double outputY = outputs[i][1];
        double targetX = dataSet[i].targets[0];
        double targetY = dataSet[i].targets[1];
        mee += sqrt(pow(outputX - targetX, 2.0) + pow(outputY - targetY, 2.0));
    }

    return (mee/outputs.size());
}

void NeuralNetworkSimulator::generateCompetition(const HyperParameters& parameters) const {

    // generate the complete dataSet (training set merged with the test set)
    DataSet dataSet;
    dataSet.dataSet = trainingSet.dataSet;
    dataSet.dataSet.insert(dataSet.dataSet.end(), testSet.dataSet.begin(), testSet.dataSet.end());

    vector<Data> competitionSet = dataSet.scale(fileManager.getCompetitionSet().dataSet);

    NeuralNetwork nn(parameters);
    nn.train(dataSet.getScaledDataSet());

    cout << "Computing the blind test results..." << endl;
    vector<vector<double>> outputs = dataSet.rescaleOutputs(nn.computeOutput(competitionSet));
    fileManager.writeCompetition(outputs);
    cout << "End of the blind test results computation." << endl;
}

/////////////////////////////////////////////////// Monk Functions //////////////////////////////////////////////////

vector<HyperParameters> NeuralNetworkSimulator::getMonkParameters() const {
    vector<HyperParameters> parameters;

    vector<int> nNeurons; nNeurons.push_back(3); nNeurons.push_back(4); nNeurons.push_back(5);
    vector<double> etas; etas.push_back(0.05);
    vector<double> alphas; alphas.push_back(0); alphas.push_back(0.2); alphas.push_back(0.5);
    vector<double> lambdas; lambdas.push_back(0); lambdas.push_back(0.02); lambdas.push_back(0.05);

    for(auto n = nNeurons.begin(); n != nNeurons.end(); ++n) {
        for(auto eta = etas.begin(); eta != etas.end(); ++eta) {
            for(auto lambda = lambdas.begin(); lambda != lambdas.end(); ++lambda) {
                for(auto alpha = alphas.begin(); alpha != alphas.end(); ++alpha) {
                    parameters.push_back(HyperParameters(monk, *n, 1000, *eta, *lambda, *alpha));
                }
            }
        }
    }

    return parameters;
}

void NeuralNetworkSimulator::generateMonkPerformance(const HyperParameters& parameters) const {
    vector<Data> trainingSet = NeuralNetworkSimulator::trainingSet.get1OfKDataSet();
    vector<Data> testSet = NeuralNetworkSimulator::testSet.get1OfKDataSet();
    vector<pair<double, double>> trainingMSE;
    vector<pair<double, double>> testMSE;
    int maxEpochs = parameters.epochs;

    cout << "Monk " << monkIndex << " Testing..." << endl;
    // save the lms errors for every epoch
    NeuralNetwork nn(parameters);
    nn.parameters.epochs = 1;
    trainingMSE.push_back(make_pair<int, double>(0, nn.computeError(trainingSet)));
    testMSE.push_back(make_pair<int, double>(0, nn.computeError(testSet)));
    for(int i = 1; i <= maxEpochs; ++i) {
        nn.train(trainingSet);
        if(i % 10 == 0) {
            trainingMSE.push_back(make_pair<int, double>(i, nn.computeError(trainingSet)));
            testMSE.push_back(make_pair<int, double>(i, nn.computeError(testSet)));
        }
        if(testMSE[testMSE.size() - 1].second == 0) break;
    }

    cout << "Writing Monk " << monkIndex << " performance on file...";
    // writing errors about training and test set
    fileManager.writeErrors(trainingMSE, "Training");
    fileManager.writeErrors(testMSE, "Test");
    cout << " end of the performance writing." << endl;
}
