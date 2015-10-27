#ifndef NEURALNETWORKSIMULATOR_H
#define NEURALNETWORKSIMULATOR_H

#include "filemanager.h"
#include "validation.h"
#include "crossvalidation.h"

class NeuralNetworkSimulator
{
protected:
    int monkIndex = 0;
    FileManager fileManager;
    DataSet trainingSet;
    DataSet testSet;

    // Loc functions
    vector<HyperParameters> getLocParameters() const;
    void generateLocPerformance(HyperParameters parameters) ;
    double computeMee(const vector<vector<double>>& targets, const vector<Data>& dataSet) const;
    void generateCompetition(const HyperParameters& parameters) const;

    // Monk functions
    vector<HyperParameters> getMonkParameters() const;
    void generateMonkPerformance(const HyperParameters& parameters) const;

public:
    NeuralNetworkSimulator();
    NeuralNetworkSimulator(const int& monkIndex);
    void simulate(HyperParameters parameters);
    void simulate();

    void locRun(HyperParameters& parameters);     // generate the points of the plots using the first 70% of trainingSet
                                                  // as training set and the last 30% of trainingSet as validation set

    void locFoldRun(HyperParameters& parameters); // generate the points of the plots using a fold of the cross-validation as validation set
};

#endif // NEURALNETWORKSIMULATOR_H
