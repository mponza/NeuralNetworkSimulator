#ifndef CROSSVALIDATION_H
#define CROSSVALIDATION_H

#include "folds.h"
#include "modelselection.h"
#include "neuralnetwork.h"

class CrossValidation : public ModelSelection
{                                               // k-folds Cross Validation
                                                // it makes k + 1 folds: the last one is used for computing
                                                // the maximum number of epochs of the best hyper-parameters
                                                // configuration found
public:
    int n;                                      // number of elements returned by the validate function
    bool earlyStopping;                         // use the early-stopping in the model selection to compute the best number of epochs
    Folds folds;

    CrossValidation();
    CrossValidation(const CrossValidation& crossValidation);
    CrossValidation(const vector<Data>& dataSet, const int& k, const int& n, const bool& earlyStopping);

    // k-folds cross validation
    vector<HyperParameters> validate(const vector<HyperParameters>& parameters) const;

    // return the vector with the n best HyperParameters in [parametersMSE, (newParameter, newMSE)]
    vector<pair<HyperParameters, double>> bestHyperParameters(vector<pair<HyperParameters, double>> parametersMSE,
                                                                 const HyperParameters& newParameters, const double& newMSE) const;
};

#endif // CROSSVALIDATION_H
