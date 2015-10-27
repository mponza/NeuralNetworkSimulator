#ifndef DATASET_H
#define DATASET_H

#include "data.h"

class DataSet
{
protected:
    // vectors used for scale dataSet and rescale the neural network outputs (regression task)
    vector<double> minFeatures;
    vector<double> maxFeatures;
    vector<double> minTargets;
    vector<double> maxTargets;

    vector<double> convertTo1OfK(const vector<double>& features) const;
    void setFeature(vector<double>& oneOfK, const int& value, const int& index) const;

    void computeMinMax();
public:
    vector<Data> dataSet;

    DataSet();

    vector<Data> get1OfKDataSet() const;

    vector<Data> getScaledDataSet();
    vector<Data> scale(vector<Data> dataSet);

    vector<vector<double>> rescaleOutputs(const vector<vector<double>>& outputs) const;
};

#endif // DATASET_H
