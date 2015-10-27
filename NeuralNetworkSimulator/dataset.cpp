#include "dataset.h"

DataSet::DataSet() {}

vector<Data> DataSet::get1OfKDataSet() const {
    vector<Data> dataSet1OfK;

    for(auto data = dataSet.begin(); data != dataSet.end(); ++data) {
        dataSet1OfK.push_back(Data(convertTo1OfK(data->features), data->targets));
    }
    return dataSet1OfK;
}

void DataSet::computeMinMax() {
    if(minFeatures.size() != 0) {   // min and max vectors are already computed
        return;
    }

    // compute the max and min features vector
    for(int i = 0; i < static_cast<int>(dataSet[0].features.size()); ++i) {
        minFeatures.push_back(dataSet[0].features[i]);
        maxFeatures.push_back(dataSet[0].features[i]);
        for(int j = 1; j < static_cast<int>(dataSet.size()); ++j) {
            if(dataSet[j].features[i] < minFeatures[i]) {
                minFeatures[i] = dataSet[j].features[i];
            }
            if(dataSet[j].features[i] > maxFeatures[i]) {
                maxFeatures[i] = dataSet[j].features[i];
            }
        }
    }

    // compute the max and min targets vector
    for(int i = 0; i < static_cast<int>(dataSet[0].targets.size()); ++i) {
        minTargets.push_back(dataSet[0].targets[i]);
        maxTargets.push_back(dataSet[0].targets[i]);
        for(int j = 1; j < static_cast<int>(dataSet.size()); ++j) {
            if(dataSet[j].targets[i] < minTargets[i]) {
                minTargets[i] = dataSet[j].targets[i];
            }
            if(dataSet[j].targets[i] > maxTargets[i]) {
                maxTargets[i] = dataSet[j].targets[i];
            }
        }
    }
}

vector<Data> DataSet::getScaledDataSet() {
    computeMinMax();
    return scale(DataSet::dataSet);
}

vector<Data> DataSet::scale(vector<Data> dataSet) {
    vector<Data> scaledDataSet;

    computeMinMax();

    // scaling the dataSet
    for(auto data = dataSet.begin(); data != dataSet.end(); ++data) {
        Data scaledData;
        for(int i = 0; i < static_cast<int>(data->features.size()); ++i) {
            scaledData.features.push_back((data->features[i] - minFeatures[i]) / (maxFeatures[i] - minFeatures[i]));
        }

        for(int i = 0; i < static_cast<int>(data->targets.size()); ++i) {
            scaledData.targets.push_back((data->targets[i] - minTargets[i]) / (maxTargets[i] - minTargets[i]));
        }

        scaledDataSet.push_back(scaledData);
    }
    return scaledDataSet;
}

vector<vector<double>> DataSet::rescaleOutputs(const vector<vector<double>>& outputs) const {
    vector<vector<double>> rescaledOutputs;

    for(int i = 0; i < static_cast<int>(outputs.size()); ++i) {
       vector<double> rescaledOutput;
       rescaledOutput.push_back(outputs[i][0] * (maxTargets[0] - minTargets[0]) + minTargets[0]);
       rescaledOutput.push_back(outputs[i][1] * (maxTargets[1] - minTargets[1]) + minTargets[1]);

       rescaledOutputs.push_back(rescaledOutput);
    }
    return rescaledOutputs;
}


vector<double> DataSet::convertTo1OfK(const vector<double>& features) const {
   int i = 0;
   vector<double> oneOfK = vector<double>(17, 0);

   for (auto feature = features.begin(); feature != features.end(); ++feature) {
       setFeature(oneOfK, static_cast<int>(*feature), i);
       ++i;
   }
   return oneOfK;
}

void DataSet::setFeature(vector<double>& oneOfK, const int &value, const int &index) const {
   const int maxValueSum[6] = {0, 3, 6, 8, 11, 15};
   oneOfK.at(maxValueSum[index] + value - 1) = 1;
}
