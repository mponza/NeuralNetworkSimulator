#include "folds.h"

Folds::Folds() {}

Folds::Folds(const Folds& folds) {
    Folds::k = folds.k;
    Folds::folds = folds.folds;
}

Folds::Folds(const vector<Data>& dataSet, const int& k) {
    Folds::k = k;
    Folds::folds = getFolds(dataSet);
}


vector<vector<Data>> Folds::getFolds(const vector<Data>& dataSet) const {
    vector<vector<Data>> folds;
    vector<Data> fold;
    int n = dataSet.size() / k;         // average number of elements in every fold
    int r = dataSet.size() - n * k;     // number of remaining elements

    int j = 0;
    for(int i = 0; i < k - 1; ++i) {
        // put n elements in fold
        for(int l = 0; l < n; ++l) {
            fold.push_back(dataSet[j]);
            ++j;
        }
        if (r != 0) {   // put one more element in fold
            --r;
            fold.push_back(dataSet[j]);
            ++j;
        }
        folds.push_back(fold);
        fold.clear();
    }

    // put all the remaining elements in the k- th fold
    while (j < static_cast<int>(dataSet.size())) {
        fold.push_back(dataSet[j]);
        ++j;
    }
    folds.push_back(fold);
    return folds;
}

vector<Data> Folds::getTrainingSet(const int& index) const {
    vector<Data> trainingSet;

    for(int i = 0; i < k; ++i) {
        if(i != index) {
            for(int j = 0; j < static_cast<int>(folds[i].size()); ++j) {
                trainingSet.push_back(folds[i][j]);
            }
        }
    }
    return trainingSet;
}
