#ifndef FOLDS_H
#define FOLDS_H

#include <vector>

#include "data.h"

class Folds
{
public:
    int k;
    vector<vector<Data>> folds;

    Folds();
    Folds(const Folds& folds);
    Folds(const vector<Data>& dataSet, const int& k);

    vector<vector<Data>> getFolds(const vector<Data>& dataSet) const;
    vector<Data> getTrainingSet(const int& index) const;
};

#endif // FOLDS_H
