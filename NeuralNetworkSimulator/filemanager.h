#ifndef DATAREADER_H
#define DATAREADER_H

#include <string>
#include <sstream>
#include <fstream>
#include <ctime>

#include "dataSet.h"

using namespace std;

class FileManager {
protected:
    int monkIndex = 0;

    // monk functions
    DataSet getMonkDataSet(const string& extension) const;
    string getMonkDataPath(const string& extension) const;
    string getMonkPerformancePath(const string& type) const;

    // loc functions
    DataSet getLocDataSet(const string& extension) const;
    string getLocPerformancePath(const string& type) const;

    string today() const;
public:
    FileManager();
    FileManager(int monkIndex);
    FileManager(const FileManager& fileManager);

    DataSet getTrainingSet() const;
    DataSet getTestSet() const;
    DataSet getCompetitionSet() const;

    void writeErrors(const vector<pair<double, double>>& errors, const string& type) const;
    void writePoints(const vector<vector<double>>& points, const vector<Data>& dataSet) const;
    void writeCompetition(const vector<vector<double>>& outputs) const;
};

#endif // DATAREADER_H
