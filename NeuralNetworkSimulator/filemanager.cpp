#include "filemanager.h"

FileManager::FileManager() {}

FileManager::FileManager(int monkIndex) {
    FileManager::monkIndex = monkIndex;
}

FileManager::FileManager(const FileManager& fileManager) : FileManager(fileManager.monkIndex) {}

DataSet FileManager::getTrainingSet() const {
    if(monkIndex != 0) {
        return getMonkDataSet(".train");
    } else {
        DataSet trainingSet = getLocDataSet("TR.csv");
        int trainingElements = trainingSet.dataSet.size() * 0.7;
        vector<Data> dataSet = vector<Data>(trainingSet.dataSet.begin(), trainingSet.dataSet.begin() + trainingElements);
        trainingSet.dataSet = dataSet;
        return trainingSet;
    }
}

DataSet FileManager::getTestSet() const {
    if(monkIndex != 0) {
        return getMonkDataSet(".test");
    } else {
        DataSet testSet = getLocDataSet("TR.csv");
        int trainingElements = testSet.dataSet.size() * 0.7;
        vector<Data> dataSet = vector<Data>(testSet.dataSet.begin() + trainingElements, testSet.dataSet.end());
        testSet.dataSet = dataSet;
        return testSet;
    }
}

DataSet FileManager::getCompetitionSet() const {
    return getLocDataSet("TS.csv");
}

DataSet FileManager::getMonkDataSet(const string& extension) const {
    DataSet dataSet;
    string path = getMonkDataPath(extension);

    ifstream ifs;
    ifs.open(path.c_str(), ifstream::in);

    if (ifs.is_open()) {
        while(ifs.good()) {
            string line;
            getline(ifs, line, '\n');
            if (line.size() != 0) {

                stringstream ss(line);
                vector<double> features;
                vector<double> target;
                double d = 0;
                if (ss >> d) {  // get the target
                    target.push_back(d);
                }
                while(ss >> d) { // get the features
                    features.push_back(d);
                }
                dataSet.dataSet.push_back(Data(features, target));
            }
        }
    } else {
        throw string("Error in getting data from file " + path);
    }
    ifs.close();
    return dataSet;
}

DataSet FileManager::getLocDataSet(const string& extension) const {
    DataSet dataSet;
    string path = "../data/data-AA1-2013-CUP/LOC-UNIPI-" + extension;

    ifstream ifs;
    ifs.open(path.c_str(), ifstream::in);

    if(ifs.is_open()) {
        if(ifs.good()) {
            string line;
            // skip comments lines (first 10 rows)
            for (int i = 0; i < 10; ++i) {
                getline(ifs, line);
            }

            while(ifs.good()) {
                getline(ifs, line, '\n');
                istringstream iss(line);
                string token;

                // skip id
                getline(iss, token, ',');

                // read features
                vector<double> features;
                vector<double> targets;
                for(int i = 0; i < 5; ++i) {
                    getline(iss, token, ',');
                    features.push_back(atof(token.c_str()));

                }

                // read targets
                for(int i = 0; i < 2; ++i) {
                    getline(iss, token, ',');
                    targets.push_back(atof(token.c_str()));
                }
                dataSet.dataSet.push_back(Data(features, targets));
            }
        }
    }
    dataSet.dataSet.pop_back(); // the last one element has all 0 (because \n at the end of the csv file)
    ifs.close();
    return dataSet;
}

string FileManager::getMonkDataPath(const string& extension) const {
    stringstream ss;
    ss << "../data/monk/monks-" << monkIndex;
    return ss.str().append(extension);
}

void FileManager::writeErrors(const vector<pair<double, double>>& errors, const string& type) const {
    ofstream writer;
    string path;
    if(monkIndex != 0) {
        path = getMonkPerformancePath(type);
    } else {
        path = getLocPerformancePath(type);
    }
    writer.open(path.c_str());
    for(auto pair = errors.begin(); pair != errors.end(); ++pair) {
        double first = pair->first;
        double second = pair->second;

        writer << first << " " << second << endl;
    }
    writer.close();
}

string FileManager::getMonkPerformancePath(const string& type) const {
    stringstream monk;
    monk << "../results/monk" << monkIndex << type << ".txt";
    return monk.str();
}

string FileManager::getLocPerformancePath(const string& type) const {
    stringstream monk;
    monk << "../results/loc" << type << ".txt";
    return monk.str();
}

void FileManager::writePoints(const vector<vector<double>>& points, const vector<Data>& dataSet) const {
    ofstream predictions;
    ofstream reals;
    predictions.open("../results/pointsPredicted.txt");
    reals.open("../results/pointsReal.txt");
    for(int i = 0; i < static_cast<int>(points.size()); ++i) {
        predictions << i << " " << points[i][0] << " " << points[i][1] << endl;
        Data data = dataSet[i];
        reals << i << " " << data.targets[0] << " " << data.targets[1] << endl;
    }
    predictions.close();
    reals.close();
}

void FileManager::writeCompetition(const vector<vector<double>>& outputs) const {
    ofstream writer;
    writer.open("../results/locCompetitionMarcoPonza.csv");
    writer << "# Marco Ponza" << endl << "# M4rc0" << endl << "# LOC-UNIPI - AA1 2013 CUP v1" << endl << " # " << today() << endl;
    for(int i = 0; i < static_cast<int>(outputs.size()); ++i) {
        double x = outputs[i][0];
        double y = outputs[i][1];

        writer << i + 1 << "," << x << "," << y << endl;
    }
    writer.close();
    cout.flush();
}

string FileManager::today() const {
    time_t t = time(0);
    struct tm * now = localtime(&t);
    stringstream date;
    date << now->tm_mday << "/" << (now->tm_mon + 1) << "/" << (now->tm_year + 1900);
    return date.str();
}
