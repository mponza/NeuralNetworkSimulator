#include "data.h"

Data::Data() {}

Data::Data(const Data& data) : Data(data.features, data.targets) {}

Data::Data(const vector<double>& features, const vector<double>& targets) {
    Data::features = features;
    Data::targets = targets;
}
