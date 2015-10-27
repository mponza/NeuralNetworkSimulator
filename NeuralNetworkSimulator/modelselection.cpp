#include "modelselection.h"

ModelSelection::ModelSelection() {}

void ModelSelection::printResults(const HyperParameters& parameters, const double& empiricalRisk) const {
    cout << "nNeurons " << parameters.nNeurons << "\t";
    cout << "epochs " << parameters.epochs << "\t";
    cout << "eta " << parameters.eta << "\t";
    cout << "alpha " << parameters.alpha << "\t";
    cout << "lambda " << parameters.lambda << "\t";
    cout << "MSE " << empiricalRisk << endl;
    cout.flush();
}
