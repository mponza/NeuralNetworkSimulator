#include <iostream>
#include <getopt.h>

#include "neuralnetworksimulator.h"

using namespace std;

bool parametersSetted(const HyperParameters& parameters) {
    if(parameters.nNeurons != 0 || parameters.epochs != 0 || parameters.eta != 0 || parameters.lambda != 0 || parameters.alpha != 0) {
        return true;
    } else {
        return false;
    }
}

int main(int argc, char ** argv)
{
    HyperParameters parameters;
    int monkIndex = 0;
    bool run = false;
    bool fold = false;

    static int verbose_flag;
    int c;

    while (1) {
        static struct option long_options[] =
        {
            {"verbose", no_argument, &verbose_flag, 1},
            {"brief",   no_argument, &verbose_flag, 0},
            {"monk", required_argument, 0, 'm'},
            {"loc", no_argument, 0, 'l'},
            {"run", no_argument, 0, 'r'},
            {"fold", no_argument, 0, 'f'},
            {"nNeurons", required_argument, 0, 'n'},
            {"epochs", required_argument, 0, 'e'},
            {"eta", required_argument, 0, 't'},
            {"lambda", required_argument, 0, 'd'},
            {"alpha", required_argument, 0, 'a'},
            {0, 0, 0, 0}
        };
        int option_index = 0;

        c = getopt_long(argc, argv, "mlvcn:e:t:l:a:s", long_options, &option_index);

        if (c == -1) break;
        switch(c) {
            case 'm':
                parameters = HyperParameters(monk);
                monkIndex = atoi(optarg);
                break;
            case 'l':
                parameters = HyperParameters(loc);
                break;
            case 'r':
                run = true;
                break;
            case 'f':
                fold = true;
                break;
            case 'n':
                parameters.nNeurons = atoi(optarg);
                break;
            case 'e':
                parameters.epochs = atoi(optarg);
                break;
            case 't':
                parameters.eta = atof(optarg);
                break;
            case 'd':
                parameters.lambda = atof(optarg);
                break;
            case 'a':
                parameters.alpha = atof(optarg);
                break;
        }
    }

    ////////////////////////////////////////////////// Monk Task ///////////////////////////////////////////////////

    if(parameters.task == monk) {
        if(monkIndex <= 0 || monkIndex > 3) {
            cout << "The monk index must be a value in [1; 3]." << endl;
            return 1;
        }

        NeuralNetworkSimulator nns(monkIndex);

        if(parametersSetted(parameters) == true) {
            nns.simulate(parameters);
        } else  {
            nns.simulate();
        }
    }

    ////////////////////////////////////////////////// Loc Task ///////////////////////////////////////////////////

    if(parameters.task == loc) {
        NeuralNetworkSimulator nns;

        if(run) {
            if(parametersSetted(parameters) == true) nns.locRun(parameters);
            else cout << "With --run you have to specify the neural network parameters." << endl;
        } else {
            if (fold) {
                if(parametersSetted(parameters) == true) nns.locFoldRun(parameters);
                else cout << "With --fold you have to specify the neural network parameters." << endl;
            } else {
                if(parametersSetted(parameters) == true) nns.simulate(parameters);
                else nns.simulate();
            }
        }
    }

    return 0;
}
