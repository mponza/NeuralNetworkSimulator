QMAKE_CXXFLAGS += -std=c++11

QT       += core

QT       -= gui

TARGET = NeuralNetworkSimulator
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    data.cpp \
    random.cpp \
    neuron.cpp \
    neuralnetwork.cpp \
    validation.cpp \
    hyperparameters.cpp \
    crossvalidation.cpp \
    neuralnetworksimulator.cpp \
    dataset.cpp \
    filemanager.cpp \
    modelselection.cpp \
    folds.cpp \

HEADERS += \
    data.h \
    random.h \
    neuron.h \
    neuralnetwork.h \
    validation.h \
    hyperparameters.h \
    crossvalidation.h \
    neuralnetworksimulator.h \
    dataset.h \
    filemanager.h \
    modelselection.h \
    folds.h \
