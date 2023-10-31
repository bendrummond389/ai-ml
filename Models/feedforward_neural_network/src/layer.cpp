#include "layer.h"
#include <iostream>

// Constructor
Layer::Layer(int inSize, int outSize, double (*act)(double),
             double (*actPrime)(double))
    : inputSize(inSize), outputSize(outSize), activationFunction(act),
      activationFunctionPrime(actPrime),
      weights(Eigen::MatrixXd::Random(inSize, outSize)),
      biases(Eigen::VectorXd::Zero(outSize)) {}

// Getter and Setter functions
int Layer::getInputSize() const { return inputSize; }

int Layer::getOutputSize() const { return outputSize; }

Eigen::MatrixXd Layer::getWeights() const { return weights; }

void Layer::setWeights(const Eigen::MatrixXd &newWeights) {
  weights = newWeights;
}

Eigen::VectorXd Layer::getBiases() const { return biases; }

void Layer::setBiases(const Eigen::VectorXd &newBiases) { biases = newBiases; }

double (*Layer::getActivationFunction())(double) { return activationFunction; }

double (*Layer::getActivationFunctionPrime())(double) {
  return activationFunctionPrime;
}

void Layer::updateWeights(const Eigen::MatrixXd &weightUpdates) {
  weights += weightUpdates.transpose();
}

// Update biases using computed updates
void Layer::updateBiases(const Eigen::VectorXd &biasUpdates) {
  biases += biasUpdates;
}