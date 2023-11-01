// The Layer class represents a layer in a neural network.
// It contains information about the input and output sizes, weights, biases,
// and activation functions.

#include "layer.h"
#include <iostream>

// Constructor: Initializes a Layer object with specified input and output
// sizes, activation functions, and initializes weights with random values and
// biases with zeros.
Layer::Layer(int inSize, int outSize, double (*act)(double),
             double (*actPrime)(double))
    : inputSize(inSize), outputSize(outSize), activationFunction(act),
      activationFunctionPrime(actPrime),
      weights(Eigen::MatrixXd::Random(inSize, outSize)),
      biases(Eigen::VectorXd::Zero(outSize)) {}

// Returns the size of input for the layer.
int Layer::getInputSize() const { return inputSize; }

// Returns the size of output for the layer.
int Layer::getOutputSize() const { return outputSize; }

// Returns the weights matrix of the layer.
Eigen::MatrixXd Layer::getWeights() const { return weights; }

// Sets the weights matrix of the layer to the provided value.
void Layer::setWeights(const Eigen::MatrixXd &newWeights) {
  weights = newWeights;
}

// Returns the biases vector of the layer.
Eigen::VectorXd Layer::getBiases() const { return biases; }

// Sets the biases vector of the layer to the provided value.
void Layer::setBiases(const Eigen::VectorXd &newBiases) { biases = newBiases; }

// Returns the activation function used in the layer.
double (*Layer::getActivationFunction())(double) { return activationFunction; }

// Returns the derivative of the activation function used in the layer.
double (*Layer::getActivationFunctionPrime())(double) {
  return activationFunctionPrime;
}

// Updates the weights matrix using the provided updates. Transposes the updates
// before applying.
void Layer::updateWeights(const Eigen::MatrixXd &weightUpdates) {
  weights += weightUpdates.transpose();
}

// Updates the biases vector using the provided updates.
void Layer::updateBiases(const Eigen::VectorXd &biasUpdates) {
  biases += biasUpdates;
}
