#include "neuralnetwork.h"
#include "layer.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>

// Constructor: Initializes the neural network with the given number of input
// neurons.
NeuralNetwork::NeuralNetwork(int inputNeurons) : inputNeurons(inputNeurons) {}

// Adds a new layer to the neural network.
// It determines the number of input neurons for the new layer based on the
// previous layer or the input size.
void NeuralNetwork::addLayer(int neurons, double (*activation)(double),
                             double (*activationPrime)(double)) {
  int inSize = layers.empty() ? inputNeurons : layers.back().getOutputSize();
  layers.emplace_back(inSize, neurons, activation, activationPrime);
}

// Performs a forward pass through the network using the given input data.
// It computes the raw and activated outputs for each layer.
ForwardPassData NeuralNetwork::forwardPass(const Eigen::VectorXd &inputData) {
  ForwardPassData fpData;
  fpData.input = inputData;

  Eigen::VectorXd currentOutput = inputData;

  for (Layer &layer : layers) {
    // Compute raw output for the current layer.
    Eigen::VectorXd rawOutput =
        layer.getWeights().transpose() * currentOutput + layer.getBiases();
    // Apply activation function.
    Eigen::VectorXd activatedOutput = rawOutput.unaryExpr(
        [&](double val) { return layer.getActivationFunction()(val); });

    fpData.rawOutputs.push_back(rawOutput);
    fpData.activatedOutputs.push_back(activatedOutput);

    currentOutput = activatedOutput;
  }

  return fpData;
}

// Prints the activated outputs of each layer for the given forward pass data.
void NeuralNetwork::printActivations(const ForwardPassData &fpData) {
  for (size_t i = 0; i < fpData.activatedOutputs.size(); ++i) {
    std::cout << "Layer " << (i + 1) << " Activations:" << std::endl;
    std::cout << fpData.activatedOutputs[i].transpose() << std::endl;
    std::cout << "---------------------------------" << std::endl;
  }
}

// Computes the output error by comparing the network's final output with the
// target output.
Eigen::VectorXd
NeuralNetwork::computeOutputError(const Eigen::VectorXd &targetOutput,
                                  const ForwardPassData &fpData) {
  return fpData.activatedOutputs.back() - targetOutput;
}

// Backpropagates the output error through the network to compute errors for
// each layer.
std::vector<Eigen::VectorXd>
NeuralNetwork::propagateErrorBackward(const Eigen::VectorXd &outputError,
                                      const ForwardPassData &fpData) {
  std::vector<Eigen::VectorXd> layerErrors;
  layerErrors.push_back(outputError);

  for (int i = layers.size() - 1; i > 0; --i) {
    // Compute the derivative using the activation function's derivative.
    Eigen::VectorXd derivative =
        fpData.rawOutputs[i].unaryExpr([&](double val) {
          return layers[i].getActivationFunctionPrime()(val);
        });

    // Compute the error for the current layer.
    Eigen::VectorXd currentError =
        layers[i].getWeights() * layerErrors.back().cwiseProduct(derivative);
    layerErrors.push_back(currentError);
  }

  std::reverse(layerErrors.begin(), layerErrors.end());
  return layerErrors;
}

// Updates the weights and biases of each layer based on the computed errors and
// a learning rate.
void NeuralNetwork::updateWeightsAndBiases(
    const std::vector<Eigen::VectorXd> &layerErrors,
    const ForwardPassData &fpData) {
  double learningRate = 0.1;
  Eigen::VectorXd previousActivatedOutput = fpData.input;

  for (size_t i = 0; i < layers.size(); ++i) {
    // Compute weight updates using the gradient descent rule.
    Eigen::MatrixXd weightUpdates =
        layerErrors[i] * previousActivatedOutput.transpose();

    layers[i].updateWeights(-learningRate * weightUpdates);
    layers[i].updateBiases(-learningRate * layerErrors[i]);

    previousActivatedOutput = fpData.activatedOutputs[i];
  }
}

// Combines the forward pass, error computation, backpropagation, and weight
// updates into a single backpropagation step.
void NeuralNetwork::backpropagate(const Eigen::VectorXd &targetOutput,
                                  const ForwardPassData &fpData) {
  Eigen::VectorXd outputError = computeOutputError(targetOutput, fpData);
  std::vector<Eigen::VectorXd> layerErrors =
      propagateErrorBackward(outputError, fpData);
  updateWeightsAndBiases(layerErrors, fpData);
}
