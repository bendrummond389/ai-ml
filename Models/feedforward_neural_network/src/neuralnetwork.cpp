#include "neuralnetwork.h"
#include "layer.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>

NeuralNetwork::NeuralNetwork(int inputNeurons) : inputNeurons(inputNeurons) {}

void NeuralNetwork::addLayer(int neurons, double (*activation)(double),
                             double (*activationPrime)(double)) {
  int inSize = layers.empty() ? inputNeurons : layers.back().getOutputSize();
  layers.emplace_back(inSize, neurons, activation, activationPrime);
}

ForwardPassData NeuralNetwork::forwardPass(const Eigen::VectorXd &inputData) {
  ForwardPassData fpData;
  fpData.input = inputData; // Add this line

  Eigen::VectorXd currentOutput = inputData;

  for (Layer &layer : layers) {
    Eigen::VectorXd rawOutput =
        layer.getWeights().transpose() * currentOutput + layer.getBiases();
    Eigen::VectorXd activatedOutput = rawOutput.unaryExpr(
        [&](double val) { return layer.getActivationFunction()(val); });

    fpData.rawOutputs.push_back(rawOutput);
    fpData.activatedOutputs.push_back(activatedOutput);

    currentOutput = activatedOutput;
  }

  return fpData;
}

void NeuralNetwork::printActivations(const ForwardPassData &fpData) {
  for (size_t i = 0; i < fpData.activatedOutputs.size(); ++i) {
    std::cout << "Layer " << (i + 1) << " Activations:" << std::endl;
    std::cout << fpData.activatedOutputs[i].transpose() << std::endl;
    std::cout << "---------------------------------" << std::endl;
  }
}

Eigen::VectorXd
NeuralNetwork::computeOutputError(const Eigen::VectorXd &targetOutput,
                                  const ForwardPassData &fpData) {
  return fpData.activatedOutputs.back() - targetOutput;
}

std::vector<Eigen::VectorXd>
NeuralNetwork::propagateErrorBackward(const Eigen::VectorXd &outputError,
                                      const ForwardPassData &fpData) {
  std::vector<Eigen::VectorXd> layerErrors;
  layerErrors.push_back(outputError);

  for (int i = layers.size() - 1; i > 0; --i) {
    Eigen::VectorXd derivative =
        fpData.rawOutputs[i].unaryExpr([&](double val) {
          return layers[i].getActivationFunctionPrime()(val);
        });

    Eigen::VectorXd currentError =
        layers[i].getWeights() * layerErrors.back().cwiseProduct(derivative);
    layerErrors.push_back(currentError);
  }

  std::reverse(layerErrors.begin(), layerErrors.end());
  return layerErrors;
}

void NeuralNetwork::updateWeightsAndBiases(
    const std::vector<Eigen::VectorXd> &layerErrors,
    const ForwardPassData &fpData) {
  double learningRate = 0.1;
  Eigen::VectorXd previousActivatedOutput = fpData.input;

  for (size_t i = 0; i < layers.size(); ++i) {
    Eigen::MatrixXd weightUpdates =
        layerErrors[i] * previousActivatedOutput.transpose();

    std::cout << "weightUpdates dimensions: " << weightUpdates.rows() << " x "
              << weightUpdates.cols() << std::endl;

    layers[i].updateWeights(-learningRate * weightUpdates);
    layers[i].updateBiases(-learningRate * layerErrors[i]);

    previousActivatedOutput = fpData.activatedOutputs[i];
    std::cout << "next previousActivatedOutput dimensions: "
              << previousActivatedOutput.rows() << " x "
              << previousActivatedOutput.cols() << std::endl;
  }
}

void NeuralNetwork::backpropagate(const Eigen::VectorXd &targetOutput,
                                  const ForwardPassData &fpData) {
  Eigen::VectorXd outputError = computeOutputError(targetOutput, fpData);
  std::vector<Eigen::VectorXd> layerErrors =
      propagateErrorBackward(outputError, fpData);
  updateWeightsAndBiases(layerErrors, fpData);
}