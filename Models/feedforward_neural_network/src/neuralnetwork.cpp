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

void NeuralNetwork::batchBackPropagate(
    const std::vector<Eigen::VectorXd> &batchInputData,
    const std::vector<Eigen::VectorXd> &batchTargetOutput) {

  // Initialize a vector to store ForwardPassData for each input in the batch
  std::vector<ForwardPassData> forwardPassResults;
  forwardPassResults.reserve(batchInputData.size());

  // Assuming the last layer of the network is accessible and we can get its
  // output size
  int outputLayerNeurons = this->layers.back().getOutputSize();

  // Matrix to store all the output errors for the batch
  Eigen::MatrixXd batchOutputErrors(outputLayerNeurons, batchInputData.size());

  // Loop through each input in the batch to perform a forward pass
  for (size_t i = 0; i < batchInputData.size(); ++i) {
    forwardPassResults.push_back(this->forwardPass(batchInputData[i]));
    // Get the activated output from the last layer
    Eigen::VectorXd activatedOutput =
        forwardPassResults[i].activatedOutputs.back();
    // Compute the output error for each sample
    batchOutputErrors.col(i) = activatedOutput - batchTargetOutput[i];
  }

  // Output the average error for logging purposes
  Eigen::VectorXd averageError = batchOutputErrors.rowwise().mean();
  std::cout << "Average Error for the batch:" << std::endl
            << averageError << std::endl;
}

Eigen::VectorXd NeuralNetwork::computeBatchOutputError(
    const std::vector<Eigen::VectorXd> &batchTargetOutput,
    const std::vector<ForwardPassData> &forwardPassResults) {

  // Initialize a vector to sum up all the errors.
  Eigen::VectorXd sumError = Eigen::VectorXd::Zero(
      forwardPassResults.front().activatedOutputs.back().size());

  // Loop through all the forward pass results and accumulate errors.
  for (size_t i = 0; i < forwardPassResults.size(); ++i) {
    sumError +=
        forwardPassResults[i].activatedOutputs.back() - batchTargetOutput[i];
  }

  // Calculate the average error by dividing by the number of samples.
  Eigen::VectorXd averageError =
      sumError / static_cast<double>(forwardPassResults.size());

  return averageError;
}

std::vector<Eigen::MatrixXd> NeuralNetwork::propagateBatchErrorBackward(
    const Eigen::MatrixXd &batchOutputErrors,
    const std::vector<ForwardPassData> &batchFpData) {
  // This vector will hold a matrix for each layer's errors in the batch.
  std::vector<Eigen::MatrixXd> batchLayerErrors;

  // Initialize the first set of errors (for the output layer) directly from the
  // output errors.
  batchLayerErrors.push_back(batchOutputErrors);

  // Iterate backwards through the layers, starting from the second-to-last
  // layer.
  for (int i = layers.size() - 2; i >= 0; --i) {
    // The derivative matrix for the current layer has the same dimensions as
    // the layer's output.
    Eigen::MatrixXd derivatives =
        batchFpData.front()
            .rawOutputs[i + 1]
            .unaryExpr([&](double val) {
              return layers[i + 1].getActivationFunctionPrime()(val);
            })
            .replicate(1, batchFpData.size());

    // The error for the current layer is calculated using the weighted sum of
    // the previous layer's errors.
    Eigen::MatrixXd currentError =
        (layers[i + 1].getWeights().transpose() * batchLayerErrors.back())
            .cwiseProduct(derivatives);

    // Since we're dealing with a batch, we sum the errors across all samples in
    // the batch.
    Eigen::MatrixXd summedError = currentError.rowwise().sum();

    // Add the summed error matrix for this layer to our vector.
    batchLayerErrors.push_back(summedError);
  }

  // Reverse the order of the layer errors to match the forward pass.
  std::reverse(batchLayerErrors.begin(), batchLayerErrors.end());

  return batchLayerErrors;
}

// Eigen::VectorXd
// NeuralNetwork::computeOutputError(const Eigen::VectorXd &targetOutput,
//                                   const ForwardPassData &fpData) {
//   return fpData.activatedOutputs.back() - targetOutput;
// }