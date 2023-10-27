#include <Eigen/Dense>
#include <neuralnetwork.h>
#include <vector>

NeuralNetwork::NeuralNetwork(int inputNeurons) : inputNeurons(inputNeurons) {}

void NeuralNetwork::addLayer(int neurons, double (*activation)(double)) {
  Layer newLayer;

  if (layers.empty()) {
    newLayer.inputSize = inputNeurons;
  } else {
    newLayer.inputSize = layers.back().outputSize;
  }

  newLayer.outputSize = neurons;
  newLayer.activationFunction = activation;

  newLayer.weights =
      Eigen::MatrixXd::Random(newLayer.inputSize, newLayer.outputSize);
  newLayer.biases = Eigen::VectorXd::Zero(newLayer.outputSize);

  layers.push_back(newLayer);
}