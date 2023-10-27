#pragma once

#include <Eigen/Dense>
#include <vector>

class NeuralNetwork {
public:
  NeuralNetwork(int inputNeurons);
  void addLayer(int neurons, double (*activation)(double));

private:
  struct Layer {
    int inputSize;
    int outputSize;
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    double (*activationFunction)(double);
  };

  int inputNeurons;
  std::vector<Layer> layers;
};
