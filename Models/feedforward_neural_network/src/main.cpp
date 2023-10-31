#include "neuralnetwork.h"
#include "parser.h"
#include <cmath> // for std::exp
#include <iostream>

int main() {
  // 1. Data Parsing
  const std::string trainImagesPath = "data/train/train-images-idx3-ubyte";
  const std::string trainLabelsPath = "data/train/train-labels-idx1-ubyte";
  Eigen::MatrixXd images = parseImages(trainImagesPath);
  Eigen::VectorXd labels = parseLabels(trainLabelsPath);
  std::cout << "Parsed " << images.rows() << " images and " << labels.size()
            << " labels." << std::endl;

  // 2. Neural Network Initialization
  NeuralNetwork nn(images.cols());
  auto sigmoid = [](double z) -> double { return 1.0 / (1.0 + exp(-z)); };
  auto sigmoidPrime = [](double z) -> double {
    double sig = 1.0 / (1.0 + exp(-z));
    return sig * (1.0 - sig);
  };
  nn.addLayer(128, sigmoid, sigmoidPrime);
  nn.addLayer(10, sigmoid, sigmoidPrime);
  std::cout << "Neural Network initialized with a hidden layer of 128 neurons "
               "and output layer of 10 neurons"
            << std::endl;

  // 3. Initial Forward Pass
  ForwardPassData fpd = nn.forwardPass(images.row(0));

  Eigen::VectorXd targetOutput = Eigen::VectorXd::Zero(10);
  std::cout << "Expected value: " << labels(0) << std::endl;
  targetOutput(static_cast<int>(labels(0))) =
      1.0; // One-hot encoding the target label

  Eigen::VectorXd initialError = targetOutput - fpd.activatedOutputs.back();
  std::cout << "Initial error: " << initialError.norm() << std::endl;

  // 4. Backpropagation
  nn.backpropagate(targetOutput, fpd);

  // 5. Forward Pass after Backpropagation
  ForwardPassData fpdAfterBP = nn.forwardPass(images.row(0));
  Eigen::VectorXd newError = targetOutput - fpdAfterBP.activatedOutputs.back();
  std::cout << "Error after backpropagation: " << newError.norm() << std::endl;

  return 0;
}
