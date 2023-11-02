#include "neuralnetwork.h"
#include "parser.h"
#include <cmath> // for std::exp
#include <iostream>
#include <vector> // for creating batches

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

  // Create batches for training
  const int batchSize = 100; // Modify as needed
  std::vector<Eigen::VectorXd> batchInputData;
  std::vector<Eigen::VectorXd> batchTargetOutput;

  std::cout << "starting loop for the batcj" << std::endl;
  for (int i = 0; i < images.rows(); i += batchSize) {

    batchInputData.clear();
    batchTargetOutput.clear();
    for (int j = i; j < i + batchSize && j < images.rows(); j++) {
      batchInputData.push_back(images.row(j));
      Eigen::VectorXd target = Eigen::VectorXd::Zero(10);
      target(static_cast<int>(labels(j))) = 1.0;
      batchTargetOutput.push_back(target);
    }

    // Batch Backpropagation
    nn.batchBackPropagate(batchInputData, batchTargetOutput);
  }

  // Evaluate the Neural Network after training
  ForwardPassData fpdAfterBP = nn.forwardPass(images.row(0));
  Eigen::VectorXd targetOutput = Eigen::VectorXd::Zero(10);
  targetOutput(static_cast<int>(labels(0))) = 1.0;
  Eigen::VectorXd newError = targetOutput - fpdAfterBP.activatedOutputs.back();
  std::cout << "Error after batch backpropagation: " << newError.norm()
            << std::endl;

  return 0;
}
