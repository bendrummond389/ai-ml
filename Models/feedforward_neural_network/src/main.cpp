#include "neuralnetwork.h"
#include "parser.h"
#include <iostream>

int main() {

  const std::string trainImagesPath = "data/train/train-images-idx3-ubyte";
  const std::string trainLabelsPath = "data/train/train-labels-idx1-ubyte";

  Eigen::MatrixXd images = parseImages(trainImagesPath);
  Eigen::VectorXd labels = parseLabels(trainLabelsPath);

  std::cout << "Parsed " << images.rows() << " images and " << labels.size()
            << " labels." << std::endl;

  NeuralNetwork nn(images.cols());

  nn.addLayer(128, [](double z) -> double { return 1.0 / (1.0 + exp(-z)); });

  std::cout << "Neural Network initialized with a hidden layer of 128 neurons."
            << std::endl;

  // Test forward pass using the first image in the dataset
  Eigen::VectorXd sample = images.row(0);
  Eigen::VectorXd output = nn.forwardPass(sample);

  std::cout << "Output from forward pass for the first sample: \n"
            << output << std::endl;

  return 0;
}
