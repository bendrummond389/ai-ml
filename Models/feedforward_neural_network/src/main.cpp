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

  for (int img_idx = 0; img_idx < 5; ++img_idx) {
    Eigen::MatrixXd image = images.row(img_idx);
    image.resize(28, 28); 

    for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 28; ++j) {
        std::cout << (image(i, j) > 0.5 ? "#" : " ");
      }
      std::cout << std::endl;
    }
    std::cout << "-----------------------------------------" << std::endl;
  }

  return 0;
}
