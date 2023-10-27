#include <fstream>
#include <stdexcept>
#include <Eigen/Dense>

Eigen::VectorXd parseLabels(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);

  if (file.is_open()) {
    int magicNumber = 0;
    int numLabels = 0;

    file.read((char *)&magicNumber, sizeof(magicNumber));
    file.read((char *)&numLabels, sizeof(numLabels));

    magicNumber = __builtin_bswap32(magicNumber);
    numLabels = __builtin_bswap32(numLabels);

    Eigen::VectorXd labels(numLabels);

    for (int i = 0; i < numLabels; ++i) {
      unsigned char label = 0;
      file.read((char *)&label, sizeof(label));
      labels(i) = label;
    }

    file.close();
    return labels;
  } else {
    throw std::runtime_error("Cannot open file: " + filename);
  }
}

Eigen::MatrixXd parseImages(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);

  if (file.is_open()) {
    int magicNumber = 0;
    int numImages = 0;
    int numRows = 0;
    int numCols = 0;

    file.read((char *)&magicNumber, sizeof(magicNumber));
    file.read((char *)&numImages, sizeof(numImages));
    file.read((char *)&numRows, sizeof(numRows));
    file.read((char *)&numCols, sizeof(numCols));

    magicNumber = __builtin_bswap32(magicNumber);
    numImages = __builtin_bswap32(numImages);
    numRows = __builtin_bswap32(numRows);
    numCols = __builtin_bswap32(numCols);

    Eigen::MatrixXd images(numImages, numRows * numCols);

    for (int i = 0; i < numImages; ++i) {
      for (int j = 0; j < numRows * numCols; ++j) {
        unsigned char pixel = 0;
        file.read((char *)&pixel, sizeof(pixel));
        images(i, j) = pixel / 255.0; // Normalize pixel value to [0, 1]
      }
    }

    file.close();
    return images;
  } else {
    throw std::runtime_error("Cannot open file: " + filename);
  }
}
