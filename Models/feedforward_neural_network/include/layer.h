#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>
#include <vector>

class Layer {
private:
  int inputSize;
  int outputSize;
  Eigen::MatrixXd weights;
  Eigen::VectorXd biases;
  double (*activationFunction)(double);
  double (*activationFunctionPrime)(double);

public:
  // Constructor
  Layer(int inSize, int outSize, double (*act)(double),
        double (*actPrime)(double));

  // Getter and Setter functions
  int getInputSize() const;
  int getOutputSize() const;
  Eigen::MatrixXd getWeights() const;
  void setWeights(const Eigen::MatrixXd &newWeights);
  Eigen::VectorXd getBiases() const;
  void setBiases(const Eigen::VectorXd &newBiases);
  double (*getActivationFunction())(double);
  double (*getActivationFunctionPrime())(double);
  void updateWeights(const Eigen::MatrixXd &weightUpdates);
  void updateBiases(const Eigen::VectorXd &biasUpdates);
};

#endif // LAYER_H
