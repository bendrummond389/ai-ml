#pragma once

#include <Eigen/Dense>
#include <layer.h>
#include <vector>
struct ForwardPassData {
  Eigen::VectorXd input;
  std::vector<Eigen::VectorXd> rawOutputs;
  std::vector<Eigen::VectorXd> activatedOutputs;
};

class NeuralNetwork {
public:
  NeuralNetwork(int inputNeurons);
  void addLayer(int neurons, double (*activation)(double),
                double (*activationPrime)(double));
  ForwardPassData forwardPass(const Eigen::VectorXd &inputData);
  void printActivations(const ForwardPassData &fpData);
  Eigen::VectorXd computeOutputError(const Eigen::VectorXd &targetOutput,
                                     const ForwardPassData &fpData);
  std::vector<Eigen::VectorXd>
  propagateErrorBackward(const Eigen::VectorXd &outputError,
                         const ForwardPassData &fpData);
  void updateWeightsAndBiases(const std::vector<Eigen::VectorXd> &layerErrors,
                              const ForwardPassData &fpData);
  void backpropagate(const Eigen::VectorXd &targetOutput,
                     const ForwardPassData &fpData);
  void
  batchBackPropagate(const std::vector<Eigen::VectorXd> &batchInputData,
                     const std::vector<Eigen::VectorXd> &batchTargetOutput);

  Eigen::VectorXd computeBatchOutputError(
      const std::vector<Eigen::VectorXd> &batchTargetOutput,
      const std::vector<ForwardPassData> &forwardPassResults);
  std::vector<Eigen::MatrixXd>
  propagateBatchErrorBackward(const Eigen::MatrixXd &batchOutputErrors,
                              const std::vector<ForwardPassData> &batchFpData);

private:
  int inputNeurons;
  std::vector<Layer> layers;
};
