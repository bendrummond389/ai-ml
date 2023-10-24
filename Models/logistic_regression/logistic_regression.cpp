#include <Eigen/Dense>
#include <cmath>
#include <iostream>

class LogisticRegression {

public:
  // Constructors and Destructors
  LogisticRegression();
  ~LogisticRegression();

  // Data Handling
  void input(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);

  // Core Computation Methods
  Eigen::VectorXd sigmoid(const Eigen::VectorXd &z);
  double computeCost();
  double computeRegularizedCost(double lambda);
  void gradientDescent(double alpha, int iterations);
  void gradientDescentWithRegularization(double alpha, int iterations,
                                         double lambda);

  // Prediction
  int predict(const Eigen::VectorXd &input);

  // Utility and Debugging
  void displayTheta();
  void displayX();
  void displayY();
  Eigen::VectorXd getTheta();
  void setTheta(const Eigen::VectorXd &newTheta);

private:
  Eigen::MatrixXd x;     // Matrix of input features
  Eigen::VectorXd y;     // Vector of output values
  Eigen::VectorXd theta; // Vector of model parameters
  int m;                 // # of training examples
  int n;                 // # of parameters
};

LogisticRegression::LogisticRegression() : m(0), n(0) {}
LogisticRegression::~LogisticRegression() {}

void LogisticRegression::input(const Eigen::MatrixXd &X,
                               const Eigen::VectorXd &y) {
  m = X.rows();
  n = X.cols() + 1; // add one for the bias term

  x = Eigen::MatrixXd(m, n);
  x << Eigen::MatrixXd::Ones(m, 1), X;

  this->y = y;

  theta = Eigen::VectorXd::Zero(n);
}

void LogisticRegression::displayX() {
  std::cout << "X matrix (input features): \n" << x << std::endl;
}
void LogisticRegression::displayY() {
  std::cout << "Y matrix (actual output values): \n" << y << std::endl;
}

int main() {
  // Sample dataset
  Eigen::MatrixXd X(5, 2); // 5 examples, 2 features
  X << 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0;

  Eigen::VectorXd y(5);
  y << 0, 1, 0, 1, 0;

  // Create an instance of the LogisticRegression class
  LogisticRegression lr;

  // Input the data into the model
  lr.input(X, y);

  // Display the modified input features matrix
  lr.displayX();
  lr.displayY();

  return 0;
}
