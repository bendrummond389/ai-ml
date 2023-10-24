#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

class LinearRegression {
public:
  LinearRegression();
  ~LinearRegression();

  void input(const Eigen::VectorXd &x_values, const Eigen::VectorXd &y_values);
  double computeCost();
  void gradientDescent(double alpha, int iterations,
                       double convergence_threshold);
  double predict(const Eigen::VectorXd &input);
  void displayX();
  void displayY();
  void displayTheta();

private:
  Eigen::MatrixXd x;
  Eigen::VectorXd y;
  Eigen::VectorXd theta;
  double prev_cost;
  int m;
};

LinearRegression::LinearRegression() {
  m = 0;
  theta = Eigen::VectorXd::Zero(2);
  prev_cost = 0.0;
}

LinearRegression::~LinearRegression() {
  // Cleanup code
}

void LinearRegression::input(const Eigen::VectorXd &x_values,
                             const Eigen::VectorXd &y_values) {
  m = x_values.size();
  x = Eigen::MatrixXd(m, 2);
  x.col(0) = Eigen::VectorXd::Ones(m);
  x.col(1) = x_values;
  y = y_values;
}

double LinearRegression::computeCost() {
  Eigen::VectorXd h = x * theta;
  double cost = 0.0;
  for (int i = 0; i < m; i++) {
    double error = h[i] - y[i];
    cost += error * error;
  }
  return cost / (2 * m);
}

void LinearRegression::gradientDescent(double alpha, int iterations,
                                       double convergence_threshold) {
  for (int iter = 0; iter < iterations; iter++) {
    Eigen::VectorXd h = x * theta;
    Eigen::VectorXd error = h - y;

    theta(0) -= alpha * (error.sum() / m);
    theta(1) -= alpha * (x.col(1).cwiseProduct(error)).sum() / m;

    double cost = computeCost();

    if (iter > 0 && abs(prev_cost - cost) < convergence_threshold) {
      break;
    }
    prev_cost = cost;
  }
}

double LinearRegression::predict(const Eigen::VectorXd &input) {
  // Placeholder for prediction function
  return 0.0;
}

void LinearRegression::displayX() {
  std::cout << "X matrix (input features): \n";
  std::cout << x << std::endl;
}

void LinearRegression::displayY() {
  std::cout << "y vector (output values): \n";
  std::cout << y << std::endl;
}

void LinearRegression::displayTheta() {
  std::cout << "Theta vector (model parameters): \n";
  std::cout << theta << std::endl;
}

int main() {
  // Extended sample dataset with more variance
  Eigen::VectorXd x_values(10);
  x_values << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0; // Input feature values

  Eigen::VectorXd y_values(10);
  y_values << 2.3, 4.7, 6.2, 7.8, 9.5, 11.1, 12.9, 14.4, 16.6, 18.7; // Corresponding output values with variance


  // Create a LinearRegression object
  LinearRegression lr;

  // Input the sample data
  lr.input(x_values, y_values);

  // gradient descent
  lr.gradientDescent(0.001, 1000, 0.001);

  // Display the data structures
  std::cout << "Displaying theta after gradient descent" << std::endl;
  lr.displayTheta();
  std::cout << "\n------------------------\n";

  return 0;
}
