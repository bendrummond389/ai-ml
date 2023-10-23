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
  void gradientDescent(double alpha, int iterations);
  double predict(const Eigen::VectorXd &input);

private:
  Eigen::VectorXd x;
  Eigen::VectorXd y;
  Eigen::VectorXd theta;
  int m;
};

LinearRegression::LinearRegression() { m = 0; }

LinearRegression::~LinearRegression() {
  // Cleanup code
}

void LinearRegression::input(const Eigen::VectorXd &x_values,
                             const Eigen::VectorXd &y_values) {
  x = x_values;
  y = y_values;
}

double LinearRegression::computeCost() {
  // Placeholder for cost computation
  return 0.0;
}

void LinearRegression::gradientDescent(double alpha, int iterations) {
  // Placeholder for gradient descent algorithm
}

double LinearRegression::predict(const Eigen::VectorXd &input) {
  // Placeholder for prediction function
  return 0.0;
}

int main(void) { std::cout << "hello" << std::endl; }