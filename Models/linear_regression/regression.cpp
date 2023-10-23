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
  void displayX();
  void displayY();
  void displayTheta();

private:
  Eigen::MatrixXd x;
  Eigen::VectorXd y;
  Eigen::VectorXd theta;
  int m;
};

LinearRegression::LinearRegression() {
  m = 0;
  theta = Eigen::VectorXd::Zero(2);
}

LinearRegression::~LinearRegression() {
  // Cleanup code
}

void LinearRegression::input(const Eigen::VectorXd &x_values,
                             const Eigen::VectorXd &y_values) {
  m = x_values.size();
  x = Eigen::MatrixXd(m, 2);           // Matrix with m rows and 2 columns
  x.col(0) = Eigen::VectorXd::Ones(m); // First column as ones
  x.col(1) = x_values;
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
  // Sample dataset
  Eigen::VectorXd x_values(5);
  x_values << 1.0, 2.0, 3.0, 4.0, 5.0; // Input feature values

  Eigen::VectorXd y_values(5);
  y_values << 2.0, 4.0, 5.5, 8.0, 10.0; // Corresponding output values

  // Create a LinearRegression object
  LinearRegression lr;

  // Input the sample data
  lr.input(x_values, y_values);

  // Display the data structures
  std::cout << "Displaying X matrix:" << std::endl;
  lr.displayX();
  std::cout << "\n------------------------\n";

  std::cout << "Displaying y vector:" << std::endl;
  lr.displayY();
  std::cout << "\n------------------------\n";

  std::cout << "Displaying theta vector:" << std::endl;
  lr.displayTheta();
  std::cout << "\n------------------------\n";

  return 0;
}
