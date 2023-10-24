#include <Eigen/Dense>
#include <cmath>
#include <iostream>

// Linear Regression class for single variable regression
class LinearRegression {
public:
  LinearRegression();  // Constructor
  ~LinearRegression(); // Destructor

  // Method to input training data
  void input(const Eigen::VectorXd &x_values, const Eigen::VectorXd &y_values);

  // Method to compute the cost
  double computeCost();

  // Method to perform gradient descent
  void gradientDescent(double alpha, int iterations,
                       double convergence_threshold);

  // Method to predict a new input value
  double predict(const Eigen::VectorXd &input);

  // Methods to display internal data structures
  void displayX();
  void displayY();
  void displayTheta();

private:
  Eigen::MatrixXd x;     // Matrix of input features
  Eigen::VectorXd y;     // Vector of output values
  Eigen::VectorXd theta; // Vector of model parameters
  double prev_cost;      // Store the previous cost for convergence check
  int m;                 // Number of training examples
};

LinearRegression::LinearRegression() : m(0), prev_cost(0.0) {
  theta = Eigen::VectorXd::Zero(2); // Initialize theta with zeros
}

LinearRegression::~LinearRegression() {}

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
  Eigen::VectorXd errors = h - y;
  return errors.squaredNorm() / (2 * m);
}

void LinearRegression::gradientDescent(double alpha, int iterations,
                                       double convergence_threshold) {
  for (int iter = 0; iter < iterations; iter++) {
    Eigen::VectorXd h = x * theta;
    Eigen::VectorXd error = h - y;

    theta -= alpha * (x.transpose() * error) / m;

    double cost = computeCost();

    if (iter > 0 && std::abs(prev_cost - cost) < convergence_threshold) {
      break;
    }
    prev_cost = cost;
  }
}

double LinearRegression::predict(const Eigen::VectorXd &input) {
  return 0.0; // Placeholder for prediction function
}

void LinearRegression::displayX() {
  std::cout << "X matrix (input features): \n" << x << std::endl;
}

void LinearRegression::displayY() {
  std::cout << "y vector (output values): \n" << y << std::endl;
}

void LinearRegression::displayTheta() {
  std::cout << "Theta vector (model parameters): \n" << theta << std::endl;
}

int main() {
  // Sample dataset
  Eigen::VectorXd x_values(10);
  x_values << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0;

  Eigen::VectorXd y_values(10);
  y_values << 2.3, 4.7, 6.2, 7.8, 9.5, 11.1, 12.9, 14.4, 16.6, 18.7;

  LinearRegression lr;
  lr.input(x_values, y_values);
  lr.gradientDescent(0.001, 1000, 0.001);

  std::cout << "Displaying theta after gradient descent:" << std::endl;
  lr.displayTheta();
  std::cout << "\n------------------------\n";

  lr.displayX();

  return 0;
}
