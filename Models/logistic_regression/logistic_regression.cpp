#include <Eigen/Dense>
#include <cmath>
#include <iostream>

// Define the LogisticRegression class
class LogisticRegression {
public:
  // Constructors and Destructors
  LogisticRegression();  // Default constructor
  ~LogisticRegression(); // Destructor

  // Data Handling
  void input(const Eigen::MatrixXd &X,
             const Eigen::VectorXd &y); // Method to input data

  // Core Computation Methods
  Eigen::VectorXd sigmoid(const Eigen::VectorXd &z); // Sigmoid function
  double computeCost(); // Compute the cost function
  void gradientDescent(double alpha,
                       int iterations); // Perform gradient descent

  // Prediction
  int predict(const Eigen::VectorXd &input); // Make a prediction

  // Utility and Debugging
  void displayTheta();        // Display current theta values
  void displayX();            // Display input features
  void displayY();            // Display output values
  Eigen::VectorXd getTheta(); // Get current theta values
  void setTheta(const Eigen::VectorXd &newTheta); // Set new theta values

private:
  Eigen::MatrixXd x;     // Matrix of input features
  Eigen::VectorXd y;     // Vector of output values
  Eigen::VectorXd theta; // Vector of model parameters
  int m;                 // Number of training examples
  int n;                 // Number of parameters
};

// Default constructor initializes m and n to 0
LogisticRegression::LogisticRegression() : m(0), n(0) {}

// Destructor (currently empty)
LogisticRegression::~LogisticRegression() {}

// Method to input data and initialize variables
void LogisticRegression::input(const Eigen::MatrixXd &X,
                               const Eigen::VectorXd &y) {
  m = X.rows();     // Number of training examples
  n = X.cols() + 1; // Number of features + 1 for the bias term

  // Initialize x matrix with a column of ones (for bias) and features
  x = Eigen::MatrixXd(m, n);
  x << Eigen::MatrixXd::Ones(m, 1), X;

  // Initialize y vector with output labels
  this->y = y;

  // Initialize theta vector with zeros
  theta = Eigen::VectorXd::Zero(n);
}

// Display the x matrix (input features)
void LogisticRegression::displayX() {
  std::cout << "X matrix (input features): \n" << x << std::endl;
}

// Display the y vector (output labels)
void LogisticRegression::displayY() {
  std::cout << "Y matrix (actual output): \n" << y << std::endl;
}

// Display the theta vector (model parameters)
void LogisticRegression::displayTheta() {
  std::cout << "Theta values (coefficents): \n" << theta << std::endl;
}

// Compute the sigmoid of the input vector
Eigen::VectorXd LogisticRegression::sigmoid(const Eigen::VectorXd &z) {
  return 1.0 / (1.0 + (-z.array()).exp());
}

// Compute the cost function
double LogisticRegression::computeCost() {
  Eigen::VectorXd h = sigmoid(x * theta); // Hypothesis function
  // Compute the cost using the logistic regression cost formula
  double cost =
      -(y.array() * h.array().log() + (1 - y.array()) * (1 - h.array()).log())
           .mean();
  return cost;
}

// Perform gradient descent to update theta values
void LogisticRegression::gradientDescent(double alpha, int iterations) {
  for (int iter = 0; iter < iterations; iter++) {
    Eigen::VectorXd h = sigmoid(x * theta); // Hypothesis function
    // Compute the gradient
    Eigen::VectorXd gradient = (x.transpose() * (h - y)) / m;
    // Update theta
    theta = theta - alpha * gradient;
  }
}

int main() {
  // Sample dataset
  Eigen::MatrixXd X(5, 2); // 5 examples, 2 features
  X << 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0;

  Eigen::VectorXd y(5);
  y << 0, 1, 0, 1, 0;

  LogisticRegression lr;

  lr.input(X, y);

  lr.displayX();
  lr.displayY();

  std::cout << "Initial cost: \n" << lr.computeCost() << std::endl;

  double alpha = 0.01;
  int iterations = 1000;
  lr.gradientDescent(alpha, iterations);

  std::cout << "Theta after gradient descent: \n";
  lr.displayTheta();

  std::cout << "Cost after gradient descent: \n"
            << lr.computeCost() << std::endl;

  return 0;
}