#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

// Function to calculate the Gaussian PDF
double gaussian_pdf(double x, double mean, double variance) {
  const double pi = 3.14159265358979323846;

  return (1.0 / sqrt(2.0 * pi * variance)) *
         exp(-1.0 * pow(x - mean, 2) / (2.0 * variance));
}

// Function to calculate the mean of a vector
double calculate_mean(const std::vector<double> &vec) {
  double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
  return sum / vec.size();
}

// Function to calculate the variance of a vector
double calculate_variance(const std::vector<double> &vec, double mean) {
  double sum = 0.0;
  for (double x : vec) {
    sum += pow(x - mean, 2);
  }
  return sum / vec.size();
}

int main() {
  // Initialize test point and data vector
  double test_point = 0.0;
  std::vector<double> myVector = {1.0, 2.0};

  // Calculate mean and variance
  double mean = calculate_mean(myVector);
  double variance = calculate_variance(myVector, mean);

  // Calculate Gaussian PDF value
  double pdf_value = gaussian_pdf(test_point, mean, variance);

  // Output results
  std::cout << "The mean is: " << mean << " and the variance is " << variance
            << std::endl;

  std::cout << "The PDF value at x = " << test_point << " is: " << pdf_value
            << std::endl;

  return 0;
}
