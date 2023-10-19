#include <cmath>
#include <iostream>

double gaussian_pdf(double x, double mean, double variance) {
  double pi = 3.14159265358979323846;

  return (1.0 / sqrt(2.0 * pi * variance)) *
         exp(-1.0 * pow(x - mean, 2) / (2.0 * variance));
};

int main(void) {

  double x = 0.0;
  double mean = 0.0;
  double variance = 1.0;
  double pdf_value = gaussian_pdf(x, mean, variance);

  std::cout << "The PDF value at x = " << x << " is: " << pdf_value
            << std::endl;
  return 0;
}