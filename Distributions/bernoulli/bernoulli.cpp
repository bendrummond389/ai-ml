#include <iostream>

class BernoulliDistribution {
public:
  BernoulliDistribution(double probability);
  double pmf(int k);

  double mean();
  double variance();

private:
  double p;
};

BernoulliDistribution::BernoulliDistribution(double probability)
    : p(probability) {}

double BernoulliDistribution::pmf(int k) {
  if (k == 1) {
    return p;
  } else if (k == 0) {
    return 1 - p;
  } else {
    return 0;
  }
}

double BernoulliDistribution::mean() { return p; }

double BernoulliDistribution::variance() { return p * (1 - p); }

int main() {
  BernoulliDistribution bernoulli(0.6);

  std::cout << "PMF(1) = " << bernoulli.pmf(1) << std::endl;
  std::cout << "PMF(0) = " << bernoulli.pmf(0) << std::endl;
  std::cout << "Mean = " << bernoulli.mean() << std::endl;
  std::cout << "Variance = " << bernoulli.variance() << std::endl;

  return 0;
}