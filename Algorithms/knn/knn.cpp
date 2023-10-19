#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

// Define a Point struct that contains coordinates (x, y) and a label
struct Point {
  double x, y;
  int label;
};

// Function to calculate the Euclidean distance between two points
double distance(Point a, Point b) {
  return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

// k-NN function to classify a given point based on its k nearest neighbors
int knn(const std::vector<Point> &data, Point p, int k) {
  // Vector to hold distances and labels of data points
  std::vector<std::pair<double, int>> distances;

  // Calculate distances from point p to all points in data
  for (const auto &point : data) {
    distances.push_back({distance(p, point), point.label});
  }

  // Sort the distances in ascending order
  std::sort(distances.begin(), distances.end());

  // Count labels of k nearest neighbors
  int count_0 = 0, count_1 = 0;
  for (int i = 0; i < k; ++i) {
    if (distances[i].second == 0) {
      ++count_0;
    } else {
      ++count_1;
    }
  }

  // Return the majority label among k nearest neighbors
  return (count_0 > count_1) ? 0 : 1;
}

int main() {
  // Initialize dataset
  std::vector<Point> data = {{1, 1, 0}, {2, 2, 0}, {3, 3, 0},
                             {6, 6, 1}, {7, 7, 1}, {8, 8, 1}};

  // Initialize test point with unknown label (-1)
  Point test_point = {7, 8, -1};

  // Number of nearest neighbors to consider
  int k = 3;

  // Predict the label for the test_point
  int label = knn(data, test_point, k);

  // Output the result
  std::cout << "The predicted label for the test_point is: " << label
            << std::endl;

  return 0;
}
