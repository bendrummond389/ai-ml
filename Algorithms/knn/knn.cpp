#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>


struct Point {
    double x, y;
    int label; 
};

double distance(Point a, Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

int knn(std::vector<Point> data, Point p, int k) {
    std::vector<std::pair<double, int> > distances;

    for (const auto& point : data) {
        distances.push_back({distance(p, point), point.label});
    }

    std::sort(distances.begin(), distances.end());

    int count_0 = 0, count_1 = 0;
    for (int i = 0; i < k; ++i) {
        if (distances[i].second == 0) {
            count_0++;
        } else {
            count_1++;
        }
    }

    return (count_0 > count_1) ? 0 : 1;
}

int main() {
    std::vector<Point> data = {
        {1, 1, 0}, {2, 2, 0}, {3, 3, 0},
        {6, 6, 1}, {7, 7, 1}, {8, 8, 1}
    };

    Point test_point = {4, 4, -1};  // -1 indicates the label is unknown

    int k = 3;
    int label = knn(data, test_point, k);

    std::cout << "The predicted label for the test_point is: " << label << std::endl;

    return 0;
}
