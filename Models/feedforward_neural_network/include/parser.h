#pragma once

#include <Eigen/Dense>
#include <string>

Eigen::VectorXd parseLabels(const std::string &filename);
Eigen::MatrixXd parseImages(const std::string &filename);
