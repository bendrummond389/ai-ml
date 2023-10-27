cmake_minimum_required(VERSION 3.10)
project(FeedForwardNeuralNetwork)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_executable(FeedForwardNeuralNetwork ${SOURCES})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
target_link_libraries(FeedForwardNeuralNetwork Eigen3::Eigen)
