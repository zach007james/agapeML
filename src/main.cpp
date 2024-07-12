#include "./../../includes/util/data_processing.hpp"
#include "./../../includes/models/linear_regression.hpp"
#include "./../../includes/util/scaler.hpp"
#include "gnuplot-iostream.h"
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>

int main()
{
  std::cout << "Starting the Linear Regression model" << std::endl;

  std::string dataPath = "../data/pennington_summer_24/complete_data.csv";
  std::vector<std::string> targetColumns = {"ALM"};
  std::vector<std::string> auxColumns = {"PPT ID", "Site", "0"};

  Data data = readCSV(dataPath, targetColumns, auxColumns, true);

  double testSize = 0.2;
  auto splitSets = data.trainTestSplit(testSize);

  Scaler scaler(ScalingMethod::Standard);

  Eigen::MatrixXd X_train_scaled = scaler.fit_transform(splitSets.first.X);
  Eigen::MatrixXd X_test_scaled = scaler.transform(splitSets.second.X);

  std::cout << "First 5 rows of feature matrix X (scaled):\n"
            << X_train_scaled.topRows(5) << std::endl;
  std::cout << "First 5 elements of target vector y:\n"
            << splitSets.first.y.head(5) << std::endl;

  std::cout << "Feature column names left in X:" << std::endl;
  for (const auto &name : data.colnames[1])
  {
    std::cout << name << std::endl;
  }

  LinearRegression model(0.001, 1000);
  model.train(X_train_scaled, splitSets.first.y);

  // Evaluate on the test set
  model.evaluateTestSet(X_test_scaled, splitSets.second.y);

  // Print coefficients
  std::cout << "\nModel Coefficients:\n";
  for (int i = 0; i < model.coefficients.size(); ++i)
  {
    std::cout << "Feature: " << (i == 0 ? "Intercept" : data.colnames[1][i - 1])
              << ": " << std::fixed << std::setprecision(4) << model.coefficients(i) << std::endl;
  }

  // testing gp
  Gnuplot gp;

  // Create a sine wave data
  std::vector<std::pair<double, double>> sine_wave;
  for (double x = 0; x < 10; x += 0.1)
  {
    sine_wave.push_back(std::make_pair(x, std::sin(x)));
  }

  // Send the sine wave data to gnuplot
  gp << "set terminal pngcairo enhanced font 'Verdana,10'\n";
  gp << "set output 'sine_wave.png'\n";
  gp << "set title 'Actual vs Predicted'\n";
  gp << "set xlabel 'ALM'\n";
  gp << "set ylabel 'Sin(X)'\n";
  gp << "plot '-' with lines title 'sin(x)'\n";
  gp.send1d(sine_wave);

  // Optional: Output to screen
  gp << "set output\n";
  gp << "set terminal wxt\n";
  gp << "replot\n";

  return 0;
}