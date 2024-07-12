#include "./../../includes/models/linear_regression.hpp"
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <chrono>
#include <thread>

LinearRegression::LinearRegression(double lr, int ep, double tol)
    : learningRate(lr), epochs(ep), tolerance(tol)
{
  // Coefficients will be initialized in the train method
}

void LinearRegression::train(const Eigen::MatrixXd &X, const Eigen::VectorXd &y)
{
  int m = X.rows();
  int n = X.cols();

  coefficients = Eigen::VectorXd::Zero(n + 1);

  Eigen::MatrixXd X_aug = Eigen::MatrixXd::Ones(m, n + 1);
  X_aug.rightCols(n) = X;

  const int barWidth = 50;

  for (int epoch = 0; epoch < epochs; ++epoch)
  {
    Eigen::VectorXd predictions = X_aug * coefficients;
    Eigen::VectorXd errors = predictions - y;
    Eigen::VectorXd gradient = (2.0 / m) * (X_aug.transpose() * errors);

    coefficients -= learningRate * gradient;

    // Update loading bar
    float progress = static_cast<float>(epoch) / epochs;
    int pos = barWidth * progress;

    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i)
    {
      if (i < pos)
        std::cout << "#";
      else if (i == pos)
        std::cout << ">";
      else
        std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% ";
    std::cout.flush();

    if (gradient.norm() < tolerance)
    {
      std::cout << "\nConvergence reached at epoch " << epoch << std::endl;
      break;
    }

    if (epoch % 50 == 0)
    {
      reportDiagnostics(epoch, errors, predictions, y);
    }

    // Add a small delay to make the loading bar visible
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  std::cout << std::endl; // Move to the next line after the loading bar
}

void LinearRegression::reportDiagnostics(int epoch, const Eigen::VectorXd &errors, const Eigen::VectorXd &predictions, const Eigen::VectorXd &y) const
{
  double mse = errors.squaredNorm() / errors.size();
  double rmse = std::sqrt(mse);
  double mae = errors.cwiseAbs().sum() / errors.size();
  double maxe = errors.cwiseAbs().maxCoeff();
  double total_variance = (y.array() - y.mean()).square().sum();
  double r2 = 1 - mse * errors.size() / total_variance;

  std::cout << "\nEpoch " << epoch << std::endl;

  for (int i = 0; i < std::min(int(predictions.size()), 5); ++i)
  {
    std::cout << "Actual: " << y(i) << ", Predicted: " << predictions(i) << ";\n";
  }
  std::cout << "| RMSE: " << rmse << " | R²: " << r2
            << " | MAE: " << mae << " | Max Error: " << maxe << std::endl;
}

Eigen::VectorXd LinearRegression::predict(const Eigen::MatrixXd &X) const
{
  Eigen::MatrixXd X_aug = Eigen::MatrixXd::Ones(X.rows(), X.cols() + 1);
  X_aug.rightCols(X.cols()) = X;
  return X_aug * coefficients;
}

void LinearRegression::evaluateTestSet(const Eigen::MatrixXd &X_test, const Eigen::VectorXd &y_test) const
{
  Eigen::VectorXd predictions = predict(X_test);
  Eigen::VectorXd errors = predictions - y_test;

  std::cout << "\nTest Set Evaluation:\n";
  std::cout << "Actual - Predicted:\n";
  for (int i = 0; i < predictions.size(); ++i)
  {
    std::cout << "(" << std::fixed << std::setprecision(2)
              << y_test(i) << " | " << predictions(i) << ")";
    if ((i + 1) % 5 == 0 || i == predictions.size() - 1)
      std::cout << "\n";
    else
      std::cout << ", ";
  }

  double mse = errors.squaredNorm() / errors.size();
  double rmse = std::sqrt(mse);
  double mae = errors.cwiseAbs().sum() / errors.size();
  double maxe = errors.cwiseAbs().maxCoeff();
  double total_variance = (y_test.array() - y_test.mean()).square().sum();
  double r2 = 1 - mse * errors.size() / total_variance;

  std::cout << "\nFinal Error Stats:\n";
  std::cout << "RMSE: " << rmse << " | R²: " << r2
            << " | MAE: " << mae << " | Max Error: " << maxe << std::endl;
}