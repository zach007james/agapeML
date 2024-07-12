#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include <Eigen/Dense>

class LinearRegression
{
public:
  Eigen::VectorXd coefficients;
  double learningRate;
  int epochs;
  double tolerance;

  LinearRegression(double lr = 0.01, int ep = 1000, double tol = 1e-6);
  void train(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
  Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
  void evaluateTestSet(const Eigen::MatrixXd &X_test, const Eigen::VectorXd &y_test) const;

private:
  void reportDiagnostics(int epoch, const Eigen::VectorXd &errors, const Eigen::VectorXd &predictions, const Eigen::VectorXd &y) const;
};

#endif // LINEAR_REGRESSION_HPP