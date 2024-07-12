#include "./../../includes/util/scaler.hpp"
#include <iostream>
#include <cmath>

Scaler::Scaler(ScalingMethod method) : method_(method) {}

void Scaler::fit(const Eigen::MatrixXd &X)
{
  switch (method_)
  {
  case ScalingMethod::Standard:
    fit_standard(X);
    break;
  case ScalingMethod::MinMax:
    fit_minmax(X);
    break;
  case ScalingMethod::MaxAbs:
    fit_maxabs(X);
    break;
  }
}

Eigen::MatrixXd Scaler::transform(const Eigen::MatrixXd &X) const
{
  switch (method_)
  {
  case ScalingMethod::Standard:
    return transform_standard(X);
  case ScalingMethod::MinMax:
    return transform_minmax(X);
  case ScalingMethod::MaxAbs:
    return transform_maxabs(X);
  default:
    throw std::invalid_argument("Unknown scaling method");
  }
}

Eigen::MatrixXd Scaler::fit_transform(const Eigen::MatrixXd &X)
{
  fit(X);
  return transform(X);
}

void Scaler::fit_standard(const Eigen::MatrixXd &X)
{
  mean_ = X.colwise().mean();
  stddev_ = ((X.rowwise() - mean_.transpose()).array().square().colwise().sum() / (X.rows() - 1)).sqrt();
}

Eigen::MatrixXd Scaler::transform_standard(const Eigen::MatrixXd &X) const
{
  Eigen::ArrayXd safe_stddev = stddev_.array().abs().max(1e-10); // Ensures no value less than 1e-10
  return (X.rowwise() - mean_.transpose()).array().rowwise() / safe_stddev.transpose();
}

void Scaler::fit_minmax(const Eigen::MatrixXd &X)
{
  min_ = X.colwise().minCoeff();
  max_ = X.colwise().maxCoeff();
}

Eigen::MatrixXd Scaler::transform_minmax(const Eigen::MatrixXd &X) const
{
  Eigen::VectorXd range = max_ - min_;
  Eigen::VectorXd safe_range = range.array().abs().max(1e-10); // Ensures no range value less than 1e-10
  return (X.rowwise() - min_.transpose()).array().rowwise() / safe_range.transpose().array();
}

void Scaler::fit_maxabs(const Eigen::MatrixXd &X)
{
  max_abs_ = X.cwiseAbs().colwise().maxCoeff();
}

Eigen::MatrixXd Scaler::transform_maxabs(const Eigen::MatrixXd &X) const
{
  return X.array().rowwise() / max_abs_.transpose().array();
}
