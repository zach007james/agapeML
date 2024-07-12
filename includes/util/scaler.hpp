#ifndef SCALER_HPP
#define SCALER_HPP

#include <Eigen/Dense>

enum class ScalingMethod
{
  Standard,
  MinMax,
  MaxAbs
};

class Scaler
{
public:
  Scaler(ScalingMethod method);
  void fit(const Eigen::MatrixXd &X);
  Eigen::MatrixXd transform(const Eigen::MatrixXd &X) const;
  Eigen::MatrixXd fit_transform(const Eigen::MatrixXd &X);

private:
  ScalingMethod method_;
  Eigen::VectorXd mean_;
  Eigen::VectorXd stddev_;
  Eigen::VectorXd min_;
  Eigen::VectorXd max_;
  Eigen::VectorXd max_abs_;

  void fit_standard(const Eigen::MatrixXd &X);
  Eigen::MatrixXd transform_standard(const Eigen::MatrixXd &X) const;
  void fit_minmax(const Eigen::MatrixXd &X);
  Eigen::MatrixXd transform_minmax(const Eigen::MatrixXd &X) const;
  void fit_maxabs(const Eigen::MatrixXd &X);
  Eigen::MatrixXd transform_maxabs(const Eigen::MatrixXd &X) const;
};

#endif // SCALER_HPP
