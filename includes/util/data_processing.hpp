#ifndef DATA_PROCESSING_HPP
#define DATA_PROCESSING_HPP

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <random>

struct Data
{
    std::vector<std::vector<std::string>> colnames;  // 0: Target values, 1: Input features, 2: OneHotables, 3: Auxiliary
    std::vector<std::vector<std::string>> oneHotables;
    Eigen::MatrixXd Y;  // Multiple target columns
    Eigen::MatrixXd X;  // Input features
    Eigen::VectorXd y;  // Single target column
    std::vector<std::vector<std::string>> auxiliary;  // Auxiliary data (PPT ID, Site, etc.)

    void setTarget(const std::string &targetName);
    std::pair<Data, Data> trainTestSplit(double testSize);
    std::vector<std::pair<Data, Data>> kFoldSplit(int k);

private:
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> splitMatrix(const Eigen::MatrixXd& matrix, const std::vector<int>& indices);
    std::pair<std::vector<std::vector<std::string>>, std::vector<std::vector<std::string>>> splitStringMatrix(const std::vector<std::vector<std::string>>& matrix, const std::vector<int>& indices);
};

std::vector<std::string> split(const std::string& s, char delimiter);

std::string toLower(const std::string &str);

Eigen::MatrixXd oneHotEncode(const std::vector<std::vector<std::string>>& oneHotables,
                             const std::vector<std::string>& oneHotColumns,
                             std::vector<std::string>& newColNames);

Data readCSV(const std::string& filePath, const std::vector<std::string>& targetColumns, const std::vector<std::string>& auxColumns, bool encodeOH = true);

#endif // DATA_PROCESSING_H
