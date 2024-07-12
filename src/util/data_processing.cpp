#include "./../../includes/util/data_processing.hpp"
#include <iostream>   // Include for std::cerr and other I/O operations
#include <fstream>    // Include for std::ifstream
#include <sstream>    // This might already be included, but ensure it's there for std::istringstream
#include <algorithm>  // Include for std::transform and other algorithm functions
#include <unordered_map>
#include <random>     // Include for std::random_device, std::mt19937

std::vector<std::string> split(const std::string& s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

std::string toLower(const std::string &str)
{
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

Eigen::MatrixXd oneHotEncode(const std::vector<std::vector<std::string>>& oneHotables,
                             const std::vector<std::string>& oneHotColumns,
                             std::vector<std::string>& newColNames) {
    std::unordered_map<std::string, std::unordered_map<std::string, int>> uniqueValues;

    for (size_t i = 0; i < oneHotColumns.size(); ++i) {
        for (const auto& row : oneHotables) {
            uniqueValues[oneHotColumns[i]][row[i]]++;
        }
    }

    int totalNewCols = 0;
    for (const auto& col : uniqueValues) {
        for (const auto& val : col.second) {
            newColNames.push_back(col.first + "_" + val.first);
        }
        totalNewCols += col.second.size();
    }

    Eigen::MatrixXd encodedMatrix = Eigen::MatrixXd::Zero(oneHotables.size(), totalNewCols);

    int colIndex = 0;
    for (size_t i = 0; i < oneHotColumns.size(); ++i) {
        const auto& column = uniqueValues[oneHotColumns[i]];
        for (size_t j = 0; j < oneHotables.size(); ++j) {
            const std::string& value = oneHotables[j][i];
            int valueIndex = std::distance(column.begin(), column.find(value));
            encodedMatrix(j, colIndex + valueIndex) = 1;
        }
        colIndex += column.size();
    }

    return encodedMatrix;
}

Data readCSV(const std::string& filePath, const std::vector<std::string>& targetColumns, const std::vector<std::string>& auxColumns, bool encodeOH)
{
    Data data;
    data.colnames.resize(4);
    std::ifstream file(filePath);
    std::string line;

    if (!file.is_open())
    {
        std::cerr << "Could not open file " << filePath << std::endl;
        return data;
    }

    std::vector<std::vector<std::string>> allData;
    std::vector<std::string> colnames;

    if (std::getline(file, line))
    {
        colnames = split(line, ',');
    }

    while (std::getline(file, line))
    {
        allData.push_back(split(line, ','));
    }
    file.close();

    std::vector<std::vector<int>> indexes(4);

    for (size_t i = 0; i < colnames.size(); ++i)
    {
        std::string value = colnames[i];
        value = value.substr(value.find_first_not_of(' '), value.find_last_not_of(' ') - value.find_first_not_of(' ') + 1);

        char* checkEnd = nullptr;
        std::string actual_val = allData[0][i];
        double checkNum = std::strtod(actual_val.c_str(), &checkEnd);

        if (std::find(targetColumns.begin(), targetColumns.end(), value) != targetColumns.end())
        {
            indexes[0].push_back(i);
        }
        else if (std::find(auxColumns.begin(), auxColumns.end(), value) != auxColumns.end())
        {
            indexes[3].push_back(i);
        }
        else if (checkEnd == actual_val.c_str() + actual_val.size())
        {
            indexes[1].push_back(i);
        }
        else
        {
            indexes[2].push_back(i);
        }
    }

    size_t rowCount = allData.size();
    data.X.resize(rowCount, indexes[1].size());
    data.Y.resize(rowCount, indexes[0].size());
    data.y.resize(rowCount);
    data.auxiliary.resize(rowCount, std::vector<std::string>(indexes[3].size()));
    data.oneHotables.resize(rowCount, std::vector<std::string>(indexes[2].size()));

    for (size_t i = 0; i < rowCount; ++i)
    {
        size_t y_idx = 0, x_idx = 0, aux_idx = 0, oh_idx = 0;
        for (size_t j = 0; j < colnames.size(); ++j)
        {
            if (std::find(targetColumns.begin(), targetColumns.end(), colnames[j]) != targetColumns.end())
            {
                double value = std::stod(allData[i][j]);
                data.Y(i, y_idx) = value;
                y_idx++;
            }
            else if (std::find(auxColumns.begin(), auxColumns.end(), colnames[j]) != auxColumns.end())
            {
                data.auxiliary[i][aux_idx++] = allData[i][j];
            }
            else if (std::find(indexes[2].begin(), indexes[2].end(), j) != indexes[2].end())
            {
                data.oneHotables[i][oh_idx++] = toLower(allData[i][j]);
            }
            else
            {
                data.X(i, x_idx++) = std::stod(allData[i][j]);
            }
        }
    }

    for (int i = 0; i < indexes.size(); ++i)
    {
        for (int j : indexes[i])
        {
            data.colnames[i].push_back(colnames[j]);
        }
    }

    if (encodeOH && !data.oneHotables.empty())
    {
        std::vector<std::string> newColNames;
        Eigen::MatrixXd encodedMatrix = oneHotEncode(data.oneHotables, data.colnames[2], newColNames);

        Eigen::MatrixXd newX(data.X.rows(), data.X.cols() + encodedMatrix.cols());
        newX << data.X, encodedMatrix;

        data.X = newX;
        data.colnames[1].insert(data.colnames[1].end(), newColNames.begin(), newColNames.end());
    }

    data.setTarget(targetColumns[0]);

    return data;
}

void Data::setTarget(const std::string &targetName)
{
    auto it = std::find(colnames[0].begin(), colnames[0].end(), targetName);
    if (it != colnames[0].end())
    {
        int index = std::distance(colnames[0].begin(), it);
        y = Y.col(index);
    }
    else
    {
        std::cerr << "Target column " << targetName << " not found in Y matrix." << std::endl;
    }
}

std::pair<Data, Data> Data::trainTestSplit(double testSize)
{
    Data trainData, testData;
    trainData.colnames = colnames;
    testData.colnames = colnames;

    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<int> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), g);

    int testCount = static_cast<int>(testSize * X.rows());
    std::vector<int> testIndices(indices.begin(), indices.begin() + testCount);
    std::vector<int> trainIndices(indices.begin() + testCount, indices.end());

    std::tie(trainData.X, testData.X) = splitMatrix(X, trainIndices);
    std::tie(trainData.Y, testData.Y) = splitMatrix(Y, trainIndices);
    std::tie(trainData.auxiliary, testData.auxiliary) = splitStringMatrix(auxiliary, trainIndices);
    std::tie(trainData.oneHotables, testData.oneHotables) = splitStringMatrix(oneHotables, trainIndices);

    trainData.y = y(trainIndices);
    testData.y = y(testIndices);

    return {trainData, testData};
}

std::vector<std::pair<Data, Data>> Data::kFoldSplit(int k)
{
    std::vector<std::pair<Data, Data>> folds;
    int foldSize = X.rows() / k;

    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<int> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), g);

    for (int i = 0; i < k; ++i)
    {
        Data trainData, testData;
        trainData.colnames = colnames;
        testData.colnames = colnames;

        std::vector<int> testIndices(indices.begin() + i * foldSize, indices.begin() + (i + 1) * foldSize);
        std::vector<int> trainIndices;
        trainIndices.reserve(X.rows() - foldSize);

        for (int j = 0; j < indices.size(); ++j)
        {
            if (std::find(testIndices.begin(), testIndices.end(), indices[j]) == testIndices.end())
            {
                trainIndices.push_back(indices[j]);
            }
        }

        std::tie(trainData.X, testData.X) = splitMatrix(X, trainIndices);
        std::tie(trainData.Y, testData.Y) = splitMatrix(Y, trainIndices);
        std::tie(trainData.auxiliary, testData.auxiliary) = splitStringMatrix(auxiliary, trainIndices);
        std::tie(trainData.oneHotables, testData.oneHotables) = splitStringMatrix(oneHotables, trainIndices);

        trainData.y = y(trainIndices);
        testData.y = y(testIndices);

        folds.push_back({trainData, testData});
    }

    return folds;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> Data::splitMatrix(const Eigen::MatrixXd& matrix, const std::vector<int>& indices)
{
    Eigen::MatrixXd trainMatrix(indices.size(), matrix.cols());
    Eigen::MatrixXd testMatrix(matrix.rows() - indices.size(), matrix.cols());

    for (size_t i = 0; i < indices.size(); ++i)
    {
        trainMatrix.row(i) = matrix.row(indices[i]);
    }

    int testIdx = 0;
    for (int i = 0; i < matrix.rows(); ++i)
    {
        if (std::find(indices.begin(), indices.end(), i) == indices.end())
        {
            testMatrix.row(testIdx++) = matrix.row(i);
        }
    }

    return {trainMatrix, testMatrix};
}

std::pair<std::vector<std::vector<std::string>>, std::vector<std::vector<std::string>>> Data::splitStringMatrix(const std::vector<std::vector<std::string>>& matrix, const std::vector<int>& indices)
{
    std::vector<std::vector<std::string>> trainMatrix(indices.size());
    std::vector<std::vector<std::string>> testMatrix(matrix.size() - indices.size());

    for (size_t i = 0; i < indices.size(); ++i)
    {
        trainMatrix[i] = matrix[indices[i]];
    }

    int testIdx = 0;
    for (size_t i = 0; i < matrix.size(); ++i)
    {
        if (std::find(indices.begin(), indices.end(), i) == indices.end())
        {
            testMatrix[testIdx++] = matrix[i];
        }
    }

    return {trainMatrix, testMatrix};
}
