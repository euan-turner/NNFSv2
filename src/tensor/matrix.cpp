#include <cassert>
#include "tensor/matrix.hpp"
#include <stdexcept>

namespace tensor {
  float& Matrix::operator()(size_t i, size_t j) {
    if (i >= _rows || j >= _cols)
      throw std::runtime_error("Matrix indices out of bounds");

    return _data[i * _cols + j];
  }
  
  const float& Matrix::operator()(size_t i, size_t j) const {
    if (i >= _rows || j >= _cols)
      throw std::runtime_error("Matrix indices out of bounds");

    return _data[i * _cols + j];
  }

  size_t Matrix::rows() const {
    return _rows;
  }

  size_t Matrix::cols() const {
    return _cols;
  }

  // all element-wise operations should be abstracted
  Matrix Matrix::add(const Matrix& other) const {
    if (_rows != other._rows || _cols != other._cols)
      throw std::runtime_error("Matrix dimensions must match for add");

    Matrix result(_rows, _cols);
    for (size_t i = 0; i < _rows * _cols; ++i)
      result._data[i] = _data[i] + other._data[i];
    return result;
  }

  Matrix Matrix::operator+(const Matrix& other) const {
    return add(other);
  }

  Matrix Matrix::sub(const Matrix& other) const {
    if (_rows != other._rows || _cols != other._cols)
      throw std::runtime_error("Matrix dimensions must match for sub");

    Matrix result(_rows, _cols);
    for (size_t i = 0; i < _rows * _cols; ++i)
      result._data[i] = _data[i] - other._data[i];
    return result;
  }

  Matrix Matrix::operator-(const Matrix& other) const {
    return sub(other);
  }

  Matrix Matrix::hadamard(const Matrix& other) const {
    if (_rows != other._rows || _cols != other._cols)
      throw std::runtime_error("Matrix dimensions must match for hadamard product");

    Matrix result(_rows, _cols);
    for (size_t i = 0; i < _rows * _cols; ++i)
      result._data[i] = _data[i] * other._data[i];
    return result;
  }

  // all scalar operations should be abstracted
  Matrix Matrix::scalMul(const float& x) const {
    Matrix result(_rows, _cols);
    for (size_t i = 0; i < _rows * _cols; ++i)
      result._data[i] = _data[i] * x;
    return result;
  }

  Matrix Matrix::operator*(const float& x) const {
    return scalMul(x);
  }

  Matrix Matrix::scalDiv(const float& x) const {
    Matrix result(_rows, _cols);
    for (size_t i = 0; i < _rows * _cols; ++i)
      result._data[i] = _data[i] / x;
    return result;
  }

  Matrix Matrix::operator/(const float& x) const {
    return scalDiv(x);
  }

  Matrix Matrix::matMul(const Matrix& other) const {
    if (_cols != other._rows)
      throw std::runtime_error("First matrix columns must equal second matrix rows for multiplication");

    Matrix result(_rows, other._cols);
    for (int i = 0; i < _rows; ++i) {
      for (int k = 0; k < _cols; ++k) {
        for (int j = 0; j < other._cols; ++j) {
          result(i, j) += (*this)(i, k) * other(k, j);
        }
      }
    }
    return result;
  }

  Matrix Matrix::operator*(const Matrix& other) const {
    return matMul(other);
  }

  // all reductions should be abstracted as folds
  float Matrix::sum() const {
    float result = 0.0f;
    for (int i = 0; i < _rows * _cols; ++i)
      result += _data[i];
    return result;
  }

  float Matrix::mean() const {
    if (_rows * _cols == 0)
      throw std::runtime_error("Cannot compute mean of empty matrix");

    return sum() / (_rows * _cols);
  }

  float Matrix::max() const {
    if (_rows * _cols == 0)
      throw std::runtime_error("Cannot compute max of empty matrix");

    float result = _data[0];
    for (int i = 1; i < _rows * _cols; ++i) {
      float elem = _data[i];
      if (elem > result)
        result = elem;
    }
    return result;
  }

  std::vector<float> Matrix::sum(int axis) const {
    if (axis != 0 && axis != 1)
      throw std::runtime_error("Axis must be 0 (rows) or 1 (cols) for reduction on matrix");
    
    std::vector<float> res;
    if (axis == 0) {
      res = std::vector<float>(_cols, 0.0f);
      for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
          res[j] += _data[i * _cols + j];
        }
      }
    } else {
      res = std::vector<float>(_cols, 0.0f);
      for (int i = 0; i < _cols; ++i) {
        for (int j = 0; j < _rows; ++j) {
          res[j] += _data[j * _cols + i];
        }
      }
    }
    return res;
  }

  std::vector<float> Matrix::mean(int axis) const {
    if (axis != 0 && axis != 1)
      throw std::runtime_error("Axis must be 0 (rows) or 1 (cols) for reduction on matrix");

    std::vector<float> res = sum(axis);
    if (axis == 0) {
      for (int i = 0; i < _cols; ++i)
        res[i] /= _rows;
    } else{
      for (int i = 0; i < _rows; ++i)
        res[i] /= _cols;
    }
    return res;

  }

  std::vector<float> Matrix::max(int axis) const {
    if (axis != 0 && axis != 1)
      throw std::runtime_error("Axis must be 0 (rows) or 1 (cols) for reduction on matrix");
  
      std::vector<float> res;
      if (axis == 0) {
        res = std::vector<float>(_cols, 0.0f);
        for (int i = 0; i < _rows; ++i) {
          for (int j = 0; j < _cols; ++j) {
            float elem = _data[i * _cols + j];
            if (elem > res[j])
              res[j] = elem;
          }
        }
      } else {
        res = std::vector<float>(_cols, 0.0f);
        for (int i = 0; i < _cols; ++i) {
          for (int j = 0; j < _rows; ++j) {
            float elem = _data[j * _cols + i];
            if (elem > res[j])
              res[j] = elem;
          }
        }
      }
      return res;
  }
}