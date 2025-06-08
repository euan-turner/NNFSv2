#include <cassert>
#include "tensor/matrix.hpp"
#include <stdexcept>

namespace tensor {
  float& Matrix::operator()(size_t i, size_t j) {
    assert(i < _rows && j < _cols && "Matrix indices out of bounds");
    return _data[i * _cols + j];
  }
  
  const float& Matrix::operator()(size_t i, size_t j) const {
    assert(i < _rows && j < _cols && "Matrix indices out of bounds");
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
    assert(_rows == other._rows && _cols == other._cols && "Matrix dimensions must match for add");

    Matrix result(_rows, _cols);
    for (size_t i = 0; i < _rows * _cols; ++i)
      result._data[i] = _data[i] + other._data[i];
    return result;
  }

  Matrix Matrix::operator+(const Matrix& other) const {
    return add(other);
  }

  Matrix Matrix::sub(const Matrix& other) const {
    assert(_rows == other._rows && _cols == other._cols && "Matrix dimensions must match for sub");

    Matrix result(_rows, _cols);
    for (size_t i = 0; i < _rows * _cols; ++i)
      result._data[i] = _data[i] - other._data[i];
    return result;
  }

  Matrix Matrix::operator-(const Matrix& other) const {
    return sub(other);
  }

  Matrix Matrix::hadamard(const Matrix& other) const {
    assert(_rows == other._rows && _cols == other._cols && "Matrix dimensions must match for hadamard product");

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
    assert(_cols == other._rows && "First matrix columns must equal second matrix rows for multiplication");
    Matrix result(_rows, other._cols);
    for (int i = 0; i < _rows; ++i) {
      for (int j = 0; j < other._cols; ++j) {
        for (int k = 0; k < _cols; ++k) {
          result(i, j) += (*this)(i, k) * other(k, j);
        }
      }
    }
    return result;
  }

  Matrix Matrix::operator*(const Matrix& other) const {
    return matMul(other);
  }
}