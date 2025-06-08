#pragma once

#include <vector>
#include <cstddef>
#include <algorithm>

namespace tensor {
  class Matrix {
    private:
      size_t _rows;
      size_t _cols;
      std::vector<float> _data;
    public:

      explicit Matrix(size_t rows, size_t cols) : _rows(rows), _cols(cols), _data(_rows * _cols, 0.0f) {}

      // Rule of Five
      // 1. Copy Constructor
      Matrix(const Matrix& other) : _rows(other._rows), _cols(other._cols), _data(other._data) {}

      // 2. Copy assignment operator
      Matrix& operator=(const Matrix& other) {
        if (this == &other) {
          return *this;
        }
        _rows = other._rows;
        _cols = other._cols;
        _data = other._data;

        return *this;
      }

      // 3. Move constructor
      Matrix(Matrix&& other) noexcept : _rows(other._rows), _cols(other._cols), _data(std::move(other._data)) {
        other._rows = 0;
        other._cols = 0;
      }

      // 4. Move assignment operator
      Matrix& operator=(Matrix&& other) noexcept {
        if (this == &other) {
          return *this;
        }

        _rows = other._rows;
        _cols = other._cols;
        _data = std::move(other._data);

        other._rows = 0;
        other._cols = 0;

        return *this;
      }

      // 5. Destructor
      ~Matrix() = default;

      // element access
      float& operator()(size_t i, size_t j);
      const float& operator()(size_t i, size_t j) const;
      
      size_t rows() const;
      size_t cols() const;

      // operations
      
      // element-wise addition
      Matrix add(const Matrix& other) const;
      Matrix operator+(const Matrix& other) const;
      // element-wise subtraction
      Matrix sub(const Matrix& other) const;
      Matrix operator-(const Matrix& other) const;
      // scalar multiplication
      Matrix scalMul(const float& x) const;
      Matrix operator*(const float &x) const;
      // scalar division
      Matrix scalDiv(const float& x) const;
      Matrix operator/(const float &x) const;
      // element-wise multiplication
      Matrix hadamard(const Matrix& other) const;
      // matrix multiplication
      Matrix matMul(const Matrix& other) const;
      Matrix operator*(const Matrix& other) const;
  };
};
