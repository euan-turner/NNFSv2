#include "tensor/tensor.hpp"
#include <stdexcept>
#include <numeric>

namespace tensor {

  Tensor::Tensor(std::vector<size_t> dims) : _dims(std::move(dims)) {
    size_t n = _dims.size();
    if (n == 0)
      throw std::runtime_error("dims cannot be empty for tensor");
    size_t size = 1;
    for (const size_t& d : _dims)
      size *= d;
    if (size == 0)
      throw std::runtime_error("size of tensor cannot be zero");
    _data = std::vector<float>(size, 0.0f);

    _strides = std::vector<size_t>(n, 1);
    size_t prod = 1;
    for (size_t i = n; i-- > 0;) {
      _strides[i] = prod;
      prod *= _dims[i];
    }
  }

  Tensor::Tensor(const Tensor& other)
    : _dims(other._dims), _strides(other._strides), _data(other._data) {}

  Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
      return *this;
    }
    _dims = other._dims;
    _strides = other._strides;
    _data = other._data;
    return *this;
  }

  Tensor::Tensor(Tensor&& other) noexcept
    : _dims(std::move(other._dims)), _strides(std::move(other._strides)), _data(std::move(other._data)) {}

  Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    _dims = std::move(other._dims);
    _strides = std::move(other._strides);
    _data = std::move(other._data);
    return *this;
  }

  Tensor::~Tensor() = default;

  float& Tensor::operator()(std::vector<size_t> indices) {
    if (indices.size() != _dims.size())
      throw std::runtime_error("Incorrect number of indices");
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] >= _dims[i])
        throw std::runtime_error("Index out of bounds");
      offset += indices[i] * _strides[i];
    }
    return _data[offset];
  }

  const float& Tensor::operator()(std::vector<size_t> indices) const {
    if (indices.size() != _dims.size())
      throw std::runtime_error("Incorrect number of indices");
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] >= _dims[i])
        throw std::runtime_error("Index out of bounds");
      offset += indices[i] * _strides[i];
    }
    return _data[offset];
  }

  size_t Tensor::dim(uint axis) const {
    if (axis >= _dims.size())
      throw std::runtime_error("axis out of bounds");
    return _dims[axis];
  }

  size_t Tensor::stride(uint axis) const {
    if (axis >= _dims.size())
      throw std::runtime_error("axis out of bounds");
    return _strides[axis];
  }

  size_t Tensor::size() const {
    return _data.size();
  }

  // TODO: Abstract element-wise operations as higher order
  Tensor Tensor::add(const Tensor& other) const {
    if (!elementWiseCompatible(other))
      throw std::runtime_error("tensor shapes must match for element-wise add");

    Tensor res(_dims);
    for (size_t i = 0; i < _data.size(); ++i)
      res._data[i] = _data[i] + other._data[i];
    return res;
  }

  Tensor Tensor::operator+(const Tensor& other) const {
    return add(other);
  }

  Tensor Tensor::sub(const Tensor& other) const {
    if (!elementWiseCompatible(other))
      throw std::runtime_error("tensor shapes must match for element-wise sub");

    Tensor res(_dims);
    for (size_t i = 0; i < _data.size(); ++i)
      res._data[i] = _data[i] - other._data[i];
    return res;
  }

  Tensor Tensor::operator-(const Tensor& other) const {
    return sub(other);
  }

  Tensor Tensor::hadamard(const Tensor& other) const {
    if (!elementWiseCompatible(other))
      throw std::runtime_error("tensor shapes must match for element-wise product");

    Tensor res(_dims);
    for (size_t i = 0; i < _data.size(); ++i)
      res._data[i] = _data[i] * other._data[i];
    return res;
  }

  // TODO: Abstract scalar operations as higher order
  Tensor Tensor::scalMul(const float& x) const {
    Tensor res(_dims);
    for (size_t i = 0; i < _data.size(); ++i)
      res._data[i] = _data[i] * x;
    return res;
  }

  Tensor Tensor::operator*(const float& x) const {
    return scalMul(x);
  }

  Tensor Tensor::scalDiv(const float& x) const {
    Tensor res(_dims);
    for (size_t i = 0; i < _data.size(); ++i)
      res._data[i] = _data[i] / x;
    return res;
  }

  Tensor Tensor::operator/(const float& x) const {
    return scalDiv(x);
  }

  Tensor Tensor::matMul(const Tensor& other) const {
    if (_dims.size() != 2 || other._dims.size() != 2)
      throw std::runtime_error("Matrix multiplication only supported for 2D tensors");

    size_t hidden = _dims[1];
    if (hidden != other._dims[0])
      throw std::runtime_error("Tensors must match in hidden dimension for multiplication");
  
    size_t rows = _dims[0];
    size_t cols = other._dims[1];
    Tensor result({rows, cols});
    for (size_t i = 0; i < rows; ++i) {
      for (size_t k = 0; k < hidden; ++k) {
        for (size_t j = 0; j < cols; ++j) {
          result({i, j}) += (*this)({i, k}) * other({k, j});
        }
      }
    }
    return result;
  }
  
  Tensor Tensor::operator*(const Tensor& other) const {
    return matMul(other);
  }

  // TODO: Abstract reductions as folds
  float Tensor::sum() const {
    float res = 0;
    for (auto elem : _data)
      res += elem;
    return res;
  }

  float Tensor::mean() const {
    return sum() / _data.size();
  }

  float Tensor::max() const {
    float res = _data[0];
    for (auto elem : _data) {
      if (elem > res)
        res = elem;
    }
    return res;
  }

  std::vector<float> Tensor::sum(int axis) const {
    throw std::runtime_error("Axis reductions not supported yet");
  }
  std::vector<float> Tensor::mean(int axis) const {
    throw std::runtime_error("Axis reductions not supported yet");
  }
  std::vector<float> Tensor::max(int axis) const {
    throw std::runtime_error("Axis reductions not supported yet");
  }
}
