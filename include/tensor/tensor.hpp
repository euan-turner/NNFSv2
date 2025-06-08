#pragma once

#include <vector>
#include <stdexcept>

namespace tensor {
  class Tensor {
    private:
      std::vector<size_t> _dims;
      std::vector<size_t> _strides;
      std::vector<float> _data;

      bool elementWiseCompatible(const Tensor& other) const {
        return std::equal(_dims.begin(), _dims.end(), other._dims.begin());
      }
    
    public:

      explicit Tensor(std::vector<size_t> dims);
      // Rule of Five
      // 1. Copy Constructor
      Tensor(const Tensor& other);

      // 2. Copy assignment operator
      Tensor& operator=(const Tensor& other);

      // 3. Move constructor
      Tensor(Tensor&& other) noexcept;

      // 4. Move assignment operator
      Tensor& operator=(Tensor&& other) noexcept;

      // 5. Destructor
      ~Tensor();

      // element access
      float& operator()(std::vector<size_t> indices);
      const float& operator()(std::vector<size_t> indices) const;

      // dim access
      size_t dim(uint axis) const;
      std::vector<size_t> dims() const {
        return _dims;
      }

      // stride access
      size_t stride(uint axis) const;
      std::vector<size_t> strides() const {
        return _strides;
      }

      size_t size() const;

      // operations 
      // TODO: make implicitly broadcast as appropriate
      
      // element-wise addition
      Tensor add(const Tensor& other) const;
      Tensor operator+(const Tensor& other) const;
      // element-wise subtraction
      Tensor sub(const Tensor& other) const;
      Tensor operator-(const Tensor& other) const;
      // element-wise multiplication
      Tensor hadamard(const Tensor& other) const;

      // scalar multiplication
      Tensor scalMul(const float& x) const;
      Tensor operator*(const float &x) const;
      // scalar division
      Tensor scalDiv(const float& x) const;
      Tensor operator/(const float &x) const;

      // Tensor multiplication
      Tensor matMul(const Tensor& other) const;
      Tensor operator*(const Tensor& other) const;

      // reduction
      float sum() const;
      float mean() const;
      float max() const;

      std::vector<float> sum(int axis) const;
      std::vector<float> mean(int axis) const;
      std::vector<float> max(int axis) const;

  };
}