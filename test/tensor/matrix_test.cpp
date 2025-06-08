#define CATCH_CONFIG_MAIN
#include "catch_amalgamated.hpp"
#include "tensor/matrix.hpp"

using tensor::Matrix;
using Catch::Approx;

TEST_CASE("Matrix element access and size", "[matrix]") {
    Matrix m(2, 3);
    REQUIRE(m.rows() == 2);
    REQUIRE(m.cols() == 3);

    m(0, 0) = 1.5f;
    m(1, 2) = -2.0f;
    REQUIRE(m(0, 0) == Approx(1.5f));
    REQUIRE(m(1, 2) == Approx(-2.0f));
}

TEST_CASE("Matrix addition", "[matrix]") {
    Matrix a(2, 2), b(2, 2);
    a(0,0) = 1; a(0,1) = 2; a(1,0) = 3; a(1,1) = 4;
    b(0,0) = 5; b(0,1) = 6; b(1,0) = 7; b(1,1) = 8;
    Matrix c = a + b;
    REQUIRE(c(0,0) == Approx(6));
    REQUIRE(c(0,1) == Approx(8));
    REQUIRE(c(1,0) == Approx(10));
    REQUIRE(c(1,1) == Approx(12));
}

TEST_CASE("Matrix subtraction", "[matrix]") {
    Matrix a(2, 2), b(2, 2);
    a(0,0) = 5; a(0,1) = 7; a(1,0) = 9; a(1,1) = 11;
    b(0,0) = 1; b(0,1) = 2; b(1,0) = 3; b(1,1) = 4;
    Matrix c = a - b;
    REQUIRE(c(0,0) == Approx(4));
    REQUIRE(c(0,1) == Approx(5));
    REQUIRE(c(1,0) == Approx(6));
    REQUIRE(c(1,1) == Approx(7));
}

TEST_CASE("Matrix scalar multiplication and division", "[matrix]") {
    Matrix a(2, 2);
    a(0,0) = 2; a(0,1) = 4; a(1,0) = 6; a(1,1) = 8;
    Matrix b = a * 2.0f;
    Matrix c = a / 2.0f;
    REQUIRE(b(0,0) == Approx(4));
    REQUIRE(b(1,1) == Approx(16));
    REQUIRE(c(0,1) == Approx(2));
    REQUIRE(c(1,0) == Approx(3));
}

TEST_CASE("Matrix multiplication", "[matrix]") {
    Matrix a(2, 2), b(2, 2);
    a(0,0) = 2; a(0,1) = 4; a(1,0) = 6; a(1,1) = 8;
    b(0,0) = 1; b(0,1) = 3; b(1,0) = 5; b(1,1) = 7;
    Matrix c = a * b;
    REQUIRE(c(0,0) == Approx(22));
    REQUIRE(c(0,1) == Approx(34));
    REQUIRE(c(1,0) == Approx(46));
    REQUIRE(c(1,1) == Approx(74));
}

TEST_CASE("Entire matrix reduction", "[matrix]") {
    Matrix a(2, 2);
    a(0,0) = 2; a(0,1) = 4; a(1,0) = 6; a(1,1) = 8;
    float allSum = a.sum();
    float allMean = a.mean();
    float allMax = a.max();
    REQUIRE(allSum == Approx(20));
    REQUIRE(allMean == Approx(5));
    REQUIRE(allMax == Approx(8));
}

TEST_CASE("Axis matrix reduction", "[matrix]") {
    Matrix a(3, 2);
    a(0,0) = 2; a(1,0) = 6; a(2,0) = 9;
    a(0,1) = 4; a(1,1) = 8; a(2,1) = 12;
    
    std::vector<float> rowSum = a.sum(0);
    std::vector<float> rowMean = a.mean(0);
    std::vector<float> rowMax = a.max(0);
    REQUIRE(rowSum[0] == Approx(17));
    REQUIRE(rowSum[1] == Approx(24));
    REQUIRE(rowMean[0] == Approx(17.0f/3.0f));
    REQUIRE(rowMean[1] == Approx(8));
    REQUIRE(rowMax[0] == Approx(9));
    REQUIRE(rowMax[1] == Approx(12));

    std::vector<float> colSum = a.sum(1);
    std::vector<float> colMean = a.mean(1);
    std::vector<float> colMax = a.max(1);
    REQUIRE(colSum[0] == Approx(6));
    REQUIRE(colSum[1] == Approx(14));
    REQUIRE(colSum[2] == Approx(21));
    REQUIRE(colMean[0] == Approx(3));
    REQUIRE(colMean[1] == Approx(7));
    REQUIRE(colMean[2] == Approx(10.5f));
    REQUIRE(colMax[0] == Approx(4));
    REQUIRE(colMax[1] == Approx(8));
    REQUIRE(colMax[2] == Approx(12));
}