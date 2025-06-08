#define CATCH_CONFIG_MAIN
#include <catch_amalgamated.hpp>
#include "tensor/tensor.hpp"

using tensor::Tensor;
using Catch::Approx;

TEST_CASE("Tensor construction and element access", "[tensor]") {
    Tensor t({2, 3});
    REQUIRE(t.dim(0) == 2);
    REQUIRE(t.dim(1) == 3);
    t({0, 0}) = 1.5f;
    t({1, 2}) = -2.0f;
    REQUIRE(t({0, 0}) == Approx(1.5f));
    REQUIRE(t({1, 2}) == Approx(-2.0f));
}

TEST_CASE("Tensor addition", "[tensor]") {
    Tensor a({2, 2}), b({2, 2});
    a({0,0}) = 1; a({0,1}) = 2; a({1,0}) = 3; a({1,1}) = 4;
    b({0,0}) = 5; b({0,1}) = 6; b({1,0}) = 7; b({1,1}) = 8;
    Tensor c = a + b;
    REQUIRE(c({0,0}) == Approx(6));
    REQUIRE(c({0,1}) == Approx(8));
    REQUIRE(c({1,0}) == Approx(10));
    REQUIRE(c({1,1}) == Approx(12));
}

TEST_CASE("Tensor scalar multiplication and division", "[tensor]") {
    Tensor a({2, 2});
    a({0,0}) = 2; a({0,1}) = 4;
    a({1,0}) = 6; a({1,1}) = 8;
    Tensor b = a * 2.0f;
    REQUIRE(b({0,0}) == Approx(4));
    REQUIRE(b({1,1}) == Approx(16));
    Tensor c = a / 2.0f;
    REQUIRE(c({0,1}) == Approx(2));
    REQUIRE(c({1,0}) == Approx(3));
}

TEST_CASE("Tensor matrix multiplication", "[tensor]") {
    Tensor a({2, 3});
    Tensor b({3, 2});
    // Fill a
    a({0,0}) = 1; a({0,1}) = 2; a({0,2}) = 3;
    a({1,0}) = 4; a({1,1}) = 5; a({1,2}) = 6;
    // Fill b
    b({0,0}) = 7; b({0,1}) = 8;
    b({1,0}) = 9; b({1,1}) = 10;
    b({2,0}) = 11; b({2,1}) = 12;
    Tensor c = a * b;
    REQUIRE(c.dim(0) == 2);
    REQUIRE(c.dim(1) == 2);
    REQUIRE(c({0,0}) == Approx(58));
    REQUIRE(c({0,1}) == Approx(64));
    REQUIRE(c({1,0}) == Approx(139));
    REQUIRE(c({1,1}) == Approx(154));
}

TEST_CASE("Tensor throws on shape mismatch", "[tensor]") {
    Tensor a({2, 2}), b({3, 2});
    REQUIRE_THROWS_AS(a + b, std::runtime_error);
    REQUIRE_THROWS_AS(a * b, std::runtime_error);
}
