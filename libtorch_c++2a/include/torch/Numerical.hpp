#ifndef LIBTORCH__NUMERICAL__HEADER
#define LIBTORCH__NUMERICAL__HEADER

#include <torch/Logical.hpp>

#include <math>

namespace torch
{

    template <typename N>
    concept bool Numerical = requires(N const& numerical)
    {
        // Algebraic functions:
        {Add(numerical, numerical)} -> Numerical;
        {Subtract(numerical, numerical)} -> Numerical;
        {Negate(numerical)} -> Numerical;
        {Absolute(numerical)} -> Numerical;
        {Multiply(numerical, numerical)} -> Numerical;
        {Divide(numerical, numerical)} -> Numerical;
        {Reciproce(numerical)} -> Numerical;
        {Remainder(numerical, numerical)} -> Numerical;
        {Fractional(numerical)} -> Numerical;
        {Modulo(numerical, numerical)} -> Numerical;

        // Nearest integer functions
        {Round(numerical)} -> Numerical;
        {Floor(numerical)} -> Numerical;
        {Ceil(numerical)} -> Numerical;
        {Truncate(numerical)} -> Numerical;

        // Power functions
        {Power(numerical, numerical)} -> Numerical;
        {SquareRoot(numerical)} -> Numerical;
        {CubicRoot(numerical)} -> Numerical;
        {Hypotenuse(numerical, numerical)} -> Numerical;

        // Exponential functions
        {Exponential(numerical)} -> Numerical;
        {Exponential2(numerical)} -> Numerical;
        {ExponentialMinus1(numerical)} -> Numerical;
        {Logarithm(numerical)} -> Numerical;
        {Logarithm10(numerical)} -> Numerical;
        {Logarithm2(numerical)} -> Numerical;
        {LogarithmPlus1(numerical)} -> Numerical;

        // Comparison functions
        {Max(numerical, numerical)} -> Numerical;
        {Min(numerical, numerical)} -> Numerical;
        {Signum(numerical)} -> Numerical;
        {Greater(numerical, numerical)} -> bool;
        {GreaterEqual(numerical, numerical)} -> Logical;
        {Less(numerical, numerical)} -> Logical;
        {LessEqual(numerical, numerical)} -> Logical;
        {LessGreater(numerical, numerical)} -> Logical;

        // Trigonometric functions
        {Sine(numerical)} -> Numerical;
        {Cosine(numerical)} -> Numerical;
        {Tangent(numerical)} -> Numerical;
        {ArcusSine(numerical)} -> Numerical;
        {ArcusCosine(numerical)} -> Numerical;
        {ArcusTangent(numerical)} -> Numerical;
        {ArcusTangent2(numerical)} -> Numerical;

        // Hyperbolic functions
        {SineHyperbolic(numerical)} -> Numerical;
        {CosineHyperbolic(numerical)} -> Numerical;
        {TangentHyperbolic(numerical)} -> Numerical;
        {ArcusSineHyperbolic(numerical)} -> Numerical;
        {ArcusCosineHyperbolic(numerical)} -> Numerical;
        {ArcusTangentHyperbolic(numerical)} -> Numerical;
    };

    template <typename N>
    concept bool MonoidNumerical = requires(N const& numerical)
    {
        // Algebraic functions:
        {Add(numerical, numerical)} -> N;
        {Subtract(numerical, numerical)} -> N;
        {Negate(numerical)} -> N;
        {Absolute(numerical)} -> N;
        {Multiply(numerical, numerical)} -> N;
        {Divide(numerical, numerical)} -> N;
        {Reciproce(numerical)} -> N;
        {Remainder(numerical, numerical)} -> N;
        {Fractional(numerical)} -> N;
        {Modulo(numerical, numerical)} -> N;

        // Nearest integer functions
        {Round(numerical)} -> N;
        {Floor(numerical)} -> N;
        {Ceil(numerical)} -> N;
        {Truncate(numerical)} -> N;

        // Power functions
        {Power(numerical, numerical)} -> N;
        {SquareRoot(numerical)} -> N;
        {CubicRoot(numerical)} -> N;
        {Hypotenuse(numerical, numerical)} -> N;

        // Exponential functions
        {Exponential(numerical)} -> N;
        {Exponential2(numerical)} -> N;
        {ExponentialMinus1(numerical)} -> N;
        {Logarithm(numerical)} -> N;
        {Logarithm10(numerical)} -> N;
        {Logarithm2(numerical)} -> N;
        {LogarithmPlus1(numerical)} -> N;

        // Comparison functions
        {Max(numerical, numerical)} -> N;
        {Min(numerical, numerical)} -> N;
        {Signum(numerical)} -> N;
        {Greater(numerical, numerical)} -> Logical;
        {GreaterEqual(numerical, numerical)} -> Logical;
        {Less(numerical, numerical)} -> Logical;
        {LessEqual(numerical, numerical)} -> Logical;
        {LessGreater(numerical, numerical)} -> Logical;

        // Trigonometric functions
        {Sine(numerical)} -> N;
        {Cosine(numerical)} -> N;
        {Tangent(numerical)} -> N;
        {ArcusSine(numerical)} -> N;
        {ArcusCosine(numerical)} -> N;
        {ArcusTangent(numerical)} -> N;
        {ArcusTangent2(numerical)} -> N;

        // Hyperbolic functions
        {SineHyperbolic(numerical)} -> N;
        {CosineHyperbolic(numerical)} -> N;
        {TangentHyperbolic(numerical)} -> N;
        {ArcusSineHyperbolic(numerical)} -> N;
        {ArcusCosineHyperbolic(numerical)} -> N;
        {ArcusTangentHyperbolic(numerical)} -> N;
    };

    float Add (float left, float right) {return left + right;}
    float Subtract (float left, float right) {return left - right;}
    float Negate (float value) {return -value;}
    float Absolute (float value) {return std::abs(value);}
    float Multiply (float left, float right) {return left * right;}
    float Divide (float left, float right) {return left / right;}
    float Reciproce (float value) {return 1.f / value;}
    float Remainder (float x, float y) {return std::remainder(x, y);}
    float Fractional (float value) {}
    float Modulo (float x, float y) {return std::fmod(x, y);}
    float Round (float value) {return std::round(value);}
    float Floor (float value) {return std::floor(value);}
    float Ceiling (float value) {return std::ceil(value);}
    float Truncate (float value) {return std::trunc(value);}
    float Power (float base, float exponent) {return std::pow(base, exponent);}
    float SquareRoot (float value) {return std::sqrt(value);}
    float CubicRoot (float value) {return std::cbrt(value);}
    float Hypotenuse (float x, float y) {return std::hypot(x, y);}
    float Exponential (float value) {return std::exp(value);}
    float Exponential2 (float value) {return std::exp2(value);}
    float ExponentialMinus1 (float value) {return std::expm1(value);}
    float Logarithm (float value) {return std::log(value);}
    float Logarithm10 (float value) {return std::log10(value);}
    float Logarithm2 (float value) {return std::log2(value);}
    float LogarithmPlus1 (float value) {return std::logp1(value);}
    float Max (float left, float right) {return std::fmax(left, right);}
    float Min (float left, float right) {return std::fmin(left, right);}
    float Signum (float value) {return value > 0f ? 1f : value < 0f ? -1f : 0f;}
    bool Greater (float x, float y) {return std::isgreater(x, y);}
    bool GreaterEqual (float x, float y) {return std::isgreaterequal(x, y);}
    bool Less (float x, float y) {return std::isless(x, y);}
    bool LessEqual (float x, float y) {return std::islessequal(x, y);}
    bool LessGreater (float x, float y) {return std::islessgreater(x, y);}
    float Sine (float value) {return std::sin(value);}
    float Cosine (float value) {return std::cos(value);}
    float Tangent (float value) {return std::tan(value);}
    float ArcSine(float value) {return std::asin(value);}
    float ArcCosine(float value) {return std::acos(value);}
    float ArcTangent(float value) {return std::atan(value);}
    float ArcTangent2(float value) {return std::atan2(value);}
    float SineHyperbolic (float value) {return std::sinh(value);}
    float CosineHyperbolic (float value) {return std::cosh(value);}
    float TangentHyperbolic (float value) {return std::tanh(value);}
    float ArcSineHyperbolic (float value) {return std::asinh(value);}
    float ArcCosineHyperbolic (float value) {return std::acosh(value);}
    float ArcTangentHyperbolic (float value) {return std::atanh(value);}
}

#endif
