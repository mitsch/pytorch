#ifndef LIBTORCH__TENSOR_LIKE__HEADER
#define LIBTORCH__TENSOR_LIKE__HEADER

#include <torch/Concepts.hpp>

namespace torch
{

    template <typename T>
    struct TensorValue
    {
        using Type = T::Value;
    };

    template <typename T>
    struct TensorIndices
    {
        using Type = T::Indices;
    };



    template <typename T>
    concept bool TensorLike = requires (T const& tensor, int index)
    {
        typename TensorValue<T>::Type;
        typename TensorIndices<T>::Type;
        {At(typename TensorIndices<T>::Type{}, tensor)};
        {Lengths(tensor)} -> std::vector<int>;
        {Map(tensor, [](typename TensorValue<T>::Type const&){return 0;})} -> TensorLike;
        {Project(tensor, {}, )};
    };

    namespace detail
    {
        /// TensorLike object \a T implementing interface via methods
        template <typename T>
        concept bool TensorLikeObject = requires (T const& tensor, int index)
        {
            {tensor.Rank()} -> int;
            {tensor.Length(index)} -> int;
        };
    }

    /// Returns the rank of \a tensor which implements it via method
    ///
    template <detail::TensorLikeObject T>
    int Rank (T const& tensor)
    {
        return tensor.Rank();
    }

    /// Returns the length of \a dimension from \tensor which implements it via
    /// method
    ///
    template <detail::TensorLikeObject T>
    int Length (T const& tensor, int dimension)
    {
        return tensor.Length(dimension);
    }





    template <typename T>
    concept bool NumericalTensorLike =
        TensorLike<T>
        and Numerical<TensorValue<T>>
        and requires(T const& tensor, TensorDimensions<T> const& dimensions)
    {
        {Product(tensor, dimensions)} -> T;
        {Sum(tensor, dimensions)} -> T;
        {Median(tensor, dimensions)} -> T;
        {Mode(tensor, dimensions)} -> T;
        {Kth(tensor, dimensions, ...)} -> T;
        {Mean(tensor, dimensions)} -> T;
        {StandardDerivate(tensor, dimensions)} -> T;
        {Variance(tensor, dimensions)} -> T;
    };

    template <TensorLike T>
    concept bool LogicalTensorLike = requires(T const& tensor)
    {
        {And(tensor, tensor)} -> LogicalTensorLike;
        {Or(tensor, tensor)} -> LogicalTensorLike;
        {XOr(tensor, tensor)} -> LogicalTensorLike;
        {Not(tensor)} -> LogicalTensorLike;
    };




    template <typename A, typename B>
        requires TensorLike<A>
             and TensorLike<B>
             and BiMappable<A, B, TensorValue<A>, TensorValue<B>>
             and Numerical<TensorValue<A>>
             and Numerical<TensorValue<B>>
    auto Add (A const& first, B const& second)
    {
        auto const adder = [](auto const& first, auto const& second)
        {
            return Add(first, second);
        };
        return Map(adder, first, second);
    }

    /// Subtracts \a second from \a first
    ///
    /// For any generic tensor like container \a first and \a second which are
    /// holding numerical values and support binary mapping, the subtraction is
    /// defined by mapping elementwise the values from \a first and \a second
    /// to the subtraction of these elements.
    ///
    template <typename A, typename B>
        requires TensorLike<A>
             and TensorLike<B>
             and BiMappable<A, B, TensorValue<A>, TensorValue<B>>
             and Numerical<TensorValue<A>>
             and Numerical<TensorValue<B>>
    auto Subtract (A const& first, B const& second)
    {
        auto const subtracter = [](auto const& first, auto const& second)
        {
            return Subtract(first, second);
        };
        return Map(subtracter, first, second);
    }

    /// Negates \a tensor
    ///
    /// For any \a tensor which is holding numerical values and supports unary
    /// mapping, the negation is defined by mapping each element to its negated
    /// value.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto Negate (A const& tensor)
    {
        auto const negater = [](auto const& element)
        {
            return Negate(element);
        };
        return Map(negater, tensor);
    }

    /// Absolutes \a tensor
    ///
    /// For any \a tensor which is holding numerical values and supports unary
    /// mapping, the absolution is defined by mapping each element to its
    /// absolute value.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto Absolute (A const& tensor)
    {
        auto const absoluter = [](auto const& element)
        {
            return Absolute(element);
        };
        return Map(absolute, tensor);
    }



    /// Divides elementwise \a first by \a second
    ///
    /// For any generic tensor like container \a first and \a second which are
    /// holding numerical values and support binary mapping, the division is
    /// defined by mapping elementwise the values from \a first and \a second
    /// to the division of these elements.
    ///
    template <typename A, typename B>
        requires TensorLike<A>
             and TensorLike<B>
             and BiMappable<A, B, TensorValue<A>, TensorValue<B>>
             and Numerical<TensorValue<A>>
             and Numerical<TensorValue<B>>
    auto CDiv (A const& first, B const& second)
    {
        auto const divider = [](auto const& first, auto const& second)
        {
            return CDiv(first, second);
        };
        return Map(divider, first, second);
    }

    /// Multiplies elementwise \a first by \a second
    ///
    /// For any generic tensor like container \a first and \a second which are
    /// holding numerical values and support binary mapping, the multiplication
    /// is defined by mapping elementwise the values from \a first and \a second
    /// to the multiplication of these elements.
    ///
    template <typename A, typename B>
        requires TensorLike<A>
             and TensorLike<B>
             and BiMappable<A, B, TensorValue<A>, TensorValue<B>>
             and Numerical<TensorValue<A>>
             and Numerical<TensorValue<B>>
    auto CMul (A const& first, B const& second)
    {
        auto const multiplier = [](auto const& first, auto const& second)
        {
            return CMul(first, second);
        };
        return Map(multiplier, first, second);
    }




    /// Elementwise rounding for \a tensor
    ///
    /// For any tensor-like \a tensor which is holding numerical values and
    /// which is supporting unary mapping, the rounding is defined by the
    /// elementwise rounding.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto Round (A const& tensor)
    {
        auto const mapper = [](auto const& element)
        {
            return Round(element);
        };
        return Map(mapper, tensor);
    }

    /// Elementwise flooring for \a tensor
    ///
    /// For any tensor-like \a tensor which is holding numerical values and
    /// which is supporting unary mapping, the flooring is defined by the
    /// elementwise flooring.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto Floor (A const& tensor)
    {
        auto const mapper = [](auto const& element)
        {
            return Floor(element);
        };
        return Map(mapper, tensor);
    }

    /// Elementwise ceiling for \a tensor
    ///
    /// For any tensor-like \a tensor which is holding numerical values and
    /// which is supporting unary mapping, the ceiling is defined by the
    /// elementwise ceiling.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto Ceil (A const& tensor)
    {
        auto const mapper = [](auto const& element)
        {
            return Ceil(element);
        };
        return Map(mapper, tensor);
    }

    /// Elementwise truncation for \a tensor
    ///
    /// For any tensor-like \a tensor which is holding numerical values and
    /// which is supporting unary mapping, the truncation is defined by the
    /// elementwise truncation.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto Truncate (A const& tensor)
    {
        auto const mapper = [](auto const& element)
        {
            return Truncate(element);
        };
        return Map(mapper, tensor);
    }




    /// Elementwise exponential for \a tensor
    ///
    /// For any tensor-like \a tensor which is holding numerical values and
    /// which is supporting unary mapping, the exponential is defined by the
    /// elementwise exponential.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto Exponential (A const& tensor)
    {
        auto const mapper = [](auto const& element)
        {
            return Exponential(element);
        };
        return Map(mapper, tensor);
    }

    /// Elementwise exponential with base two for \a tensor
    ///
    /// For any tensor-like \a tensor which is holding numerical values and
    /// which is supporting unary mapping, the exponential-based-two is defined
    /// by the elementwise exponential-based-two.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto Exponential2 (A const& tensor)
    {
        auto const mapper = [](auto const& element)
        {
            return Exponential2(element);
        };
        return Map(mapper, tensor);
    }

    /// Elementwise exponential mapping minus one for \a tensor
    ///
    /// For any tensor-like \a tensor which is holding numerical values and
    /// which is supporting unary mapping, the exponential-minus-one is defined
    /// by the elementwise exponential-minus-one.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto ExponentialMinus1 (A const& tensor)
    {
        auto const mapper = [](auto const& element)
        {
            return ExponentialMinus1(element);
        };
        return Map(mapper, tensor);
    }

    /// Elementwise logarithm with base two for \a tensor
    ///
    /// For any tensor-like \a tensor which is holding numerical values and
    /// which is supporting unary mapping, the logarithm-based-two is defined by
    /// the elementwise logarithm-based-two.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto Logarithm2 (A const& tensor)
    {
        auto const mapper = [](auto const& element)
        {
            return Logarithm2(element);
        };
        return Map(mapper, tensor);
    }

    /// Elementwise logarithm naturalis for \a tensor
    ///
    /// For any tensor-like \a tensor which is holding numerical values and
    /// which is supporting unary mapping, the logarithm-naturalis is defined by
    /// the elementwise logarithm-naturalis.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto Logarithm (A const& tensor)
    {
        auto const mapper = [](auto const& element)
        {
            return Logarithm(element);
        };
        return Map(mapper, tensor);
    }

    /// Elementwise logarithm with base ten for \a tensor
    ///
    /// For any tensor-like \a tensor which is holding numerical values and
    /// which is supporting unary mapping, the logarithm-based-ten is defined by
    /// the elementwise logarithm-based-ten.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto Logarithm10 (A const& tensor)
    {
        auto const mapper = [](auto const& element)
        {
            return Logarithm10(element);
        };
        return Map(mapper, tensor);
    }

    /// Elementwise logarithm with base two for \a tensor
    ///
    /// For any tensor-like \a tensor which is holding numerical values and
    /// which is supporting unary mapping, the logarithm-based-two is defined by
    /// the elementwise logarithm-based-two.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto Logarithm2 (A const& tensor)
    {
        auto const mapper = [](auto const& element)
        {
            return Logarithm2(element);
        };
        return Map(mapper, tensor);
    }

    /// Elementwise logarithm naturalis plus one for \a tensor
    ///
    /// For any tensor-like \a tensor which is holding numerical values and
    /// which is supporting unary mapping, the logarithm-plus-one is defined by
    /// the elementwise logarithm-plus-one.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto LogarithmPlus1 (A const& tensor)
    {
        auto const mapper = [](auto const& element)
        {
            return LogarithmPlus1(element);
        };
        return Map(mapper, tensor);
    }

    /// Elementwise maximum of \a left and \a right tensor
    ///
    /// For any tensor-like \a left and \a right which are holding identical and
    /// numerical values and which are supporting binary mapping, the maximum
    /// is defined by the elementwise maximum.
    template <typename A, typename B>
        requires TensorLike<A>
             and TensorLike<B>
             and BiMappable<A, B, TensorValue<A>, TensorValue<B>>
             and std::is_same_v<TensorValue<A>, TensorValue<B>>
             and Numerical<TensorValue<A>>
             and Numerical<TensorValue<B>>
    auto Max (A const& left, B const& right)
    {
        auto const mapper = [](auto const& left, auto const& right)
        {
            return Max(left, right);
        };
        return Map(mapper, left, right);
    }

    /// Elementwise minimum of \a left and \a right tensor
    ///
    /// For any tensor-like \a left and \a right which are holding identical and
    /// numerical values and which are supporting binary mapping, the minimum
    /// is defined by the elementwise minimum.
    template <typename A, typename B>
        requires TensorLike<A>
             and TensorLike<B>
             and BiMappable<A, B, TensorValue<A>, TensorValue<B>>
             and std::is_same_v<TensorValue<A>, TensorValue<B>>
             and Numerical<TensorValue<A>>
             and Numerical<TensorValue<B>>
    auto Min (A const& left, B const& right)
    {
        auto const mapper = [](auto const& left, auto const& right)
        {
            return Min(left, right);
        };
        return Map(mapper, left, right);
    }

    /// Signum of tensor-like \a tensor
    ///
    /// For any tensor-like \a tensor which is holding numerical values and
    /// supports unary mapping, the signum is defined by the elementwise signum.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto Signum (A const& tensor)
    {
        auto const mapper = [](auto const& element)
        {
            return Signum(element);
        };
        return Map(mapper, tensor);
    }

    /// Comparator if \a left is greater than \a right
    ///
    /// For any tensor-like \a left and \a right which are holding identical and
    /// numerical values and support binary mapping, the greater comparison is
    /// defined by the elementwise greater comparison.
    ///
    template <typename A, typename B>
        requires TensorLike<A>
             and TensorLike<B>
             and BiMappable<A, B, TensorValue<A>, TensorValue<B>>
             and std::is_same_v<TensorValue<A>, TensorValue<B>>
             and Numerical<TensorValue<A>>
             and Numerical<TensorValue<B>>
    auto Greater (A const& left, B const& right)
    {
        auto const comparator = [](auto const& left, auto const& right)
        {
            return Greater(left, right);
        };
        return Map(comparator, left, right);
    }

    /// Comparator if \a left is greater or equal to \a right
    ///
    /// For any tensor-like \a left and \a right which are holding identical and
    /// numerical values and support binary mapping, the greater-or-equal
    /// comparison is defined by the elementwise greater-or-equal comparison.
    ///
    template <typename A, typename B>
        requires TensorLike<A>
             and TensorLike<B>
             and BiMappable<A, B, TensorValue<A>, TensorValue<B>>
             and std::is_same_v<TensorValue<A>, TensorValue<B>>
             and Numerical<TensorValue<A>>
             and Numerical<TensorValue<B>>
    auto GreaterEqual (A const& left, B const& right)
    {
        auto const comparator = [](auto const& left, auto const& right)
        {
            return GreaterEqual(left, right);
        };
        return Map(comparator, left, right);
    }

    /// Comparator if \a left is less than \a right
    ///
    /// For any tensor-like \a left and \a right which are holding identical and
    /// numerical values and support binary mapping, the less comparison is
    /// defined by the elementwise less comparison.
    ///
    template <typename A, typename B>
        requires TensorLike<A>
             and TensorLike<B>
             and BiMappable<A, B, TensorValue<A>, TensorValue<B>>
             and std::is_same_v<TensorValue<A>, TensorValue<B>>
             and Numerical<TensorValue<A>>
             and Numerical<TensorValue<B>>
    auto Less (A const& left, B const& right)
    {
        auto const comparator = [](auto const& left, auto const& right)
        {
            return Less(left, right);
        };
        return Map(comparator, left, right);
    }

    /// Comparator if \a left is less or equal to \a right
    ///
    /// For any tensor-like \a left and \a right which are holding identical and
    /// numerical values and support binary mapping, the less-or-equal
    /// comparison is defined by the elementwise less-or-equal comparison.
    ///
    template <typename A, typename B>
        requires TensorLike<A>
             and TensorLike<B>
             and BiMappable<A, B, TensorValue<A>, TensorValue<B>>
             and std::is_same_v<TensorValue<A>, TensorValue<B>>
             and Numerical<TensorValue<A>>
             and Numerical<TensorValue<B>>
    auto LessEqual (A const& left, B const& right)
    {
        auto const comparator = [](auto const& left, auto const& right)
        {
            return LessEqual(left, right);
        };
        return Map(comparator, left, right);
    }

    /// Comparator if \a left is less or greater than \a right
    ///
    /// For any tensor-like \a left and \a right which are holding identical and
    /// numerical values and support binary mapping, the less-or-greater
    /// comparison is defined by the elementwise less-or-greater comparison.
    ///
    template <typename A, typename B>
        requires TensorLike<A>
             and TensorLike<B>
             and BiMappable<A, B, TensorValue<A>, TensorValue<B>>
             and std::is_same_v<TensorValue<A>, TensorValue<B>>
             and Numerical<TensorValue<A>>
             and Numerical<TensorValue<B>>
    auto LessGreater (A const& left, B const& right)
    {
        auto const comparator = [](auto const& left, auto const& right)
        {
            return LessGreater(left, right);
        };
        return Map(comparator, left, right);
    }

    /// Sine of \a tensor
    ///
    /// For any \a tensor which is holding numerical values and supports unary
    /// mapping, the sine is defined by its elementwise sine.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto Sine (A const& tensor)
    {
        auto const mapper = [](auto const& value)
        {
            return Sine(value);
        };
        return Map(mapper, tensor);
    }

    /// Cosine of \a tensor
    ///
    /// For any \a tensor which is holding numerical values and supports unary
    /// mapping, the cosine is defined by its elementwise cosine.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto Cosine (A const& tensor)
    {
        auto const mapper = [](auto const& value)
        {
            return Cosine(value);
        };
        return Map(mapper, tensor);
    }

    /// Tangent of \a tensor
    ///
    /// For any \a tensor which is holding numerical values and supports unary
    /// mapping, the sine is defined by its elementwise sine.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto Tangent (A const& tensor)
    {
        auto const mapper = [](auto const& value)
        {
            return Tangent(value);
        };
        return Map(mapper, tensor);
    }

    /// Arcus sine of \a tensor
    ///
    /// For any \a tensor which is holding numerical values and supports unary
    /// mapping, the arcus sine is defined by its elementwise arcus sine.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto ArcusSine (A const& tensor)
    {
        auto const mapper = [](auto const& value)
        {
            return ArcusSine(value);
        };
        return Map(mapper, tensor);
    }

    /// Arcus cosine of \a tensor
    ///
    /// For any \a tensor which is holding numerical values and supports unary
    /// mapping, the arcus cosine is defined by its elementwise arcus cosine.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto ArcusCosine (A const& tensor)
    {
        auto const mapper = [](auto const& value)
        {
            return ArcusCosine(value);
        };
        return Map(mapper, tensor);
    }

    /// Arcus tangent of \a tensor
    ///
    /// For any \a tensor which is holding numerical values and supports unary
    /// mapping, the arcus tangent is defined by its elementwise arcus tangent.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto ArcusTangent (A const& tensor)
    {
        auto const mapper = [](auto const& value)
        {
            return ArcusTangent(value);
        };
        return Map(mapper, tensor);
    }

    /// Arcus tangent alternative of \a tensor
    ///
    /// For any \a tensor which is holding numerical values and supports unary
    /// mapping, the arcus tangent alternative is defined by its elementwise
    /// arcus tangent alternative.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto ArcusTangent2 (A const& tensor)
    {
        auto const mapper = [](auto const& value)
        {
            return ArcusTangent2(value);
        };
        return Map(mapper, tensor);
    }

    /// Sine hyperbolic of each element in \a tensor
    ///
    /// For any \a tensor which is holding numerical values and supports unary
    /// mapping, the sine hyperbolic is defined by mapping each value to its
    /// sine hyperbolic.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto SineHyperbolic (A const& tensor)
    {
        auto const mapper = [](auto const& value)
        {
            return SineHyperbolic(value);
        };
        return Map(mapper, tensor);
    }

    /// Cosine hyperbolic of each element in \a tensor
    ///
    /// For any \a tensor which is holding numerical values and supports unary
    /// mapping, the cosine hyperbolic is defined by mapping each value to its
    /// cosine hyperbolic.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto CosineHyperbolic (A const& tensor)
    {
        auto const mapper = [](auto const& value)
        {
            return CosineHyperbolic(value);
        };
        return Map(mapper, tensor);
    }

    /// Tangent hyperbolic of each element in \a tensor
    ///
    /// For any \a tensor which is holding numerical values and supports unary
    /// mapping, the tangent hyperbolic is defined by mapping each value to its
    /// tangent hyperbolic.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto TangentHyperbolic (A const& tensor)
    {
        auto const mapper = [](auto const& value)
        {
            return TangentHyperbolic(value);
        };
        return Map(mapper, tensor);
    }

    /// Arcus sine hyperbolic of each element in \a tensor
    ///
    /// For any \a tensor which is holding numerical values and supports unary
    /// mapping, the arcus sine hyperbolic is defined by mapping each value to
    /// its arcus sine hyperbolic.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto ArcusSineHyperbolic (A const& tensor)
    {
        auto const mapper = [](auto const& value)
        {
            return ArcusSineHyperbolic(value);
        };
        return Map(mapper, tensor);
    }

    /// Arcus cosine hyperbolic of each element in \a tensor
    ///
    /// For any \a tensor which is holding numerical values and supports unary
    /// mapping, the arcus cosine hyperbolic is defined by mapping each value to
    /// its arcus cosine hyperbolic.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto ArcusCosineHyperbolic (A const& tensor)
    {
        auto const mapper = [](auto const& value)
        {
            return ArcusCosineHyperbolic(value);
        };
        return Map(mapper, tensor);
    }

    /// Arcus tangent hyperbolic of each element in \a tensor
    ///
    /// For any \a tensor which is holding numerical values and supports unary
    /// mapping, the arcus tangent hyperbolic is defined by mapping each value
    /// to its arcus tangent hyperbolic.
    ///
    template <typename A>
        requires TensorLike<A>
             and Mappable<A, TensorValue<A>>
             and Numerical<TensorValue<A>>
    auto ArcusTangentHyperbolic (A const& tensor)
    {
        auto const mapper = [](auto const& value)
        {
            return ArcusTangentHyperbolic(value);
        };
        return Map(mapper, tensor);
    }


    /// Finds maximum values
    ///
    template <NumericalTensorLike A>
    auto Maximum (A::Mask const& mask, bool keep_dimensions, A const& tensor)
    {
        return Project(tensor, mask, keep_dimensions, []
            (auto const& left, auto const& right)
        {
            return Maximum(left, right);
        });
    }

    template <NumericalTensorLike A>
    auto Minimum (A::Mask const& mask, A const& tensor)
    {
        return Project(tensor, mask, keep_dimensions, []
            (auto const& left, auto const& right)
        {
            return Minimum(left, right);
        });
    }

    template <NumericalTensorLike A>
    auto Product (A::Mask const& mask, A const& tensor)
    {
        return Project(tensor, mask, keep_dimensions, []
            (auto const& left, auto const& right)
        {
            return Multiply(left, right);
        });
    }

    template <NumericalTensorLike A>
    auto Sum (A::Mask const& mask, A const& tensor)
    {
        return Project(tensor, mask, keep_dimensions, []
            (auto const& left, auto const& right)
        {
            return Add(left, right);
        });
    }

    template <NumericalTensorLike A>
    auto Mean (TensorDimensions<A> const& mask, A const& tensor)
    {
        return Divide(Sum(mask, tensor), );
    }

    template <typename A, typename C>
    auto Partition (TensorDimensions<A> const& mask, A const& tensor, C comparer)
    {

    }

    template <NumericalTensorLike A>
    auto Median (TensorDimensions<A> const& mask, A const& tensor)
    {
        return
    }







    /// Conjuncts elementwise \a first and \a second
    ///
    /// For any generic tensor like container \a first and \a second which are
    /// holding identical and logical values and support binary mapping, the
    /// conjunction is defined by the elementwise conjunction.
    ///
    template <typename A, typename B>
        requires TensorLike<A>
             and TensorLike<B>
             and BiMappable<A, B, TensorValue<A>, TensorValue<B>>
             and std::is_same_v<TensorValue<A>, TensorValue<B>>
             and Logical<TensorValue<A>>
             and Logical<TensorValue<B>>
    auto And (A const& first, B const& second)
    {
        auto const conjuncter = [](auto const& first, auto const& second)
        {
            return And(first, second);
        };
        return Map(conjuncter, first, second);
    }

    /// Disjuncts elementwise \a first and \a second
    ///
    /// For any generic tensor like container \a first and \a second which are
    /// holding identical and logical values and support binary mapping, the
    /// disjunction is defined by the elementwise disjunction.
    ///
    template <typename A, typename B>
        requires TensorLike<A>
             and TensorLike<B>
             and BiMappable<A, B, TensorValue<A>, TensorValue<B>>
             and std::is_same_v<TensorValue<A>, TensorValue<B>>
             and Logical<TensorValue<A>>
             and Logical<TensorValue<B>>
    auto Or (A const& first, B const& second)
    {
        auto const disjuncter = [](auto const& first, auto const& second)
        {
            return Or(first, second);
        };
        return Map(disjuncter, first, second);
    }

    /// Exclusively disjuncts elementwise \a first and \a second
    ///
    /// For any generic tensor like container \a first and \a second which are
    /// holding identical and logical values and support binary mapping, the
    /// exclusive disjunction is defined by the elementwise exclusive
    /// disjunction.
    ///
    template <typename A, typename B>
        requires TensorLike<A>
             and TensorLike<B>
             and BiMappable<A, B, TensorValue<A>, TensorValue<B>>
             and std::is_same_v<TensorValue<A>, TensorValue<B>>
             and Logical<TensorValue<A>>
             and Logical<TensorValue<B>>
    auto XOr (A const& first, B const& second)
    {
        auto const disjuncter = [](auto const& first, auto const& second)
        {
            return XOr(first, second);
        };
        return Map(disjuncter, first, second);
    }

    /// Logically negates elementwise \a tensor
    ///
    /// For any generic \a tensor which is holding identical and logical values
    /// and supports mapping, the logically negation is defined by the
    /// elementwise logical negation.
    ///
    template <typename A, typename B>
        requires TensorLike<A>
             and TensorLike<B>
             and BiMappable<A, B, TensorValue<A>, TensorValue<B>>
             and std::is_same_v<TensorValue<A>, TensorValue<B>>
             and Logical<TensorValue<A>>
             and Logical<TensorValue<B>>
    auto Not (A const& tensor)
    {
        auto const negater = [](auto const& first, auto const& second)
        {
            return Not(first, second);
        };
        return Map(negater, tensor);
    }



}

#endif
