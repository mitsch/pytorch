#ifndef LIBTORCH__TEN__TENSOR_LIKE__HEADER
#define LIBTORCH__TEN__TENSOR_LIKE__HEADER

namespace torch::ten
{

    template <typename T>
    concept bool TensorLike = requires(T const& tensor, int index)
    {
        {Rank(tensor)} -> int;
        {Length(tensor, index)} -> int;
        {At(tensor, std::vector<int>{})};
    };

    template <TensorLike T>
    using TensorValue = std::remove_cv_t<std::remove_reference_t<decltype(At(std::declval<T const&>(), std::vector<int>{}))>>;

    template <TensorLike T>
    concept bool NumericalTensorLike = requires(T const& tensor, TensorValue<T> const& value)
    {
      {Add(tensor, tensor)} -> TensorLike;
      {Subtract(tensor, tensor)} -> TensorLike;
      {CDiv(tensor, tensor)} -> TensorLike;
      {CMul(tensor, tensor)} -> TensorLike;
      {Mul(value, tensor)} -> TensorLike;
      {Div(tensor, value)} -> TensorLike;
      {Mul(tensor, tensor)} -> TensorLike;
      {Pow(value, tensor)} -> TensorLike;
      {Pow(tensor, value)} -> TensorLike;
      {Pow(tensor, tensor)} -> TensorLike;
      {Abs(tensor)} -> TensorLike;
      {Sign(tensor)} -> TensorLike;
      {Ceil(tensor)} -> TensorLike;
      {Floor(tensor)} -> TensorLike;
      {Log(tensor)} -> TensorLike;
      {Log1p(tensor)} -> TensorLike;
      {Negate(tensor)} -> TensorLike;
    };


    template <TensorLike T>
    concept bool LogicalTensorLike = requires(T const& tensor)
    {
        {And(tensor, tensor)} -> LogicalTensorLike;
        {Or(tensor, tensor)} -> LogicalTensorLike;
        {XOr(tensor, tensor)} -> LogicalTensorLike;
        {Not(tensor)} -> LogicalTensorLike;
    };

}

#endif
