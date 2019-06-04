#ifndef LIBTORCH__TEN__VECTOR_LIKE__HEADER
#define LIBTORCH__TEN__VECTOR_LIKE__HEADER

namespace torch::ten
{

    template <typename V>
    concept bool VectorLike = requires(V const& vector, int index)
    {
        {vector.At(index)};
        {vector.Length()} -> int;
    };

    template <VectorLike V>
    concept bool NumericalVectorLike = requires(V const& vector, int index)
    {
        {Add(vector, vector)} -> V;
        {Subtract(vector, vector)} -> V;
        {CDivide(vector, vector)} -> V;

    };

}

#endif
