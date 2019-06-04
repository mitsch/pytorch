#ifndef LIBTORCH__NN__LINEAR__HEADER
#define LIBTORCH__NN__LINEAR__HEADER

namespace torch::nn
{

  template <typename W, typename B>
  class Linear
  {
  public:

    Linear (W weight, B bias)
      : weight(std::move(weight))
      , bias(std::move(bias))
    {}

    template <typename I>
    auto operator() (I && input) const
    {
      return weight * std::forward<Input>(input) + bias;
    }

  private:

    W weight;
    B bias;
  };

  template <typename W, typename B> Linear (W, B) -> Linear<W, B>;

}

#endif
