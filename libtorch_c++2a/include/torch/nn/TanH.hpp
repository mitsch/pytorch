#ifndef LIBTORCH__NN__TANH__HEADER
#define LIBTORCH__NN__TANH__HEADER

namespace torch::nn
{

  class TanH
  {

  public:

    template <typename Input>
      requires Trigonometrical<Input>
    auto operator() (Input && input) const
    {
      return tanh(std::forward<Input>(input));
    }

  };

}

#endif
