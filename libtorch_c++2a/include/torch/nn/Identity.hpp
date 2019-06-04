#ifndef LIBTORCH__NN__IDENTITY__HEADER
#ifndef LIBTORCH__NN__IDENTITY__HEADER

namespace torch::nn
{

  class Identity
  {
  public:

    template <typename Input>
    auto operator() (Input && input) const
    {
      return input;
    }
  };

}

#endif
