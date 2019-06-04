#ifndef LIBTORCH__NN__MODULELIKE__HEADER
#define LIBTORCH__NN__MODULELIKE__HEADER

namespace torch::nn
{

  template <typename M, typename I>
  concept bool ModuleLike = requires(M const& module, I const& input)
  {
    {module(input)};
  };


}

#endif
