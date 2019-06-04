#ifndef LIBTORCH__NN__SEQUENTIAL__HEADER
#define LIBTORCH__NN__SEQUENTIAL__HEADER

#include <torch/nn/ModuleLike.hpp>

namespace torch::nn
{

  template <typename ... Ms>
  class Sequential
  {

  public:

    Sequential (Ms ... modules)
      : modules{std::move(modules) ...}
    {}

    template <typename Input>
    auto operator (Input && input) const
    {
      return DoNext<0>(std::forward<Input>(input));
    }

  private:

    template <int index, typename Input>
    auto DoNext (Input && input) const
    {
      if constexpr (index < sizeof...(modules))
      {
        return DoNext(std::get<index>(modules)(std::forward<Input>(input)));
      }
      else
      {
        return input;
      }
    }

    std::tuple<Ms ...> modules;

  };

}

#endif
