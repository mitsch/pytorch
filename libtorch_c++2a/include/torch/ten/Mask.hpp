#ifndef LIBTORCH__TEN__MASK__HEADER
#define LIBTORCH__TEN__MASK__HEADER

namespace torch::ten
{

  template <typename ... Ms>
  class Mask
  {

  public:

    Mask<Ms ..., auto> Narrow (int dimension, int index, int count) const
    {

    }
  };

}

#endif
