#ifndef LIBTORCH__TIME_BATCH__HEADER
#define LIBTORCH__TIME_BATCH__HEADER

namespace torch
{

  template <TensorLike T>
  class TimeBatch
  {

  public:

    constexpr int Rank () const
    {
      assert(tensor.Rank() == 3);
      return 3;
    }

    int Length (int dimension) const
    {
      assert(dimension >= 0);
      assert(dimension < 3);
      assert(tensor.Rank() == 3);
      return tensor.Length(dimension);
    }

    int BatchSize () const
    {
      assert(tensor.Rank() == 3);
      return tensor.Length(0);
    }

    int TimeSize () const
    {
      assert(tensor.Rank() == 3);
      return tensor.Length(1);
    }

    int FeatureSize () const
    {
      assert(tensor.Rank() == 3);
      return tensor.Length(2);
    }

    TimeBatch Contiguous () const&
    {
      return {tensor.Contiguous()};
    }

    TimeBatch Contiguous () &&
    {
      return {std::move(tensor).Contiguous()};
    }


    auto const& At (int batch, int time, int )

  private:

    T tensor;
  };

}

#endif
