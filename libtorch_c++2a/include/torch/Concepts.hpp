#ifndef LIBTORCH__CONCEPTS__HEADER
#define LIBTORCH__CONCEPTS__HEADER

namespace torch
{

    template <typename M, typename T, typename ... Ts>
    concept Mappable = requires (M const& mapper, T const& head, Ts const& ... tail)
    {
        {Map(mapper, head, tail ...)};
    };

    template <typename P, typename T, typename ... Ts>
    concept Projectable = requires (P const& projector, T const& head, Ts const& ... tail)
    {
        {Project(projector, head, tail ...)};
    };


}

#endif
