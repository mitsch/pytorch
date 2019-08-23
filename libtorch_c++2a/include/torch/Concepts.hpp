#ifndef LIBTORCH__CONCEPTS__HEADER
#define LIBTORCH__CONCEPTS__HEADER

namespace torch
{

    template <typename T, typename ... As>
    concept bool Constructible = requires (As ... arguments)
    {
        {T(arguments ...)} -> T;
    };

    template <typename T>
    concept bool CopyConstructible = Constructible<T, T const&>;

    template <typename T>
    concept bool MoveConstructible = Constructible<T, T &&>;

    template <typename C, typename ... As>
    concept bool Callable = requires (C const& callee, As ... arguments)
    {
        {callee(arguments ...)};
    };

    template <typename C, typename R, typename ... As>
    concept bool CallableTo = requires (C const& callee, As ... arguments)
    {
        {callee(arguments ...)} -> R;
    };

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
