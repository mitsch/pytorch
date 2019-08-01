#ifndef LIBTORCH__LOGICAL__HEADER
#define LIBTORCH__LOGICAL__HEADER

namespace torch
{

    template <typename L>
    concept bool Logical = requires(L const& logical)
    {
        {And(logical, logical)} -> Logical;
        {Or(logical, logical)} -> Logical;
        {XOr(logical, logical)} -> Logical;
        {Not(logical)} -> Logical;
        {Imply(logical, logical)} -> Logical;
    };

    template <typename L>
    concept bool MonoidLogical = requires(L const& logical)
    {
        {And(logical, logical)} -> L;
        {Or(logical, logical)} -> L;
        {XOr(logical, logical)} -> L;
        {Not(logical)} -> L;
        {Imply(logical, logical)} -> L;
    };


    bool And (bool left, bool right) {return left and right;}
    bool Or (bool left, bool right) {return left or right;}
    bool XOr (bool left, bool right) {return left xor right;}
    bool Not (bool value) {return not value;}
    bool Imply (bool left, bool right) {return not left or right;}

}

#endif
