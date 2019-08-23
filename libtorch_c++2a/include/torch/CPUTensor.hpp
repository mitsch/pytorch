#ifndef LIBTORCH__CPU_TENSOR__HEADER
#define LIBTORCH__CPU_TENSOR__HEADER

#include <vector>
#include <cassert>
#include <numeric>
#include <tuple>
#include <algorithm>

#include <torch/Concepts.hpp>

namespace torch
{

    template <typename T>
    class CPUTensor
    {
    public:

        struct PrefixDimensions
        {
            int count;
            PrefixDimensions () = delete;
            PrefixDimensions (int count) : count{count} {}
        };

        struct SuffixDimensions
        {
            int count;
            SuffixDimensions () = delete;
            SuffixDimensions (int count) : count{count} {}
        };

        /// Constructs an empty tensor
        ///
        CPUTensor () = default;

        /// Constructs a copy of that \a tensor
        ///
        CPUTensor (CPUTensor const& tensor) = default;

        /// Constructs a move of that \a tensor
        ///
        CPUTensor (CPUTensor && tensor) = default;

        /// Constructs by direct initialisation
        ///
        /// A tensor of dimensional \a lengths will be created which
        /// subsequently will be directly initialised by \a initialiser.
        ///
        template <typename I>
        requires CallableTo<I, T, std::vector<int> const&>
             and MoveConstructible<T>
        CPUTensor (std::vector<int> lengths, I initialiser)
        {
            assert(lengths.size() > 0);
            for (int length : lengths)
            {
                assert(length > 0);
            }

            values.reserve(Product(lengths));
            
            auto indices = std::vector<int>(0, lengths.size());
            while (true)
            {
                for (int index = 0; index < lengths.back(); ++index)
                { 
                    indices.back() = index;
                    values.emplace_back(initialiser(std::as_const(indices)));
                }
                auto iter_indices = indices.rbegin() + 1;
                auto iter_lengths = lengths.rbegin() + 1;
                while (iter_indices != indices.rend())
                {
                    ++*iter_indices;
                    if (*iter_indices >= *iter_lengths)
                    {
                        *iter_indices = 0;
                        ++iter_indices;
                        ++iter_lengths;
                    }
                    else
                    {
                        break;
                    }
                }
                if (iter_indices == indices.rend())
                {
                    break;
                }
            }

            this->lengths = std::move(lengths);
            this->strides.resize(this->lengths.size());
            std::partial_sum(std::begin(this->lengths), std::end(this->lengths), std::begin(this->strides));
            std::move_backward(std::begin(this->strides),
                               std::end(this->strides) - 1,
                               std::end(this->strides));
            this->strides[0] = 0;
        }

        /// Dimensional lengths
        ///
        std::vector<int> const& Lengths () const
        {
            return lengths;
        }

#if 0
        /// Single component on host at \a indices
        ///
        T const& At (std::vector<int> const& indices) const
        {
            assert(indices.size() == lengths.size());
            auto const offset = Offset(indices);
            return values[offset];
        }
#endif

        /// Visits each element with the respective indices
        ///
        template <typename V>
        requires Callable<V, T const&, std::vector<int> const&>
        void Visit (V visitor) const
        {
            std::vector<int> indices{0, lengths.size()};
            for (auto const& value : values)
            {
                visitor(value, indices);
                auto iter_indices = indices.rbegin();
                auto iter_lengths = lengths.rbegin();
                while (iter_indices != indices.rend())
                {
                    ++*iter_indices;
                    if (*iter_indices < *iter_lengths)
                    {
                        break;
                    }
                    *iter_indices = 0;
                    ++iter_indices;
                    ++iter_lengths;
                }
            }
        }

        /// Maps a copy of each component into a new CPUTensor
        ///
        template <typename M>
        // requires Callable<M, T const&>
        auto Map (M mapper) const
        {
            using Mapped = std::result_of_t<M(T const&)>;
            std::vector<Mapped> mapped_values;
            mapped_values.reserve(values.size());
            for (auto const& value : values)
            {
                mapped_values.emplace_back(mapper(value));
            }
            return CPUTensor<Mapped>{lengths, std::move(mapped_values)};
        }

#if 0
        /// Maps a copy of each component with its index into a new CPUTensor
        ///
        template <typename M>
        // requires Callable<M, T const&, std::vector<int> const&>
        auto MapIndexed (M mapper) const
        {
            using Mapped = std::result_of_t<M(T const&, std::vector<int> const&)>;
            std::vector<Mapped> mapped_values;
            mapped_values.reserve(values.size());
            // TODO efficiently iterate over the indices whilst iterating over
            // the values ...
            return CPUTensor<Mapped>{lengths, std::move(mapped_values)};
        }

        template <typename M, typename ... Ts>
        // requires Callable<M, T const&, Ts const& ...>
        friend auto Map (M mapper,
                         CPUTensor const& that,
                         CPUTensor<Ts> const& ... those)
        {
            assert((that.lengths == those.lengths and ...))
            using Mapped = std::result_of_t<M(T const&, Ts const& ...)>;
            std::vector<Mapped> mapped_values;
            mapped_values.reserve(that.values.size());
            auto iters = std::make_tuple(that.values.begin(),
                                         those.values.begin() ...);
            while (std::get<0>(iters) != that.values.end())
            {
                mapped_values.emplace_back(std::apply(
                    [&](auto ... iters)
                    {
                        return mapper(*iters++ ...);
                    },
                    iters));
            }
            return CPUTensor<Mapped>{lengths, std::move(mapped_values)};
        }
#endif

        /// Scans along the last \a dimensions
        ///
        /// @details
        /// A tensor wil be returned with same dimensional lengths like this
        /// tensor except for the last \a dimensions. With \a keep_dimensions
        /// being true, these last \a dimensions will be mapped to dimensions of
        /// length one. With \a keep_dimensions being false, these last \a
        /// dimensions will be removed. For each configuration of the remaining
        /// dimensions, the components will be mapped onto a single value which
        /// ...
        ///
        template <typename D, typename S>
        auto Scan (D dimensions, bool keep_dimensions, S scanner)
        const
        {
            if constexpr (std::is_same_v<D, PrefixDimensions>)
            {
                assert(dimensions.count >= -lengths.size());
                assert(dimensions.count <= lengths.size());
            }
            else if constexpr (std::is_same_v<D, SuffixDimensions>)
            {
                assert(dimensions.count >= -lengths.size());
                assert(dimensions.count <= lengths.size());
            }
            else if constexpr (std::is_same_v<D, std::vector<bool>>)
            {
                assert(dimensions.size() == lengths.size());
            }

            auto new_lengths = NewLengths(dimensions, keep_dimensions, 0);
            auto new_values = std::vector<T>{};
            new_values.reserve(Product(new_lengths));

            // TODO assumes linear traversal!!!
            FoldAdjacents(
                dimensions,
                [&](auto const& indices, auto begin, auto const end)
                {
                    assert(std::distance(begin, end) > 0);
                    auto init = *begin++;
                    return std::accumulate(
                        begin,
                        end,
                        std::move(init),
                        scanner);
                },
                [&](auto init, auto const begin, auto const end)
                {
                    assert(std::distance(begin, end) > 0);
                    return std::accumulate(
                        begin,
                        end,
                        std::move(init),
                        scanner);
                },
                [&](auto init)
                {
                    new_values.push_back(std::move(init));
                });

            return CPUTensor{std::move(new_lengths), std::move(new_values)};
        }

        /// Reorders a copy of components along the last \a dimensions into a
        /// single dimension
        ///
        /// @detail
        /// A tensor will be returned. The returning tensor will have identical
        /// dimensional lengths except for the last \a dimensions which will be
        /// replaced by one dimension with a length equivalent to the length
        /// product of those last \a dimensions. The components will be
        /// reordered separatelly for each configuration in the head dimensions.
        ///
        /// @param dimensions TODO description
        ///
        /// @param order TODO description
        ///
        /// @pre dimensions >= tensor.Lengths().size()
        ///
        /// @pre dimensions <= tensor.Lengths().size()
        template <typename D, typename O>
        // requires Callable<O, std::vector<int> const&, std::vector<T>::iterator, std::vector<T>::iterator>
        auto Reorder (D dimensions, bool keep_dimensions, O order) const
        {
            if constexpr (std::is_same_v<D, PrefixDimensions>)
            {
                assert(dimensions.count >= -lengths.size());
                assert(dimensions.count <= lengths.size());
            }
            else if constexpr (std::is_same_v<D, SuffixDimensions>)
            {
                assert(dimensions.count >= -lengths.size());
                assert(dimensions.count <= lengths.size());
            }
            else if constexpr (std::is_same_v<D, std::vector<bool>>)
            {
                assert(dimensions.size() == lengths.size());
            }
            else
            {
                assert(false);
            }

            auto new_lengths = NewLengths(dimensions, keep_dimensions, -1);
            assert(Product(new_lengths) == values.size());
            auto new_values = std::vector<T>{};
            new_values.reserve(values.size());

            // TODO assumes linear traversal; must be generalised
            FoldAdjacents(
                dimensions,
                [&](auto const&, auto const begin, auto const end)
                {
                    new_values.insert(new_values.end(), begin, end);
                    return nullptr;
                },
                [&](auto const, auto const begin, auto const end)
                {
                    new_values.insert(new_values.end(), begin, end);
                    return nullptr;
                },
                [&](auto const)
                {
                    auto const begin = new_values.end() - new_lengths.back();
                    auto const end = new_values.end();
                    order(begin, end);
                });

            return CPUTensor{std::move(new_lengths), std::move(new_values)};
        }


#if 0
        /// Folds along the last \a dimensions
        ///
        /// @pre dimensions <= Lengths().size()
        /// @pre dimensions >= Lengths().size()
        ///
        template <typename I, typename F, typename G>
        auto Fold (int dimensions, I initialiser, F folder, G finisher) const
        {
            assert(dimensions <= lengths.size());
            assert(dimensions >= -lengths.size());

            if (dimensions < 0)
            {
                dimensions += lengths.size();
            }


        }
#endif

        /// Reduces \a dimensions to one value
        ///
        /// All values along the \a dimensions will be reduced to one value for
        /// each configuration of the remaining dimensions.
        ///
        ///
        template <typename D, typename R>
        requires std::is_same_v<D, SuffixDimensions> or std::is_same_v<D, PrefixDimensions> or std::is_same_v<D, std::vector<bool>>
        auto Reduce (D dimensions, bool keep_dimensions, R reducer) const
        {
            if constexpr (std::is_same_v<D, PrefixDimensions>)
            {
                assert(dimensions.count >= -lengths.size());
                assert(dimensions.count <= lengths.size());               
            }
            else if constexpr (std::is_same_v<D, SuffixDimensions>)
            {
                assert(dimensions.count >= -lengths.size());
                assert(dimensions.count <= lengths.size());
            }
            else if constexpr (std::is_same_v<D, std::vector<bool>>)
            {
                assert(dimensions.size() == lengths.size());
            }
            else
            {
                assert(false);
            }

            auto new_lengths = NewLengths(dimensions, keep_dimensions, 0);

            std::vector<T> new_values;
            new_values.reserve(Product(new_lengths));

            FoldAdjacents(
                dimensions,
                [&](auto const& indices, auto const begin, auto const end)
                {
                    auto chunk = std::vector<T>{};
                    chunk.reserve(values.size() / new_values.size());
                    chunk.insert(chunk.end(), begin, end);
                    return std::move(chunk);
                },
                [](auto chunk, auto const begin, auto const end)
                {
                    chunk.insert(chunk.end(), begin, end);
                    return std::move(chunk);
                },
                [&](auto chunk)
                {
                    new_values.push_back(reducer(chunk.begin(), chunk.end()));
                });

            return CPUTensor{std::move(new_lengths), std::move(new_values)};
        }

    private:

        template <typename C>
        static auto Product (C const& container)
        {
            return std::accumulate(
                std::begin(container),
                std::end(container),
                1,
                std::multiplies<>{});
        }

        std::vector<int> NewLengths (SuffixDimensions dimensions, bool keep_dimensions, int postfix)
        {
            assert(dimensions.count >= -lengths.size());
            assert(dimensions.count <= lengths.size());
            assert(postfix >= 0);

            if (dimensions.count < 0)
            {
                dimensions.count += lengths.size();
            } 
            auto const new_rank = keep_dimensions
                                   ? lengths.size()
                                   : dimensions.count + (postfix == 0 ? 0 : 1);
            std::vector<int> new_lengths;
            new_lengths.reserve(new_rank);
            new_lengths.insert(
                new_lengths.end(),
                lengths.begin(),
                lengths.begin() + dimensions.count);
            if (keep_dimensions)
            {
                new_lengths.resize(lengths.size(), 1);
            }
            if (postfix != 0)
            {
                new_lengths.push_back(postfix);
            }
            return new_lengths;
        }

        std::vector<int> NewLengths (PrefixDimensions dimensions,
                                     bool keep_dimensions,
                                     int postfix)
        {
            assert(dimensions.count >= - lengths.size());
            assert(dimensions.count <= lengths.size());
            assert(postfix >= 0);

            if (dimensions.count < 0)
            {
                dimensions.count += lengths.size();
            } 
            auto const new_rank = keep_dimensions
                ? lengths.size()
                : (lengths.size() - dimensions.count + (postfix == 0 ? 0 : 1));
            std::vector<int> new_lengths;
            new_lengths.reserve(new_rank);
            if (keep_dimensions)
            {
                new_lengths.resize(dimensions.count, 1);
            }
            new_lengths.insert(
                new_lengths.end(),
                lengths.begin() + dimensions.count,
                lengths.end());
            if (postfix != 0)
            {
                new_lengths.push_back(postfix);
            }
            return new_lengths;
        }

        std::vector<int> NewLengths (std::vector<bool> dimensions,
                                     bool keep_dimensions,
                                     int postfix)
        {
            assert(dimensions.size() == lengths.size());
            assert(postfix >= 0);
            
            auto const new_rank = keep_dimensions
                ? lengths.size()
                : std::count(dimensions.begin(), dimensions.end(), false);
            std::vector<int> new_lengths;
            new_lengths.reserve(new_rank + (postfix == 0 ? 0 : 1));
            for (int alpha = 0; alpha < lengths.size(); ++alpha)
            {
                if (dimensions[alpha])
                    new_lengths.push_back(lengths[alpha]);
                else if (keep_dimensions)
                    new_lengths.push_back(1);
            }
            if (postfix != 0)
            {
                new_lengths.push_back(postfix);
            }
            return new_lengths;
        }

        /// Folds chunks of adjacent elements for projecting the last \a
        /// dimensions
        ///
        /// For each configuration, all chunks of adjacent elements for
        /// projecting the last \a dimensions will be folded by \a head_folder
        /// (the first fold) and \a tail_folder (the subsequent folds). The
        /// final fold will be passed to \a finisher.
        /// 
        template <typename H, typename U, typename F>
        void FoldAdjacents (SuffixDimensions dimensions,
                            H head_folder,
                            U tail_folder,
                            F finisher) const
        {
            assert(dimensions.count >= -lengths.size());
            assert(dimensions.count <= lengths.size());

            if (dimensions.count < 0)
            {
                dimensions.count += lengths.size();
            }

            auto const chunk_size = std::accumulate(
                lengths.rbegin(),
                lengths.rbegin() + dimensions.count,
                1,
                std::multiplies<int>{});

            auto indices = std::vector<int>{lengths.size(), 0};
            auto begin = values.begin();
            while (begin != values.end())
            {
                auto const end = begin + chunk_size;
                finisher(head_folder(
                    std::as_const(indices),
                    std::as_const(begin),
                    end));
                begin = end;
                auto iter_indices = indices.rbegin() + dimensions.count;
                auto iter_lengths = lengths.rbegin() + dimensions.count;
                while (iter_indices != indices.rend())
                {
                    ++*iter_indices;
                    if (*iter_indices < *iter_lengths)
                    {
                        break;
                    }
                    *iter_indices = 0;
                    ++iter_indices;
                    ++iter_lengths;
                }
                assert(iter_indices != indices.rend() or begin == values.end());
            }
        }

        /// Folds chunks of adjacent elements for projecting the first \a
        /// dimensions
        ///
        /// For each configuration, all chunks of adjacent elements for
        /// projecting the first \a dimensions will be folded by \a head_folder
        /// (the first fold) and \a tail_folder (the subsequent folds). The
        /// final fold will be passed to \a finisher.
        /// 
        template <typename H, typename U, typename F>
        void FoldAdjacents (PrefixDimensions dimensions,
                            H head_folder,
                            U tail_folder,
                            F finisher) const
        {
            assert(dimensions.count >= -lengths.size());
            assert(dimensions.count <= lengths.size());

            if (dimensions.count < 0)
            {
                dimensions.count += lengths.size();
            }

            auto const stride = std::accumulate(
                lengths.rbegin(),
                lengths.rbegin() + dimensions.count,
                1,
                std::multiplies<int>{});
            assert(values.size() % stride == 0);

            auto const counts = values.size() / stride;

            auto indices = std::vector<int>{lengths.size(), 0};
            for (int alpha = 0; alpha < stride; ++alpha)
            {
                auto const head_begin = values.begin() + alpha;
                auto const head_end = head_begin + 1;
                auto folded = head_folder(
                    std::as_const(indices),
                    head_begin,
                    head_end);
                for (int beta = 1; beta < counts; ++beta)
                {
                    auto const begin = values.begin() + alpha + beta * stride;
                    auto const end = begin + 1;
                    folded = tail_folder(std::move(folded), begin, end);
                }
                finisher(std::move(folded));
                auto iter_indices = indices.rbegin();
                auto iter_lengths = lengths.rbegin();
                while (iter_indices != indices.rbegin() + dimensions.count)
                {
                    ++*iter_indices;
                    if (*iter_indices < *iter_lengths)
                    {
                        break;
                    }
                    *iter_indices = 0;
                    ++iter_indices;
                    ++iter_lengths;
                }
            }
        }

        /// 
        template <typename H, typename U, typename F>
        void FoldAdjacents (std::vector<bool> const& dimensions,
                            H head_folder,
                            U tail_folder,
                            F finisher)
        const
        {
            assert(dimensions.size() == lengths.size());

            auto const chunk_count = std::distance(
                dimensions.rbegin(),
                std::find(dimensions.rbegin(), dimensions.rend(), false));

            auto const free_count = dimensions.size() - chunk_count;

            auto const chunk_length = std::accumulate(
                lengths.rbegin(),
                lengths.rbegin() + chunk_count,
                1,
                std::multiplies<int>{});

            auto indices = std::vector<int>{lengths.size(), 0};

            while (true)
            {
                // TODO offset is not well defined
                auto const offset = 0;
                /* Inner loop ... */
                auto folded = head_folder(
                    std::as_const(indices),
                    values.begin() + offset,
                    values.begin() + offset + chunk_length);

                for (auto inner_index = free_count; inner_index > 0;)
                {
                    --inner_index;
                    if (not dimensions[inner_index])
                        continue;
                    ++indices[inner_index];
                    if (indices[inner_index] >= lengths[inner_index])
                    {
                        indices[inner_index] = 0;
                        continue;
                    }

                    // TODO begin and end are not well defined
                    auto const begin = values.begin();
                    auto const end = begin + offset;
                    folding = folder(std::move(folding), begin, end);
                    inner_index = free_dimensions;
                }
                finisher(std::move(folded));

                auto outer_index = free_count;
                while (outer_index > 0)
                {
                    --outer_index;
                    if (dimensions[outer_index])
                        continue;
                    ++indices[outer_index];
                    if (indices[outer_index] >= lengths[outer_index])
                    {
                        indices[outer_index] = 0;
                        continue;
                    }
                }
                if (outer_index == 0)
                    break;
            }
        }

#if 0
        bool Increment (std::vector<int> & indices)
        const
        {
            assert(indices.size() == lengths.size());
            auto iter_length = lengths.rbegin();
            auto iter_indices = indices.rbegin();
            while (iter_indices != indices.rend())
            {
                ++(*iter_indices);
                if (*iter_indices <= *iter_length)
                {
                    break;
                }
                *iter_indices = 0;
                ++iter_indices;
                ++iter_length;
            }

            return iter_indices != indices.rend();
        }

        bool Increment (std::vector<int> & indices,
                        std::vector<bool> const& mask)
        const
        {
            assert(indices.size() == lengths.size());
            assert(mask.size() == lengths.size());

            auto iter_length = lengths.rbegin();
            auto iter_indices = indices.rbegin();
            auto iter_mask = mask.rbegin();

            while (iter_indices != indices.rend())
            {
                if (*iter_mask)
                {
                    ++(*iter_indices);
                    if (*iter_indices <= *iter_length)
                    {
                        break;
                    }
                    *iter_indices = 0;
                }
                ++iter_length;
                ++iter_indices;
                ++iter_mask;
            }

            return iter_indices != indices.rend();
        }
#endif

        /// Computes the offset to the indexed value
        ///
        /// @pre indices.size() == lengths.size()
        /// @pre for i in indices: i >= 0
        /// @pre for (i,l) in (indices, lengths): i < l
        /// 
        int Offset (std::vector<int> const& indices) const
        {
            assert(indices.size() == lengths.size());
            assert(indices.size() == strides.size());
            for (int alpha = 0; alpha < indices.size(); ++alpha)
            {
                assert(indices[alpha] >= 0);
                assert(indices[alpha] < lengths[alpha]);
            }
            auto offset = 0;
            for (int alpha = 0; alpha < indices.size(); ++alpha)
            {
                offset += indices[alpha] * strides[alpha];
            }
            assert(offset < values.size());
            return offset;
        }

        std::vector<T> values;
        std::vector<int> lengths;
        std::vector<int> strides;

    };

#if 0
    // -------------------------------------------------------------------------
    // Matrix Algebra
    // -------------------------------------------------------------------------

    template <typename A>
        requires NumericalMonoid<A>
    auto MatrixMultiply (CPUTensor<A> const& left, CPUTensor<A> const& right)
    {

        if (not left.Lengths().empty() and not right.Lengths().empty())
        {
            assert(left.Lengths().back() == right.Lengths().front());

        }
        else if (left.Lengths().empty() and right.Lengths().empty())
        {
            return CPUTensor<A>{};
        }
        else
        {
            assert(false);
        }
    }

    // -------------------------------------------------------------------------
    // Convolutions
    // -------------------------------------------------------------------------

    template <typename T>
    auto Convolute2D (CPUTensor<T> const& input,
                      CPUTensor<T> const& kernel,
                      int padding,
                      int stride)
    {
        assert(input.Lengths().size() >= 2);
        assert(kernel.Lengths().size() >= 2);
        assert(kernel.Lengths().size() % 2 == 1);
        assert(kernel.Lengths()[kernel.Lengths()-1] ==
               kernel.Lengths()[kernel.Lengths()-2]);
        assert(padding >= 0);
        assert(stride > 0);
        assert((*(input.Lengths().rbegin()) - *(kernel.Lengths().rbegin()) +
               2 * padding) % stride == 0);
        assert((*(input.Lengths().rbegin()+1) - *(kernel.Lengths().rbegin()+1) +
               2 * padding) % stride == 0);

        auto const& input_lengths = input.Lengths();
        auto const& kernel_Lengths = kernel.Lengths();
        auto new_lengths = std::vector<int>{};
        new_lengths.reserve(input_lengths.size() + kernel_lengths().size() - 2);
        new_lengths.insert(new_lengths.end(), input_lengths.begin(), input_lengths.end() - 2);
        new_lengths.insert(new_lengths.end(), kernel_lengths.begin(), kernel_lengths.end() - 2);
        new_lengths.push_back((*(input_lengths.rbegin()+1) - *(kernel_lengths.rbegin()+1) + 2 * padding) / stride + 1);
        new_lengths.push_back((*(input_lengths.rbegin()) - *(kernel_lengths.rbegin()) + 2 * padding) / stride + 1);

        auto new_values = std::vector<T>{};
        new_values.reserve(std::accumulate(
            new_lengths.begin(),
            new_lengths.end(),
            1,
            std::multiplies<int>{});

            
    }

    // -------------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------------

    /// Means over the last dimension of \a tensor
    ///
    /// If the \a tensor is empty, then an empty tensor will be returned. If the
    /// \a tensor is not empty, then the arithmetical means will be computed
    /// over the last dimension and a tensor with the means will be returned.
    ///
    template <NumericalMonoid T, typename D>
    auto Mean (CPUTensor<T> const& tensor, D dimensions, bool is_kept)
    {
        return tensor.Fold(
            std::move(dimensions),
            is_kept,
            [](auto const& indices){ return std::tuple{Zero<T>, Zero<T>}; },
            [](auto folding, T const& component)
            {
                auto const& [sum, count] = folding;
                auto const new_count = Add(count, One<T>);
                return std::tuple
                {
                    Divide(Add(Multiply(sum, count), component), new_count),
                    new_count
                };
            },
            [](auto folding) { return std::get<0>(folding); });
    }


    template <NumericalMonoid T>
    auto Mode (CPUTensor<T> const& tensor)
    {
        return tensor.Fold(
            [](auto const& indices)
            {
                std::unordered_map<T, int> counts;
                return std::move(counts);
            },
            [](auto folding, T cons}t& component)
            {
                ++folding[component];
                return std::move(folding);
            },
            [](auto folding)
            {
                assert(not folding.empty());
                auto mode_iter = folding.begin();
                for (auto iter = folding.begin(); iter != folding.end(); ++iter)
                {
                    auto const& mode_count = mode_iter->second;
                    auto const& current_count = iter->second;
                    if (mode_count < current_count)
                    {
                        mode_iter = iter;
                    }
                }
                return mode_iter->first;
            }
        );
    }

    template <NumericalMonoid A>
    auto Kth (CPUTensor<A> const& tensor, int kth, int dimensions, bool is_kept)
    {
        assert(dimensions <= tensor.Lengths().size());
        assert(dimensions >= tensor.Lengths().size());
        assert(kth < ...);
        assert(kth >= 0);

        return tensor.Reduce(dimensions, is_kept, [&](auto begin, auto end)
        {
            auto iter = begin + kth;
            std::nth_element(begin, iter, end);
            return *iter;
        });
    }

    /// Sorts components over arbitrary \a dimensions
    ///
    /// All componenents along all true \a dimensions will be sorted into a
    /// single dimension. The order is defined by \a comparer.
    ///
    /// @pre dimensions.size() == tensor.Lengths().size()
    ///
    /// @return ...
    ///
    template <typename T, typename D, typename C>
    requires CallableTo<C, bool, T const&, T const&>
    auto Sort (CPUTensor<T> const& tensor, D dimensions, C comparer)
    {
        return tensor.Reorder(std::move(dimensions), [&](auto begin, auto end)
        {
            std::sort(begin, end, comparer);
        });
    }

    /// Partitions components over \a dimensions according to \a prediction
    ///
    template <typename T, typename D, typename P>
    requires CallableTo<P, bool, T const&>
    auto Partition (CPUTensor<T> const& tensor, D dimensions, P predictor)
    {
        return tensor.Reorder(std::move(dimensions), [&](auto begin, auto end)
        {
            std::partition(begin, end, predictor);
        });
    }
#endif

}

#endif
