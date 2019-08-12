#ifndef LIBTORCH__CPU_TENSOR__HEADER
#define LIBTORCH__CPU_TENSOR__HEADER

#include <vector>
#include <cassert>
#include <numeric>
#include <tuple>

namespace torch
{

    template <typename T>
    class CPUTensor
    {
    public:

        /// Constructs an empty tensor
        ///
        CPUTensor () = default;

        /// Constructs a copy of that \a tensor
        ///
        CPUTensor (CPUTensor const& tensor) = default;

        /// Constructs a move of that \a tensor
        ///
        CPUTensor (CPUTensor && tensor) = default;

        template <typename I>
        // requires CallableTo<I, T, std::vector<int> const&>
        //      and MoveConstructible<T>
        CPUTensor (std::vector<int> lengths, I initialiser)
        {
            assert(lengths.size() > 0);
            for (int length : lengths)
            {
                assert(length > 0);
            }

            auto const total_length = std::accumulate(
                std::begin(lengths),
                std::end(lengths) ,
                [](int count, int length){return count + length;});
            values.reserve(total_length);
            
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
            std::shift_right(std::begin(this->strides), std::end(this->strides), 1);
            this->strides[0] = 0;
        }

        /// Dimensional lengths
        ///
        std::vector<int> const& Lengths () const
        {
            return lengths;
        }

        /// Single component on host at \a indices
        ///
        T const& At (std::vector<int> const& indices) const
        {
            assert(indices.size() == lengths.size());
            auto const offset = Offset(indices);
            return values[offset];
        }

        /// Visits each element with the respective indices
        ///
        template <typename V>
        // requires Callable<V, T const&, std::vector<int> const&>
        void Visit (V visitor) const
        {
            std::vector<int> indices{0, lengths.size()};
            for (auto const& value : values)
            {
                visitor(value, indices);
                Increment(indices);
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
        template <typename S>
        // requires CallableTo<S, T, T, T>
        auto Scan (int dimensions, bool keep_dimensions, S scanner) const
        {
            assert(dimensions <= lengths.size());
            assert(dimensions >= -lengths.size());

            if (dimensions < 0)
            {
                dimensions += lengths.size();
            }

            auto new_lengths = std::vector<int>{};
            new_lengths.reserve(keep_dimensions ? lengths.size() : dimensions);
            new_lengths.insert(
                new_lengths.end(),
                lengths.begin(),
                lengths.begin() + dimensions);
            if (keep_dimensions)
            {
                new_lengths.resize(lengths.size(), 1);
            }

            auto const new_total_length = std::accumulate(
                new_lengths.begin(),
                new_lengths.end(),
                1,
                std::multiplies<int>{});

            auto new_values = std::vector<T>{};
            new_values.reserve(new_total_length);

            FoldChunks(
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

        template <typename S>
        // requires CallableTo<S, T, T, T>
        auto Scan (std::vector<bool> dimensions, bool keep_dimensions, S scanner)
        const
        {
            assert(dimensions.size() == lengths.size());

            auto new_lengths = std::vector<int>{};
            new_lengths.reserve(keep_dimensions ? lengths.size() : std::count(
                dimensions.begin(),
                dimensions.end(),
                false));
            for (int alpha = 0; alpha < dimensions.size(); ++alpha)
            {
                if (not dimensions[alpha])
                {
                    new_lengths.push_back(lengths[alpha]);
                }
                else if (keep_dimensions)
                {
                    new_lengths.push_back(1);
                }
            }

            auto const new_total_length = std::accumulate(
                new_lengths.begin(),
                new_lengths.end(),
                1,
                std::multiplies<int>{});

            auto new_values = std::vector<T>{};
            new_values.reserve(new_total_length);

            FoldChunks(
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
                    return std::accumulate(
                        begin,
                        end,
                        std::move(init),
                        scanner);
                },
                [&](auto scanned)
                {
                    new_values.push_back(std::move(scanned));
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
        template <typename O>
        // requires Callable<O, std::vector<int> const&, std::vector<T>::iterator, std::vector<T>::iterator>
        auto Reorder (int dimensions, O order) const
        {
            assert(dimensions >= lengths.size());
            assert(dimensions <= lengths.size());

            if (dimensions < 0)
            {
                dimensions += lengths.size();
            }

            auto new_lengths = std::vector<int>{};
            new_lengths.reserve(dimensions + 1);
            new_lengths.insert(
                new_lengths.end(),
                lengths.begin(),
                lengths.begin() + dimensions);
            new_lengths.push_back(std::accumulate(
                lengths.begin() + dimensions,
                lengths.end(),
                1,
                std::multiplies<int>{}));

            auto new_values = std::vector<T>{};
            new_values.reserve(values.size());

            FoldChunks(
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

        /// Reorders a copy of components along all true \a dimensions into a
        /// single dimension
        ///
        /// @detail
        /// A tensor will be returned. The returning tensor will have identical
        /// dimensional lengths except for all true \a dimensions which will be
        /// replaced by one dimension with a length equivalent to the length of
        /// those true \a dimensions. The components will be rerordered
        /// separatelly for each configuration in the false \a dimensions.
        ///
        /// @param dimensions
        /// Boolean mapping for each dimension to be included in the reordering
        /// (true value) or excluded (false value).
        ///
        /// @param order
        /// TODO description
        ///
        /// @pre dimensions.size() == Lengths().size()
        ///
        template <typename O>
        auto Reorder (std::vector<bool> dimensions, O order) const
        {
            assert(dimensions.size() == lengths.size());

            auto new_lengths = std::vector<int>{};
            new_lengths.reserve(1 + std::count(
                dimensions.begin(),
                dimensions.end(),
                false));
            for (int alpha = 0; alpha < dimensions.size(); ++alpha)
            {
                if (not dimensions[alpha])
                {
                    new_lengths.push_back(lengths[alpha]);
                }
            }
            new_lengths.push_back(values.size() / std::accumulate(
                new_lengths.begin(),
                new_lengths.end(),
                1,
                std::multiplies<int>{}));

            assert(values.size() == std::accumulate(
                new_values.begin(),
                new_values.end(),
                1,
                std::multiplies<int>{}));

            auto new_values = std::vector<T>{};
            new_values.reserve(values.size());

            FoldChunks(
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

        /// Reduces the last \a dimensions to one value
        ///
        /// All values along the true \a dimensions will be reduced to one value
        /// for each configuration of the remaining dimensions.
        ///
        /// @pre dimensions <= Lengths().size()
        /// @pre dimensions >= -Lengths().size()
        ///
        ///
        template <typename R>
        auto Reduce (int dimensions, bool keep_dimensions, R reducer) const
        {
            assert(dimensions >= -lengths.size());
            assert(dimensions <= lengths.size());

            if (dimensions < 0)
            {
                dimensions += lengths.size();
            }

            std::vector<int> new_lengths;
            new_lengths.reserve(keep_dimensions ? lengths.size() : dimensions);
            new_lengths.insert(
                new_lengths.end(),
                lengths.begin(),
                lengths.begin() + dimensions);
            if (keep_dimensions)
            {
                new_lengths.resize(lengths.size(), 1);
            }

            auto const new_total_length = std::accumulate(
                new_lengths.begin(),
                new_lengths.end(),
                1,
                std::multiplies<int>{});

            std::vector<T> new_values;
            new_values.reserve(new_total_length);

            FoldChunks(
                dimensions,
                [](auto const& indices, auto const begin, auto const end)
                {
                    auto chunk = std::vector<T>{};
                    chunk.reserve(values.size() / new_total_length);
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

        /// Reduces to one component over arbitrary \a dimensions
        ///
        /// All values along the true \a dimensions will be reduced to one value
        /// for each configuration of false \a dimensions.
        ///
        /// @pre dimensions.size() == Lengths().size()
        ///
        template <typename R>
        // requires CallableTo<R, T, std::vector<int> const&, std::vector<T>::iterator, std::vector<T>::iterator>
        auto Reduce (std::vector<bool> const& dimensions,
                     bool const keep_dimensions,
                     R reducer)
        const
        {
            assert(dimensions.size() == lengths.size());

            std::vector<int> new_lengths;
            new_lengths.reserve(keep_dimensions
                ? lengths.size()
                : std::count(dimensions.begin(), dimensions.end(), false));

            for (int alpha = 0; alpha < dimensions.size(); ++alpha)
            {
                if (not dimensions[alpha])
                    new_lengths.push_back(lengths[alpha]);
                else if (keep_dimensions)
                    new_lengths.push_back(1);
            }

            auto const new_total_length = std::accumulate(
                new_lengths.begin(),
                new_lengths,
                1,
                std::multiplies<int>{});

            std::vector<T> new_values;
            new_values.reserve(new_total_length);

            FoldChunks(
                dimensions,
                [](auto const& indices, auto const begin, auto const end)
                {
                    auto chunk = std::vector<T>{};
                    chunk.reserve(values.size() / new_total_length);
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

        template <typename C, typename F, typename D>
        void FoldChunks (std::vector<bool> const& dimensions, C constructor, F folder, D destructor)
        const
        {
            assert(dimensions.size() == lengths.size());

            auto const chunk_dimensions = std::distance(
                dimensions.rbegin(),
                std::find(dimensions.rbegin(), dimensions.rend(), false));

            auto const free_dimensions = dimensions.size() - chunk_dimensions;

            auto const chunk_length = std::accumulate(
                lengths.rbegin(),
                lengths.rbegin() + chunk_dimensions,
                1,
                std::multiplies<int>{});

            auto indices = std::vector<int>{lengths.size(), 0};

            while (true)
            {
                /* Inner loop ... */
                auto folding = constructor(
                    std::as_const(indices),
                    values.begin() + offset,
                    values.begin() + offset + chunk_length);

                for (auto inner_index = free_dimensions; inner_index > 0;)
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

                    auto const begin = /*  ... */;
                    auto const end = /* ... */;
                    folding = folder(std::move(folding), begin, end);
                    inner_index = free_dimensions;
                }
                destructor(std::move(folding));

                auto outer_index = free_dimensions;
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

        template <typename C, typename F, typename D>
        void FoldChunks (int dimensions, C constructor, F folder, D destructor)
        const
        {
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
