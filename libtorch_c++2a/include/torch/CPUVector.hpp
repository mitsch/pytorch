#ifndef LIBTORCH__CPU_VECTOR__HEADER
#define LIBTORCH__CPU_VECTOR__HEADER

#include <vector>

namespace torch
{

    template <typename V>
    class CPUVector
    {
    public:

        CPUVector () = default;

        template <typename I>
            requires CallableTo<I, V, std::vector<int> const&>
                 and MoveConstructible<V>
        CPUTensor (std::vector<int> lengths, I initialiser)
        {
            assert(lengths.size() > 0);
            for (int length : lengths)
            {
                assert(length > 0);
            }

            auto const total_length = std::accumulate(
                std::begin(lengths),
                std::end(lengths),
                [](int count, int length){return count + length;});

            values.reserve(total_length);
            auto indices = std::vector<int>{0, lengths.size()};

            auto is_overflown = false;
            while (not is_overflown)
            {
                values.emplace_back(initialiser(indices));
                std::tie(is_overflown, indices) = Increment(std::move(indices));
            }

            this->lengths = std::move(lengths);
            this->strides.resize(this->lengths.size());
            std::partial_sum(std::begin(this->lengths), std::end(this->lengths), std::begin(this->strides));
            std::shift_right(std::begin(this->strides), std::end(this->strides), 1);
            this->strides[0] = 0;
        }

        int Rank () const
        {
            return lengths.size();
        }

        int Length (int index) const
        {
            assert(index < lengths.size());
            assert(index >= -lengths.size());
            return lengths[index >= 0 ? index : (lengths.size()-index)];
        }

        V const& At (std::vector<int> const& indices) const
        {
            assert(indices.size() == lengths.size());
            assert(lengths.size() == strides.size());
        }

        template <typename M, typename ... Ts>
            requires Callable<M, V const&>
        auto Map (M mapper, CPUTensor<Ts> const& ... others) const
        {
            using Mapped = std::result_of_t<M(V const&, Ts const& ...)>;
            std::vector<Mapped> mapped_values;
            mapped_values.reserve(values.size());
            for (auto const& value : values)
            {
                mapped_values.emplace_back(mapper(value));
            }
            return CPUTensor<Mapped>{
                std::move(mapped_values),
                lengths,
                strides};
        }


        template <typename P>
        auto Project (std::vector<int> dimensions, P projector) const
        {
            std::vector<
        }


    private:

        std::tuple<bool, std::vector<int>> Increment (std::vector<int> indices)
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
            }

            return {iter_indices == indices.rend(), std::move(indices)};
        }

        std::vector<V> values;
        std::vector<int> lengths;
        std::vector<int> strides;

    };






}

#endif
