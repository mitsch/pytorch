#include <torch/CPUTensor.hpp>
#include <catch2/catch.hpp>


TEST_CASE("Construction of a CPUTensor")
{
    SECTION("A default construction results in an empty tensor.")
    {
        auto const tensor = torch::CPUTensor<int>{};

        REQUIRE(tensor.Lengths().empty());
        tensor.Visit([](int, std::vector<int> const&)
        {
            REQUIRE(false);
        });

        SECTION("A copy construction returns a tensor with the same "\
                "properties.")
        {
            torch::CPUTensor<int> copied{tensor};

            REQUIRE(copied.Lengths().empty());
            copied.Visit([](int, std::vector<int> const&)
            {
                REQUIRE(false);
            });
        }

        SECTION("A copy assignment returns a tensor with the same "\
                "properties.")
        {
            auto const copied = tensor;

            REQUIRE(copied.Lengths().empty());
            copied.Visit([](int, std::vector<int> const&)
            {
                REQUIRE(false);
            });
        }
    }

    SECTION("An inplace construction with no lengths results in an empty "\
            "tensor.")
    {
        auto const tensor = torch::CPUTensor<int>{
            std::vector<int>{},
            [](std::vector<int> const&){ return 0; }};

        REQUIRE(tensor.Lengths().empty());
        tensor.Visit([](int, std::vector<int> const&)
        {
            REQUIRE(false);
        });

        SECTION("A copy construction returns a tensor with the same "\
                "properties.")
        {
            torch::CPUTensor<int> copied{tensor};

            REQUIRE(copied.Lengths().empty());
            copied.Visit([](int, std::vector<int> const&)
            {
                REQUIRE(false);
            });
        }

        SECTION("A copy assignment returns a tenosr with the same "\
                "properties.")
        {
            auto const copied = tensor;

            REQUIRE(copied.Lengths().empty());
            copied.Visit([](int, std::vector<int> const&)
            {
                REQUIRE(false);
            });
        }
    }

    SECTION("An inplace construction with some lengths results in an "\
            "non-empty tensor of same lengths.")
    {
        auto const tensor = torch::CPUTensor<int>{
            std::vector<int>{3, 4},
            [](std::vector<int> const& indices)
            {
                return indices[0] + indices[1] * 3;
            }};

        auto const lengths = tensor.Lengths();
        REQUIRE(lengths.size() == 2);
        REQUIRE(lengths[0] == 3);
        REQUIRE(lengths[1] == 4);

        std::set<int> visited;
        tensor.Visit([&](int value, std::vector<int> const& indices)
        {
            REQUIRE(indices.size() == 2);
            REQUIRE(indices[0] >= 0);
            REQUIRE(indices[0] < 3);
            REQUIRE(indices[1] >= 0);
            REQUIRE(indices[1] < 4);
            REQUIRE(value == indices[0] + 3 * indices[1]);
            visited.insert(value);
        });
        REQUIRE(visited.size() == 12);
        for (int value = 0; value < 12; ++value)
        {
            REQUIRE(visited.count(value) == 1);
        }

        SECTION("A copy construction returns a tensor with the same "\
                "properties.")
        {
            torch::CPUTensor<int> copied{tensor};

            auto const lengths = copied.Lengths();
            REQUIRE(lengths.size() == 2);
            REQUIRE(lengths[0] == 3);
            REQUIRE(lengths[1] == 4);

            std::set<int> visited;
            copied.Visit([&](int value, std::vector<int> const& indices)
            {
                REQUIRE(indices.size() == 2);
                REQUIRE(indices[0] >= 0);
                REQUIRE(indices[0] < 3);
                REQUIRE(indices[1] >= 0);
                REQUIRE(indices[1] < 4);
                REQUIRE(value == indices[0] + 3 * indices[1]);
                visited.insert(value);
            });
            REQUIRE(visited.size() == 12);
            for (int value = 0; value < 12; ++value)
            {
                REQUIRE(visited.count(value) == 1);
            }
        }

        SECTION("A copy assignment returns a tensor with the same "\
                "properties.")
        {
            auto const copied = tensor;

            auto const lengths = copied.Lengths();
            REQUIRE(lengths.size() == 2);
            REQUIRE(lengths[0] == 3);
            REQUIRE(lengths[1] == 4);

            std::set<int> visited;
            copied.Visit([&](int value, std::vector<int> const& indices)
            {
                REQUIRE(indices.size() == 2);
                REQUIRE(indices[0] >= 0);
                REQUIRE(indices[0] < 3);
                REQUIRE(indices[1] >= 0);
                REQUIRE(indices[1] < 4);
                REQUIRE(value == indices[0] + 3 * indices[1]);
                REQUIRE(visited.insert(value));
            });
            REQUIRE(visited.size() == 12);
            for (int value = 0; value < 12; ++value)
            {
                REQUIRE(visited.count(value) == 1);
            }
        }
    }
}

TEST_CASE("Mapping of components")
{
    SECTION("An empty tensor will result in an empty tensor.")
    {
        auto const tensor = torch::Tensor<int>{};
        auto const mapped = tensor.Map([](int){ return true; });

        REQUIRE(std::is_same_v<decltype(mapped), torch::CPUTensor<bool> const>);
        REQUIRE(mapped.Lengths().empty());
    }

    SECTION("A non-empty tensor will result in a tensor of the same shape")
    {
        auto const tensor = torch::Tensor<int>{
            {3, 3, 3},
            [](auto const& indices)
            {
                return indices[0] + 3 * indices[1] + 9 * indices[2];
            }};
        auto const mapped = tensor.Map([](int value){ return value * value; });

        REQUIRE(std::is_same_v<decltype(mapped), torch::CPUTensor<int> const>);
        REQUIRE(mapped.Length() == std::vector{3, 3});
        mapped.Visit([](int value, std::vector<int> const& indices)
        {
            auto const origin = indices[0] + 3 * indices[1] + 9 * indices[2];
            auto const expected = origin * origin;
            REQUIRE(value == expected);
        });
    }
}

TEST_CASE("Scanning of non-empty tensor")
{
    auto const tensor = torch::Tensor<int>{
        {2, 3, 2},
        [](auto const& indices)
        {
            return indices[0] + indices[1] * 2 + indices[2] * 6;
        }};

    SECTION("Scanning zero of the suffix dimensions returns a copy.")
    {
        auto const scanned = tensor.Scan(SuffixDimensions(0), std::min<int>);

        REQUIRE(scanned.Lenghts() == std::vector{{2, 3, 2}});
        scanned.Visit([](int value, auto const& indices)
        {
            REQUIRE(indices.size() == 3);
            REQUIRE(indices[0] >= 0);
            REQUIRE(indices[0] < 2);
            REQUIRE(indices[1] >= 0);
            REQUIRE(indices[1] < 3);
            REQUIRE(indices[2] >= 0);
            REQUIRE(indices[2] < 2);
            auto const expected = indices[0] + indices[1] * 2 + indices[2] * 6;
            REQUIRE(value == expected);
        })
    }

    SECTION("Scanning last dimension for sum returns the value "\
            "tensor(i, j, 1).")
    {
        auto const scanned = tensor.Scan(SuffixDimensions(1), std::plus<int>{});

        REQUIRE(scanned.Lengths() == std::vector{{2, 3}});
        scanned.Visit([](int value, auto const& indices)
        {
            REQUIRE(indices.size() == 2);
            REQUIRE(indices[0] >= 0);
            REQUIRE(indices[0] < 2);
            REQUIRE(indices[1] >= 0),
            REQUIRE(indices[1] < 3);
            auto const expected = indices[0] + indices[1] * 2 + 6;
            REQUIRE(value == expected);
        });
    }

    SECTION("Scanning the last two dimensions for sum returns the value "\
            "tensor(i, 0, 0) + 12.")
    {
        auto const scanned = tensor.Scan(SuffixDimensions(2), std::plus<int>{});

        REQUIRE(scanned.Lengths() == std::vector{{2}});
        scanned.Visit([](int value, auto const& indices)
        {
            REQUIRE(indices.size() == 1);
            REQUIRE(indices[0] >= 0);
            REQUIRE(indices[0] < 2);
            auto const expected = indices[0] + 12;
            REQUIRE(expected == value);
        });
    }

    SECTION("Scanning the last three dimensions for sum returns 13.")
    {
        auto const scanned = tensor.Scan(SuffixDimensions(3), std::plus<int>{});

        REQUIRE(scanned.Lengths() == std::vector{{1}});
        scanned.Visit([](int value, auto const& indices)
        {
            REQUIRE(indices.size() == 1);
            REQUIRE(indices[0] == 0);
            REQUIRE(value == 13);
        });
    }

    SECTION("Scanning all but the first three dimensions for sum returns a "\
            "copy.")
    {
        auto const scanned = tensor.Scan(
            SuffixDimensions(-3),
            std::plus<int>{});

        REQUIRE(scanned.Lengths() == std::vector{{2, 3, 2}});
        scanned.Visit([](int value, auto const& indices)
        {
            REQUIRE(indices.size() == 3);
            REQUIRE(indices[0] >= 0);
            REQUIRE(indices[0] < 2);
            REQUIRE(indices[1] >= 0);
            REQUIRE(indices[1] < 3);
            REQUIRE(indices[2] >= 0);
            REQUIRE(indices[2] < 2);
            auto const expected = indices[0] + indices[1] * 2 + indices[2] * 6;
            REQUIRE(expected == value);
        });
    }

    SECTION("Scanning all but the first two dimensions for sum returns the "\
            "values tensor(i, j, 0)+tensor(i, j, 1).")
    {
        auto const scanned = tensor.Scan(
            SuffixDimensions(-2),
            std::plus<int>{});

        REQUIRE(scanned.Lengths() == std::vector{2, 3});
        scanned.Visit([](int value, auto const& indices)
        {
            REQUIRE(indices.size() == 2);
            REQUIRE(indices[0] >= 0);
            REQUIRE(indices[0] < 2);
            REQUIRE(indices[1] >= 0);
            REQUIRE(indices[1] < 3);
            REQUIRE(indices[2] >= 0);
            REQUIRE(indices[2] < 2);
            auto const expected = indices[0] * 2 + indices[1] * 4 + 6;
            REQUIRE(expected == value);
        });
    }

    SECTION("Scanning all but the first dimension for sum returns the values "\
            "6 * tensor(i, 0, 0) + 30.")
    {
        auto const scanned = tensor.Scan(
            SuffixDimensions(-1),
            std::plus<int>{});

        REQUIRE(scanned.Lengths() == std::vector{{2}});
        scanned.Visit([](int value, auto const& indices)
        {
            REQUIRE(indices.size() == 1);
            REQUIRE(indices[0] >= 0);
            REQUIRE(indices[1] < 2);
            auto const expected = indices[0] * 6 + 30;
            REQUIRE(value == expected);
        });
    }

}
