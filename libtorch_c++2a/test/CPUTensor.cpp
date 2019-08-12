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
        tensor.Visit([](int value, std::vector<int> const& indices)
        {
            REQUIRE(indices.size() == 2);
            REQUIRE(indices[0] >= 0);
            REQUIRE(indices[0] < 3);
            REQUIRE(indices[1] >= 0);
            REQUIRE(indices[1] < 4);
            REQUIRE(value == indices[0] + 3 * indices[1]);
        });
    }
}

TEST_CASE("Visiting all components")
{
    SECTION("An empty tensor won't touch the visitor")
    {
        auto const tensor = torch::Tensor<int>{};
        tensor.Visit([&](int, std::vector<int> const&){ REQUIRE(false); });
    }

    SECTION("A non-empty tensor will visit each component at least once")
    {
        auto const tensor = torch:::Tensor<int>{{3, 3}, [](auto const& indices)
        {
            return indices[0] + 3 * indices[1] + 9 * indices[2];
        }};
        std::vector<int> values;
        tensor.Visit([&](int actual, std::vector<int> const& indices)
        {
            REQUIRE(indices.size() == 3);
            REQUIRE(indices[0] >= 0);
            REQUIRE(indices[0] < 3);
            REQUIRE(indices[1] >= 0);
            REQUIRE(indices[1] < 3);
            REQUIRE(indices[2] >= 0);
            REQUIRE(indices[2] < 3);
            auto const expected = indices[0] + 3 * indices[1] + 9 * indices[2];
            REQUIRE(actual == expected);
            values.push_back(actual);
        });
        REQUIRE(std::count(values.begin(), values.end(), 0) > 0);
        REQUIRE(std::count(values.begin(), values.end(), 1) > 0);
        REQUIRE(std::count(values.begin(), values.end(), 2) > 0);
        REQUIRE(std::count(values.begin(), values.end(), 3) > 0);
        REQUIRE(std::count(values.begin(), values.end(), 4) > 0);
        REQUIRE(std::count(values.begin(), values.end(), 5) > 0);
        REQUIRE(std::count(values.begin(), values.end(), 6) > 0);
        REQUIRE(std::count(values.begin(), values.end(), 7) > 0);
        REQUIRE(std::count(values.begin(), values.end(), 8) > 0);
    }
}


TEST_CASE("Mapping of components")
{
    SECTION("An empty tensor will result in an empty tensor.")
    {
        auto const tensor = torch::Tensor<int>{};
        auto const mapped = tensor.Map([](int){ return true; });
        REQUIRE(mapped.Lengths() == std::vector<int>{});
    }

    SECTION("A non-empty tensor will result in a tensor of the same shape")
    {
        auto const tensor = torch::Tensor<int>{{3, 3}, [](auto const& indices)
        {
            return indices[0] + 3 * indices[1] + 9 * indices[2];
        }};
        auto const mapped = tensor.Map([](int value){ return value * value; });
        REQUIRE(mapped.Length() == std::vector{3, 3}; );

    }
}
