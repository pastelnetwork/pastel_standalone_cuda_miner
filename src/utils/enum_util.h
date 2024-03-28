// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once

#include <type_traits>

template<typename _EnumClass>
constexpr auto to_integral_type(const _EnumClass e)
{
	return static_cast<std::underlying_type_t<_EnumClass>>(e);
}

template <typename _EnumClass>
constexpr auto enum_or(const _EnumClass e1, const _EnumClass e2)
{
    return to_integral_type<_EnumClass>(e1) | to_integral_type<_EnumClass>(e2);
}

template <typename _EnumClass>
constexpr bool is_enum_valid(const std::underlying_type_t<_EnumClass> e, const _EnumClass eLowValid, const _EnumClass eHighValid)
{
    return (e >= to_integral_type<_EnumClass>(eLowValid) && (e <= to_integral_type<_EnumClass>(eHighValid)));
}

// typecheck helper
template<typename _Class, typename... Ts>
using AllParamsHaveSameType = typename std::enable_if_t<std::conjunction_v<std::is_same<Ts, _Class>...>>;

// returns true if enumToCheck equals any of the parameters (all params should have same enumeration type)
// function is enabled if all Ts... have the same enumeration type
template <typename _EnumClass, typename... Ts, typename = AllParamsHaveSameType<_EnumClass, Ts...>>
bool is_enum_any_of(const _EnumClass enumToCheck, const _EnumClass enum1, Ts... xs) noexcept
{
    if (enumToCheck == enum1)
        return true;
    if constexpr (sizeof... (xs) > 0)
        return is_enum_any_of(enumToCheck, xs...);
    return false;
}
