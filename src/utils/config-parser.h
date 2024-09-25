#pragma once
// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <string>
#include <variant>
#include <optional>
#include <unordered_map>
#include <stdexcept>

#include <tinyformat.h>

using ConfigValue = std::variant<std::string, int64_t, double, bool>;

class  CConfigParser
{
public:
	CConfigParser() = default;

	bool load(std::string &error, const std::string &sFileName);
	std::optional<ConfigValue> get(const std::string& sKey) const;

    template<typename T>
    std::optional<T> getAs(const std::string& sKey) const
    {
        if (const auto value = get(sKey); value.has_value())
        {
            try
            {
                return convertValue<T>(value.value());
            } catch (const std::runtime_error& e) {
				throw std::runtime_error(strprintf("Failed to convert value for key '%s': %s", sKey, e.what()));
            }
        }
        return std::nullopt;
    }

    template<typename T>
    T getOrDefault(const std::string& sKey, const T& defaultValue) const
    {
        if (auto value = getAs<T>(sKey); value.has_value())
            return value.value();
        return defaultValue;
    }

protected:
	std::unordered_map<std::string, ConfigValue> m_ConfigData;

	ConfigValue detectValueType(const std::string& valueStr) const;

    // Template method to convert stored value to the requested type
    template<typename T>
    std::optional<T> convertValue(const ConfigValue& value) const
    {
        if constexpr (std::is_same_v<T, std::string>)
        {
            if (std::holds_alternative<std::string>(value)) {
                return std::get<std::string>(value);
            }
        } else if constexpr (std::is_integral_v<T>) {
            if (std::holds_alternative<int64_t>(value)) {
                auto intValue = std::get<int64_t>(value);
                if (intValue >= std::numeric_limits<T>::min() && intValue <= std::numeric_limits<T>::max()) {
                    return static_cast<T>(intValue);
                } else {
                    throw std::runtime_error("Integer out of bounds for type");
                }
            }
        } else if constexpr (std::is_floating_point_v<T>) {
            if (std::holds_alternative<double>(value)) {
                return static_cast<T>(std::get<double>(value));
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            if (std::holds_alternative<bool>(value)) {
                return std::get<bool>(value);
            }
        }

        throw std::runtime_error("Incorrect type for key");
    }
};