// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <iostream>
#include <fstream>
#include <regex>

#include <src/utils/config-parser.h>
#include <src/utils/str_utils.h>

using namespace std;

bool CConfigParser::load(string &error, const string &sFileName)
{
    ifstream file(sFileName);
    if (!file.is_open())
    {
        error = strprintf("Failed to open config file: %s", sFileName);
        return false;
    }

    string sLine, sKey, sValue;
    while (getline(file, sLine))
    {
        // Remove leading and trailing whitespaces
        trim(sLine);

        // Skip empty lines or comments
       if (sLine.empty() || sLine[0] == '#' || sLine[0] == ';')
            continue;

        // Find the delimiter (e.g. '=')
        const auto nDelimiterPos = sLine.find('=');
        if (nDelimiterPos == string::npos)
        {
			error = strprintf("Invalid config line: %s", sLine);
            continue;
        }

        // Extract key and value
        sKey = sLine.substr(0, nDelimiterPos);
        rtrim(sKey);
        sValue = sLine.substr(nDelimiterPos + 1);
        ltrim(sValue);

        // Detect the type of the value and store it
        m_ConfigData[sKey] = detectValueType(sValue);
    }

    file.close();
    return true;
}

    // Detect the type of the value and convert to appropriate type
ConfigValue CConfigParser::detectValueType(const std::string& sValue) const
{
    // Try to detect and convert to int64_t (as the main integer type)
    regex intRegex(R"(^-?\d+$)");
    if (regex_match(sValue, intRegex))
        return stoll(sValue);

    // Try to detect and convert to double
    regex doubleRegex(R"(^[+-]?([0-9]*[.])?[0-9]+$)");
    if (regex_match(sValue, doubleRegex))
        return stod(sValue);

    // Try to detect and convert to bool
	bool bValue;
	if (str_tobool(sValue, bValue))
		return bValue;

    // Fallback to string
    return sValue;
}

optional<ConfigValue> CConfigParser::get(const string& sKey) const
{
    const auto it = m_ConfigData.find(sKey);
    if (it != m_ConfigData.end())
        return it->second;
    return nullopt;
}

