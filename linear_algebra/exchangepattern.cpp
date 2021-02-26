#include "exchangepattern.h"

ExchangePattern::ExchangePattern() :
    _neighbor_processes(),
    _receive_indices(),
    _send_indices() {}

ExchangePattern::ExchangePattern(std::vector<int> neighboring_processes, std::vector<std::vector<int>> receive_indices,
                                 std::vector<std::vector<int>> send_indices)
{
    _neighbor_processes = std::move(neighboring_processes);
    _receive_indices = std::move(receive_indices);
    _send_indices = std::move(send_indices);
}

ExchangePattern::ExchangePattern(ExchangePattern&& other) noexcept
{
    _neighbor_processes = std::move(other._neighbor_processes);
    _receive_indices = std::move(other._receive_indices);
    _send_indices = std::move(other._send_indices);
}

ExchangePattern::ExchangePattern(const ExchangePattern& other)
{
    _receive_indices.resize(other._receive_indices.size());
    _send_indices.resize((other._send_indices).size());
    for(int i = 0; i < (other._neighbor_processes).size(); i++)
    {
        _neighbor_processes.push_back(other._neighbor_processes[i]);
    }
    for(int i = 0; i < (other._receive_indices).size(); i++)
    {
        for(int j = 0; j < (other._receive_indices[i]).size(); j++)
        {
            _receive_indices[i].push_back(other._receive_indices[i][j]);
        }
    }
    for(int i = 0; i < (other._send_indices).size(); i++)
    {
        for(int j = 0; j < (other._send_indices[i]).size(); j++)
        {
            _send_indices[i].push_back(other._send_indices[i][j]);
        }
    }
}

ExchangePattern& ExchangePattern::operator=(const ExchangePattern& other)
{
    _receive_indices.resize(other._receive_indices.size());
    _send_indices.resize((other._send_indices).size());
    for(int i = 0; i < (other._neighbor_processes).size(); i++)
    {
        _neighbor_processes.push_back(other._neighbor_processes[i]);
    }
    for(int i = 0; i < (other._receive_indices).size(); i++)
    {
        std::vector<int> receive_rows;
        for(int j = 0; j < (other._receive_indices[i]).size(); j++)
        {
            receive_rows.push_back(other._receive_indices[i][j]);
        }
        _receive_indices[i] = receive_rows;
    }
    for(int i = 0; i < (other._receive_indices).size(); i++)
    {
        std::vector<int> send_rows;
        for(int j = 0; j < (other._send_indices[i]).size(); j++)
        {
            send_rows.push_back(other._send_indices[i][j]);
        }
        _send_indices[i] = send_rows;
    }
    return *this;
}

ExchangePattern& ExchangePattern::operator=(ExchangePattern&& other) noexcept
{
    _neighbor_processes = std::move(other._neighbor_processes);
    _receive_indices = std::move(other._receive_indices);
    _send_indices = std::move(other._send_indices);
    return *this;
}

const std::vector<int>& ExchangePattern::neighboring_processes() const
{
    return _neighbor_processes;
}

const std::vector<std::vector<int>>& ExchangePattern::receive_indices() const
{
    return _receive_indices;
}

const std::vector<std::vector<int>>& ExchangePattern::send_indices() const
{
    return _send_indices;
}

bool entry_in_vector(std::vector<int>& vector, int value)
{
    int i = 0;
    while(i < vector.size())
    {
        if(vector[i] == value)
        {
            return true;
        }
        i += 1;
    }
    return false;
}
