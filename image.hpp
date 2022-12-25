#pragma once

#include <string>
#include <stddef.h>
#include <sstream>
#include <fstream>
template <class T>
class image
{
private:
    T *m_data = nullptr;
    size_t m_x = 0;
    size_t m_y = 0;
    void allocate_memory()
    {
        const size_t size = m_x * m_y;
        if (size)
        {
            m_data = new T[m_x * m_y];
        }
    }
    void initial_memory(const T value)
    {
        const size_t size = m_x * m_y;
        for (size_t i = 0; i < size; ++i)
        {
            m_data[i] = value;
        }
    }
    void delete_memory()
    {
        if (m_data)
        {
            delete[] m_data;
            m_data = nullptr;
        }
    }

public:
    image(){};
    image(const std::string);
    image(const size_t width, const size_t height);
    image(const size_t width, const size_t height, const T fill_value);
    void *address() { return reinterpret_cast<void *>(m_data); };
    void reset_image_size(const size_t x, const size_t y, const T fill_value);
    void reset_image_size(const size_t x, const size_t y);
    void read_image_file(std::string filename);
    void save_image_file(std::string filename);
    ~image();
    size_t index(size_t x, size_t y) { return y * m_x + x; };
    T &operator()(size_t x, size_t y) const { return m_data[index(x, y)]; };
    T &operator()(size_t x, size_t y) { return m_data[index(x, y)]; };
    T &operator[](size_t x) { return m_data[x]; };
    T &operator[](size_t x) const { return m_data[x]; };
    size_t x() const
    {
        return m_x;
    };
    size_t y() const
    {
        return m_y;
    };
};