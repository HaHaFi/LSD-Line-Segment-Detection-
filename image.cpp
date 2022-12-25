

#include "image.hpp"

template <class T>
image<T>::image(const std::string file_name)
{
}
template <class T>
void image<T>::read_image_file(std::string filename)
{
    std::ifstream ifs(filename, std::ios::in);
    if (!ifs.is_open())
    {
        // std::cout << "Failed to open file.\n";
        return;
    }

    std::stringstream ss;
    ss << ifs.rdbuf();
    double width, height;
    ss >> width >> height;
    this->reset_image_size(width, height);
    size_t i = 0;
    while (ss)
    {
        ss >> this->m_data[i++];
    }

    ifs.close();
}

template <class T>
void image<T>::save_image_file(std::string filename)
{
    std::ofstream ofs(filename, std::ios::out);
    std::ofstream qfs("debug.txt", std::ios::out);
    if (!ofs.is_open())
    {
        // std::cout << "Failed to open file.\n";
        return;
    }
    ofs << this->x() << " " << this->y() << " ";
    for (size_t i = 0; i < this->x() * this->y(); ++i)
    {
        ofs << (int)round(this->m_data[i]) << " ";
        qfs << this->m_data[i] << " ";
    }
    qfs.close();
    ofs.close();
}

template <class T>
image<T>::~image()
{

    if (m_data)
    {
        delete[] m_data;
        m_x = 0;
        m_y = 0;
        m_data = nullptr;
    }
}

template <class T>
image<T>::image(const size_t width, const size_t height) : m_x(width), m_y(height)
{
    allocate_memory();
}
template <class T>
image<T>::image(const size_t width, const size_t height, const T fill_value) : m_x(width), m_y(height)
{
    allocate_memory();
    initial_memory(fill_value);
}

template <class T>
void image<T>::reset_image_size(const size_t x, const size_t y, const T fill_value)
{
    this->m_x = x;
    this->m_y = y;
    delete_memory();
    allocate_memory();
    initial_memory(fill_value);
}

template <class T>
void image<T>::reset_image_size(const size_t x, const size_t y)
{
    this->m_x = x;
    this->m_y = y;
    delete_memory();
    allocate_memory();
}