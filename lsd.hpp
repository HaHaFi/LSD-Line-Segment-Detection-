#pragma once
#include <iostream>
#include <algorithm>
#include <string>
#include <stddef.h>
#include <vector>
#include <cmath>
#include <map>
#include <list>
#include "image.hpp"
#include "image.cpp"
#include "lsdParameter.hpp"
#include <float.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

constexpr int32_t NOTUSED = 0;
constexpr int32_t NOTDEF = 1024;
constexpr int32_t USED = 128;

namespace py = pybind11;

struct line_segmentation
{

    double x1, y1, x2, y2; /* first and second point of the line segment */
    double width;          /* rectangle width */
    double p;
    double log_nfa;
};

struct rect
{
    double x1, y1, x2, y2; /* first and second point of the line segment */
    double width;          /* rectangle width */
    double x, y;           /* center of the rectangle */
    double theta;          /* angle */
    double dx, dy;         /* (dx,dy) is vector oriented as the line segment */
    double prec;           /* tolerance angle */
    double p;              /* probability of a point with angle within 'prec' */
    double length;
    rect &operator=(rect const &other)
    {
        this->x1 = other.x1;
        this->y1 = other.y1;
        this->x2 = other.x2;
        this->y2 = other.y2;
        this->width = other.width;
        this->x = other.x;
        this->y = other.y;
        this->theta = other.theta;
        this->dx = other.dx;
        this->dy = other.dy;
        this->prec = other.prec;
        this->p = other.p;
        this->length = other.length;
    }
};

struct coord
{
    coord() : x(0), y(0) {}
    coord(size_t x, size_t y) : x(x), y(y){};
    size_t x;
    size_t y;
    coord *next = nullptr;
};

struct point
{
    point() : x(0), y(0){};
    int32_t x;
    int32_t y;
};

class lsd
{
public:
    lsdParameter m_lsd_parameter;
    image<double> origin_image;

    py::object imread = py::module_::import("cv2").attr("imread");
    py::object imwrite = py::module_::import("cv2").attr("imwrite");
    py::object IMREAD_GRAYSCALE = py::module_::import("cv2").attr("IMREAD_GRAYSCALE");
    py::object zeros = py::module_::import("numpy").attr("zeros");

    lsd();
    ~lsd();
    image<double> gaussian_subsampling(image<double> &);
    image<double> gradient_computation(image<double> &img, coord *list, coord **list_head, image<double> &norm_img);
    void region_grow(int32_t x, int32_t y, image<double> &angle, point *reg, size_t *reg_size, double *reg_angle, image<uint8_t> &used, double prec);

    void region2rect(point *reg, size_t reg_size, image<double> &gradient_img, double reg_angel, struct rect *rec);
    double get_theta(point *reg, size_t reg_size, double x, double y, image<double> &gradient_img, double reg_angle);

    double rect_improve(rect &rec, image<double> &angles, double logNT, double log_eps);

    double rect_nfa(struct rect &rec, image<double> &angles, double logNT);
    double nfa(int n, int k, double p, double logNT);
    bool refine(point *reg, size_t *reg_size, image<double> &gradient_img,
                double reg_angle, rect *rec,
                image<uint8_t> &used, image<double> &angles);
    void read_img(py::object input_file_name)
    {

        py::array_t<uint8_t> source_img = imread(input_file_name, IMREAD_GRAYSCALE);
        auto r = source_img.unchecked<2>();
        origin_image.reset_image_size(r.shape(1), r.shape(0), 128);
        for (py::ssize_t i = 0; i < r.shape(0); i++)
        {
            for (py::ssize_t j = 0; j < r.shape(1); j++)
            {
                origin_image(static_cast<size_t>(j), static_cast<size_t>(i)) = static_cast<double>(r(i, j) + 1);
            }
        }
    }
    void write_img(std::string filename, image<double> &input_img)
    {

        std::pair<int, int> output_size{input_img.y(), input_img.x()};
        py::array_t<int32_t> output_img = zeros(py::cast(output_size));
        auto r = output_img.mutable_unchecked<2>();

        for (py::ssize_t i = 0; i < r.shape(0); i++)
            for (py::ssize_t j = 0; j < r.shape(1); j++)
            {
                r(i, j) = static_cast<int32_t>(input_img(j, i));
            }
        imwrite(py::cast(filename), output_img);
    }

    std::vector<double> line_segment_detector()
    {

        std::vector<line_segmentation> output;
        std::vector<double> test_output;
        int32_t output_count = 0;
        line_segmentation output_element;

        image<double> scale_img = gaussian_subsampling(origin_image);

        coord *list = new coord[scale_img.x() * scale_img.y()];
        coord *list_head = nullptr;

        image<double> gradient_img(scale_img.x(), scale_img.y());

        image<double> angle = gradient_computation(scale_img, list, &list_head, gradient_img);

        const size_t x_size = angle.x();
        const size_t y_size = angle.y();

        const double logNT = 5.0 * (log10(static_cast<double>(x_size)) + log10(static_cast<double>(y_size))) / 2.0 + log10(11.0);

        uint32_t min_reg_size = static_cast<uint32_t>(-logNT / log10(m_lsd_parameter.p));
        image<int32_t> region(x_size, y_size, 0);
        image<uint8_t> used(x_size, y_size, NOTUSED);

        point *reg = new point[x_size * y_size];
        size_t reg_size = 0;
        double reg_angle = 0.0;

        for (; list_head; list_head = list_head->next)
        {
            const size_t x = list_head->x;
            const size_t y = list_head->y;
            const size_t index = x + y * x_size;
            rect rec;
            if (used[index] == NOTUSED && angle[index] != NOTDEF)
            {

                region_grow(x, y, angle, reg, &reg_size, &reg_angle, used, m_lsd_parameter.prec);

                // must large than threshold
                if (reg_size < min_reg_size)
                {

                    continue;
                }

                region2rect(reg, reg_size, gradient_img, reg_angle, &rec);

                if (!refine(reg, &reg_size, gradient_img, reg_angle,
                            &rec, used, angle))
                    continue;

                // double log_nfa = rect_improve(rec, angle, logNT, m_lsd_parameter.log_eps);
                // if (log_nfa <= m_lsd_parameter.log_eps)
                //     continue;

                // for output
                rec.x1 += 0.5;
                rec.y1 += 0.5;
                rec.x2 += 0.5;
                rec.y2 += 0.5;

                rec.length = sqrt((rec.x1 - rec.x2) * (rec.x1 - rec.x2) + (rec.y1 - rec.y2) * (rec.y1 - rec.y2));

                /* scale the result values if a subsampling was performed */
                rec.x1 /= m_lsd_parameter.scale_factor;
                rec.y1 /= m_lsd_parameter.scale_factor;
                rec.x2 /= m_lsd_parameter.scale_factor;
                rec.y2 /= m_lsd_parameter.scale_factor;

                rec.width /= m_lsd_parameter.scale_factor;

                output_element.x1 = static_cast<int32_t>(rec.x1);
                output_element.y1 = static_cast<int32_t>(rec.y1);
                output_element.x2 = static_cast<int32_t>(rec.x2);
                output_element.y2 = static_cast<int32_t>(rec.y2);
                // output_element.log_nfa = log_nfa;

                output_element.p = rec.p;
                output.push_back(output_element);
                test_output.push_back(rec.x1);
                test_output.push_back(rec.y1);
                test_output.push_back(rec.x2);
                test_output.push_back(rec.y2);
            }
        }
        delete[] list;
        delete[] reg;
        // return;
        return test_output;
    }
};
