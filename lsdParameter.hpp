#pragma once
#include <cmath>

class lsdParameter
{
private:
    // data
public:
    // LSD parameters

    // for gaussian_subsampling
    static constexpr double sigma_scale = 0.6; // Sigma for Gaussian filter is computed as sigma = sigma_scale/scale.
    static constexpr double scale_factor = 0.8;
    static constexpr double standard_deviation = sigma_scale / scale_factor;

    static constexpr double gaussian_prec = 3.0;
    const size_t gaussian_h;
    const size_t gaussian_n;

    // for gradient_computation
    static constexpr double max_gradient = 0.0;
    static constexpr double quant = 2.0;   // Bound to the quantization error on the gradient norm.
    static constexpr double ang_th = 22.5; // Gradient angle tolerance in degrees.

    static constexpr double prec = M_PI * ang_th / 180.0;
    static constexpr double p = ang_th / 180.0;

    const double threshold;
    const double rho;

    static constexpr double log_eps = 0.0;    // Detection threshold: -log10(NFA) > log_eps
    static constexpr double density_th = 0.7; // Minimal density of region points in rectangle.
    static constexpr int32_t n_bins = 1024;   // Number of bins in pseudo-ordering of gradient modulus.

    lsdParameter(/* args */) : gaussian_h(static_cast<size_t>(ceil(sigma_scale * sqrt(2.0 * gaussian_prec * log(10.0))))),
                               gaussian_n(1 + 2 * gaussian_h),
                               threshold(quant / sin(prec)),
                               rho(threshold)
    {
    }
};
