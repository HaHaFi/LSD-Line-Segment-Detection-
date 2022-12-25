#include "lsd.hpp"

lsd::lsd(/* args */) : m_lsd_parameter(), origin_image(0, 0)
{
}

lsd::~lsd()
{
}

std::vector<double> gaussian_kernel_withoutpassref(double sigma, double mean)
{
    double sum = 0.0;
    double val;
    std::vector<double> kernel{7, 0};
    for (int32_t i = 0; i < kernel.size(); i++)
    {
        val = (static_cast<double>(i) - mean) / sigma;
        kernel[i] = exp(-0.5 * val * val);
        sum += kernel[i];
    }

    // normalization
    if (sum >= 0.0)
        for (int32_t i = 0; i < kernel.size(); i++)
            kernel[i] /= sum;
    return kernel;
}

void gaussian_kernel(std::vector<double> &kernel, double sigma, double mean)
{
    double sum = 0.0;
    double val;

    for (int32_t i = 0; i < kernel.size(); i++)
    {
        val = (static_cast<double>(i) - mean) / sigma;
        kernel[i] = exp(-0.5 * val * val);
        sum += kernel[i];
    }

    // normalization
    if (sum >= 0.0)
        for (int32_t i = 0; i < kernel.size(); i++)
            kernel[i] /= sum;
}

image<double> lsd::gaussian_subsampling(image<double> &img)
{

    // compute width and height after scaled
    const size_t scaled_x = static_cast<size_t>(ceil(img.x() * m_lsd_parameter.scale_factor));
    const size_t scaled_y = static_cast<size_t>(ceil(img.y() * m_lsd_parameter.scale_factor));

    // preallocate the image memory
    image<double> middle_img(scaled_x, img.y());
    image<double> output_img(scaled_x, scaled_y);

    // kernel size with size n
    std::vector<double> kernel(m_lsd_parameter.gaussian_n);
    // py::print("x,y ", scaled_x, img.y());
    // py::print("x,y ", scaled_x, scaled_y);
    // first subsample with x_axis
    for (int32_t x_new = 0; x_new < middle_img.x(); ++x_new)
    {

        double x_ori_double = static_cast<double>(x_new) / m_lsd_parameter.scale_factor;
        int32_t x_ori_uint = static_cast<int32_t>(floor(x_ori_double + 0.5));

        gaussian_kernel(kernel, m_lsd_parameter.sigma_scale, static_cast<double>(m_lsd_parameter.gaussian_h) + x_ori_double - static_cast<double>(x_ori_uint));

        for (int32_t y = 0; y < middle_img.y(); ++y)
        {
            double sum = 0.0;
            double kernel_check = 0.0;
            for (int32_t i = 0; i < m_lsd_parameter.gaussian_n; ++i)
            {
                int32_t x_index = x_ori_uint + i - m_lsd_parameter.gaussian_h;
                if (x_index < 0)
                {
                    x_index = -x_index - 1;
                }
                if (x_index >= img.x())
                {
                    x_index = img.x() - 1;
                }

                sum += kernel[i] * img(x_index, y);
                kernel_check += kernel[i];
            }

            middle_img(x_new, y) = sum;
        }
    }

    // second subsample with y_axis
    for (int32_t y_new = 0; y_new < output_img.y(); ++y_new)
    {

        double y_ori_double = static_cast<double>(y_new) / m_lsd_parameter.scale_factor;
        int32_t y_ori_uint = static_cast<int32_t>(floor(y_ori_double + 0.5));

        gaussian_kernel(kernel, m_lsd_parameter.sigma_scale, static_cast<double>(m_lsd_parameter.gaussian_h) + y_ori_double - static_cast<double>(y_ori_uint));

        for (int32_t x = 0; x < output_img.x(); ++x)
        {
            double sum = 0.0;

            for (int32_t i = 0; i < m_lsd_parameter.gaussian_n; ++i)
            {
                int32_t y_index = y_ori_uint + i - m_lsd_parameter.gaussian_h;
                if (y_index < 0)
                {
                    y_index = -y_index - 1;
                }
                if (y_index >= middle_img.y())
                {
                    y_index = middle_img.y() - 1;
                }

                sum += kernel[i] * middle_img(x, y_index);
            }

            output_img(x, y_new) = sum;
        }
    }

    return output_img;
}

image<double> lsd::gradient_computation(image<double> &img, coord *list, coord **list_head, image<double> &norm_img)
{

    double max_gradient = 0.0;

    image<double> gradient_img(img.x(), img.y());
    // compute gradient
    for (int32_t x = 0; x < gradient_img.x() - 1; ++x)
    {
        // const size_t y_index = y * gradient_img.x();
        for (int32_t y = 0; y < gradient_img.y() - 1; ++y)
        {
            // const size_t index = y_index + x;
            const size_t index = y * gradient_img.x() + x;
            /*
               Norm 2 computation using 2x2 pixel window:
                 A B
                 C D
               and
                 com1 = D-A,  com2 = B-C.
               Then
                 gx = B+D - (A+C)   horizontal difference
                 gy = C+D - (A+B)   vertical difference
               com1 and com2 are just to avoid 2 additions.
             */
            // const double com1 = img[index + 1 + img.x()] - img[index];
            // const double com2 = img[index + 1] - img[index + img.x()];
            const double com1 = img(x + 1, y + 1) - img(x, y);
            const double com2 = img(x + 1, y) - img(x, y + 1);
            const double gx = com1 + com2;
            const double gy = com1 - com2;
            const double norm = sqrt((gx * gx + gy * gy) * 0.25);

            norm_img[index] = norm;
            if (norm <= m_lsd_parameter.threshold)
            {
                gradient_img[index] = NOTDEF;
            }
            else
            {
                gradient_img[index] = atan2(gx, -gy);

                if (max_gradient < norm)
                {
                    max_gradient = norm;
                }
            }
        }
    }

    // puedo sort with bin 1024
    constexpr size_t n_bin = 1024;

    // size 1024
    std::vector<coord *> start_list(n_bin, nullptr);
    std::vector<coord *> end_list(n_bin, nullptr);
    for (size_t i = 0; i < n_bin; i++)
    {
        start_list[i] = nullptr;
        end_list[i] = nullptr;
    }

    coord *head_of_list = nullptr;
    coord *end_of_list = nullptr;

    for (size_t y = 0; y < gradient_img.y() - 1; ++y)
    {
        const size_t y_index = y * gradient_img.x();
        for (size_t x = 0; x < gradient_img.x() - 1; ++x)
        {
            const size_t index = y_index + x;

            size_t bin_index = static_cast<size_t>(norm_img[index] * static_cast<double>(n_bin) / max_gradient);

            if (bin_index >= n_bin)
            {
                bin_index -= 1;
            }

            if (!start_list[bin_index])
            {

                start_list[bin_index] = end_list[bin_index] = list + index;
            }
            else
            {

                end_list[bin_index]->next = list + index;
                end_list[bin_index] = list + index;
            }

            end_list[bin_index]->x = x;
            end_list[bin_index]->y = y;
        }
    }

    for (int64_t i = n_bin - 1; i > 0; --i)
    {
        if (!head_of_list && start_list[i])
        {
            head_of_list = start_list[i];
            end_of_list = end_list[i];
        }
        else if (start_list[i])
        {
            end_of_list->next = start_list[i];
            end_of_list = end_list[i];
        }
    }
    *list_head = head_of_list;

    return gradient_img;
}

void lsd::region2rect(point *reg, size_t reg_size, image<double> &gradient_img, double reg_angel, struct rect *rec)
{

    double x = 0, y = 0, sum = 0, weight = 0, dx = 0, dy = 0, l_min = 0, l_max = 0, w_min = 0, w_max = 0;
    for (int32_t i = 0; i < reg_size; i++)
    {
        weight = gradient_img(reg[i].x, reg[i].y);
        x += static_cast<double>(reg[i].x) * weight;
        y += static_cast<double>(reg[i].y) * weight;
        sum += weight;
    }

    x /= sum;
    y /= sum;
    double theta = get_theta(reg, reg_size, x, y, gradient_img, reg_angel);

    dx = cos(x);
    dy = sin(y);

    double l, w;
    for (int32_t i = 0; i < reg_size; ++i)
    {
        l = ((double)reg[i].x - x) * dx + ((double)reg[i].y - y) * dy;
        w = -((double)reg[i].x - x) * dy + ((double)reg[i].y - y) * dx;

        if (l > l_max)
            l_max = l;
        if (l < l_min)
            l_min = l;
        if (w > w_max)
            w_max = w;
        if (w < w_min)
            w_min = w;
    }

    /* store values */
    rec->x1 = x + l_min * dx;
    rec->y1 = y + l_min * dy;
    rec->x2 = x + l_max * dx;
    rec->y2 = y + l_max * dy;
    rec->width = w_max - w_min;
    rec->x = x;
    rec->y = y;
    rec->theta = theta;
    rec->dx = dx;
    rec->dy = dy;
    rec->prec = m_lsd_parameter.prec;
    rec->p = m_lsd_parameter.p;

    /* we impose a minimal width of one pixel

       A sharp horizontal or vertical step would produce a perfectly
       horizontal or vertical region. The width computed would be
       zero. But that corresponds to a one pixels width transition in
       the image.
     */
    if (rec->width < 1.0)
        rec->width = 1.0;
}
double angle_diff(double a, double b)
{
    a -= b;
    constexpr double m_2_pi = (M_PI)*2;
    while (a <= -M_PI)
        a += m_2_pi;
    while (a > M_PI)
        a -= m_2_pi;
    if (a < 0.0)
        a = -a;
    return a;
}

double angle_diff_withoutcheck(double a, double b)
{
    a -= b;
    constexpr double m_2_pi = (M_PI)*2;
    while (a <= -M_PI)
        a += m_2_pi;
    while (a > M_PI)
        a -= m_2_pi;

    return a;
}
double lsd::get_theta(point *reg, size_t reg_size, double x, double y, image<double> &gradient_img, double reg_angle)
{
    double weight = 0, Ixx = 0, Iyy = 0, Ixy = 0, lambda = 0, theta = 0;

    for (int32_t i = 0; i < reg_size; ++i)
    {
        weight = gradient_img(reg[i].x, reg[i].y);
        Ixx += (static_cast<double>(reg[i].y) - y) * (static_cast<double>(reg[i].y) - y) * weight;
        Iyy += (static_cast<double>(reg[i].x) - x) * (static_cast<double>(reg[i].x) - x) * weight;
        Ixy -= (static_cast<double>(reg[i].x) - x) * (static_cast<double>(reg[i].y) - y) * weight;
    }

    lambda = 0.5 * (Ixx + Iyy - sqrt((Ixx - Iyy) * (Ixx - Iyy) + 4.0 * Ixy * Ixy));
    theta = fabs(Ixx) > fabs(Iyy) ? atan2(lambda - Ixx, Ixy) : atan2(Ixy, lambda - Iyy);

    if (angle_diff(theta, reg_angle) > m_lsd_parameter.prec)
        theta += M_PI;

    return theta;
}

bool is_similiar_direction(double angle, double theta, double prec)
{
    constexpr double m_3_2_pi = ((M_PI) / 2) * 3;
    constexpr double m_2_pi = (M_PI)*2;
    theta -= angle;

    if (theta < 0.0)
    {
        theta = -theta;
    }
    if (theta > m_3_2_pi)
    {
        theta -= m_2_pi;
        if (theta < 0.0)
        {
            theta = -theta;
        }
    }
    return theta <= prec;
}

void lsd::region_grow(int32_t x, int32_t y, image<double> &angle, point *reg, size_t *reg_size, double *reg_angle, image<uint8_t> &used, double prec)
{
    double sumdx, sumdy;

    // the first point
    *reg_size = 1;
    reg[0].x = x;
    reg[0].y = y;
    *reg_angle = angle[x + y * angle.x()];
    sumdx = cos(*reg_angle);
    sumdy = sin(*reg_angle);
    used[x + y * angle.x()] = USED;

    for (size_t i = 0; i < *reg_size; ++i)
    {
        for (int32_t xx = x - 1; xx <= reg[i].x + 1; ++xx)
        {
            for (int32_t yy = y - 1; yy <= reg[i].y + 1; ++yy)
            {
                const size_t index = xx + yy * angle.x();
                if (xx >= 0 && yy >= 0 && xx < static_cast<int32_t>(angle.x()) && yy < static_cast<int32_t>(angle.y()) && used[xx + yy * angle.x()] != USED && is_similiar_direction(angle[xx + yy * angle.x()], *reg_angle, prec))
                {
                    used[index] = USED;
                    reg[*reg_size].x = xx;
                    reg[*reg_size].y = yy;
                    *reg_size = (*reg_size) + 1;

                    sumdx += cos(angle[index]);
                    sumdy += sin(angle[index]);

                    *reg_angle = atan2(sumdy, sumdx);
                }
            }
        }
    }
}

double dot(double x1, double y1, double x2, double y2)
{
    return x1 * x2 + y1 * y2;
}

double cross(double x1, double y1, double x2, double y2)
{
    return x1 * y2 - x2 * y1;
}

double length(double dx, double dy)
{
    return sqrt(dx * dx + dy * dy);
}
bool check_in_rect(struct rect &rec, double x, double y)
{
    double dx = rec.x2 - rec.x1;
    double dy = rec.y2 - rec.y1;

    double x1_dx = x - rec.x1;
    double y1_dy = y - rec.y1;

    double x2_dx = x - rec.x2;
    double y2_dy = y - rec.y2;

    if (dot(dx, dy, x1_dx, y1_dy) <= 0)
        return false;
    if (dot(dx, dy, x2_dx, y2_dy) >= 0)
        return false;
    return rec.width > (fabs(cross(dx, dy, x1_dx, y1_dy)) / length(dx, dy));
}

/*----------------------------------------------------------------------------*/
/** Doubles relative error factor
 */
static constexpr double RELATIVE_ERROR_FACTOR = 100.0;
bool double_equal(double a, double b)
{
    double abs_diff, aa, bb, abs_max;

    /* trivial case */
    if (a == b)
        return true;

    abs_diff = fabs(a - b);
    aa = fabs(a);
    bb = fabs(b);
    abs_max = aa > bb ? aa : bb;

    /* DBL_MIN is the smallest normalized number, thus, the smallest
       number whose relative error is bounded by DBL_EPSILON. For
       smaller numbers, the same quantization steps as for DBL_MIN
       are used. Then, for smaller numbers, a meaningful "relative"
       error should be computed by dividing the difference by DBL_MIN. */
    if (abs_max < DBL_MIN)
        abs_max = DBL_MIN;

    /* equal if relative error <= factor x eps */
    return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * DBL_EPSILON);
}

static constexpr size_t TABSIZE = 100000;
double log_gamma_windschitl(double x)
{
    return 0.918938533204673 + (x - 0.5) * log(x) - x + 0.5 * x * log(x * sinh(1 / x) + 1 / (810.0 * pow(x, 6.0)));
}
double log_gamma_lanczos(double x)
{
    constexpr double q[7] = {75122.6331530, 80916.6278952, 36308.2951477,
                             8687.24529705, 1168.92649479, 83.8676043424,
                             2.50662827511};
    double a = (x + 0.5) * log(x + 5.5) - (x + 5.5);
    double b = 0.0;

    for (int32_t n = 0; n < 7; n++)
    {
        a -= log(x + (double)n);
        b += q[n] * pow(x, (double)n);
    }
    return a + log(b);
}
double log_gamma(double x)
{
    if (x > 15.0)
    {
        return log_gamma_windschitl(x);
    }
    return log_gamma_lanczos(x);
}

// reference by  https://github.com/theWorldCreator/LSD
double lsd::nfa(int n, int k, double p, double logNT)
{
    static double inv[TABSIZE]; /* table to keep computed inverse values */
    double tolerance = 0.1;     /* an error of 10% in the result is accepted */
    double log1term, term, bin_term, mult_term, bin_tail, err, p_term;
    int i;

    /* check parameters */
    if (n < 0 || k < 0 || k > n || p <= 0.0 || p >= 1.0)
        std::cout << "nfa: wrong n, k or p values.\n";

    /* trivial cases */
    if (n == 0 || k == 0)
        return -logNT;
    if (n == k)
        return -logNT - (double)n * log10(p);

    /* probability term */
    p_term = p / (1.0 - p);

    /* compute the first term of the series */
    /*
       binomial_tail(n,k,p) = sum_{i=k}^n bincoef(n,i) * p^i * (1-p)^{n-i}
       where bincoef(n,i) are the binomial coefficients.
       But
         bincoef(n,k) = gamma(n+1) / ( gamma(k+1) * gamma(n-k+1) ).
       We use this to compute the first term. Actually the log of it.
     */
    log1term = log_gamma((double)n + 1.0) - log_gamma((double)k + 1.0) - log_gamma((double)(n - k) + 1.0) + (double)k * log(p) + (double)(n - k) * log(1.0 - p);
    term = exp(log1term);

    /* in some cases no more computations are needed */
    if (double_equal(term, 0.0)) /* the first term is almost zero */
    {
        if ((double)k > (double)n * p)         /* at begin or end of the tail?  */
            return -log1term / M_LN10 - logNT; /* end: use just the first term  */
        else
            return -logNT; /* begin: the tail is roughly 1  */
    }

    /* compute more terms if needed */
    bin_tail = term;
    for (i = k + 1; i <= n; i++)
    {
        /*
           As
             term_i = bincoef(n,i) * p^i * (1-p)^(n-i)
           and
             bincoef(n,i)/bincoef(n,i-1) = n-1+1 / i,
           then,
             term_i / term_i-1 = (n-i+1)/i * p/(1-p)
           and
             term_i = term_i-1 * (n-i+1)/i * p/(1-p).
           1/i is stored in a table as they are computed,
           because divisions are expensive.
           p/(1-p) is computed only once and stored in 'p_term'.
         */
        bin_term = (double)(n - i + 1) * (i < TABSIZE ? (inv[i] != 0.0 ? inv[i] : (inv[i] = 1.0 / (double)i)) : 1.0 / (double)i);

        mult_term = bin_term * p_term;
        term *= mult_term;
        bin_tail += term;
        if (bin_term < 1.0)
        {
            /* When bin_term<1 then mult_term_j<mult_term_i for j>i.
               Then, the error on the binomial tail when truncated at
               the i term can be bounded by a geometric series of form
               term_i * sum mult_term_i^j.                            */
            err = term * ((1.0 - pow(mult_term, (double)(n - i + 1))) /
                              (1.0 - mult_term) -
                          1.0);

            /* One wants an error at most of tolerance*final_result, or:
               tolerance * abs(-log10(bin_tail)-logNT).
               Now, the error that can be accepted on bin_tail is
               given by tolerance*final_result divided by the derivative
               of -log10(x) when x=bin_tail. that is:
               tolerance * abs(-log10(bin_tail)-logNT) / (1/bin_tail)
               Finally, we truncate the tail if the error is less than:
               tolerance * abs(-log10(bin_tail)-logNT) * bin_tail        */
            if (err < tolerance * fabs(-log10(bin_tail) - logNT) * bin_tail)
                break;
        }
    }
    return -log10(bin_tail) - logNT;
}

double lsd::rect_nfa(struct rect &rec, image<double> &angles, double logNT)
{

    int pts = 0;
    int alg = 0;

    std::vector<double> corner_x{rec.x1 - rec.dy * rec.width / 2.0,
                                 rec.x2 - rec.dy * rec.width / 2.0,
                                 rec.x2 + rec.dy * rec.width / 2.0,
                                 rec.x1 + rec.dy * rec.width / 2.0};
    std::vector<double> corner_y{rec.y1 + rec.dx * rec.width / 2.0,
                                 rec.y2 + rec.dx * rec.width / 2.0,
                                 rec.y2 - rec.dx * rec.width / 2.0,
                                 rec.y1 - rec.dx * rec.width / 2.0};

    const int32_t max_x = static_cast<int32_t>(*std::max_element(corner_x.begin(), corner_x.end()));
    const int32_t min_x = static_cast<int32_t>(*std::min_element(corner_x.begin(), corner_x.end()));
    const int32_t max_y = static_cast<int32_t>(*std::max_element(corner_y.begin(), corner_y.end()));
    const int32_t min_y = static_cast<int32_t>(*std::min_element(corner_y.begin(), corner_y.end()));
    /* compute the total number of pixels and of aligned points in 'rec' */
    for (int32_t xx = min_x; xx <= max_x; ++xx)
    {
        for (int32_t yy = min_y; yy <= max_y; ++yy)
        {
            if (xx >= 0 && yy >= 0 && xx < angles.x() && yy < angles.y() && check_in_rect(rec, xx, yy))
            {
                ++pts; /* total number of pixels counter */
                if (is_similiar_direction(angles(xx, yy), rec.theta, rec.prec))
                    ++alg; /* aligned points counter */
            }
        }
    }

    return nfa(pts, alg, rec.p, logNT); /* compute NFA value */
}

double lsd::rect_improve(rect &rec, image<double> &angles, double logNT, double log_eps)
{
    struct rect r = rec;
    double log_nfa, log_nfa_new;
    double delta = 0.5;
    double delta_2 = delta / 2.0;

    log_nfa = rect_nfa(rec, angles, logNT);

    if (log_nfa > log_eps)
        return log_nfa;

    /* try finer precisions */
    r = rec;

    for (int32_t n = 0; n < 5; n++)
    {
        r.p /= 2.0;
        r.prec = r.p * M_PI;
        log_nfa_new = rect_nfa(r, angles, logNT);
        if (log_nfa_new > log_nfa)
        {
            log_nfa = log_nfa_new;
            rec = r;
        }
    }

    if (log_nfa > log_eps)
        return log_nfa;

    /* try to reduce width */
    r = rec;

    for (int32_t n = 0; n < 5; n++)
    {
        if ((r.width - delta) >= 0.5)
        {
            r.width -= delta;
            log_nfa_new = rect_nfa(r, angles, logNT);

            if (log_nfa_new > log_nfa)
            {
                rec = r;
                log_nfa = log_nfa_new;
            }
        }
    }

    if (log_nfa > log_eps)
        return log_nfa;

    /* try to reduce one side of the rectangle */
    r = rec;
    for (int32_t n = 0; n < 5; n++)
    {
        if ((r.width - delta) >= 0.5)
        {
            r.x1 += -r.dy * delta_2;
            r.y1 += r.dx * delta_2;
            r.x2 += -r.dy * delta_2;
            r.y2 += r.dx * delta_2;
            r.width -= delta;
            log_nfa_new = rect_nfa(r, angles, logNT);
            if (log_nfa_new > log_nfa)
            {
                rec = r;
                log_nfa = log_nfa_new;
            }
        }
    }

    if (log_nfa > log_eps)
        return log_nfa;

    /* try to reduce the other side of the rectangle */
    r = rec;
    for (int32_t n = 0; n < 5; n++)
    {
        if ((r.width - delta) >= 0.5)
        {
            r.x1 -= -r.dy * delta_2;
            r.y1 -= r.dx * delta_2;
            r.x2 -= -r.dy * delta_2;
            r.y2 -= r.dx * delta_2;
            r.width -= delta;
            log_nfa_new = rect_nfa(r, angles, logNT);
            if (log_nfa_new > log_nfa)
            {
                rec = r;
                log_nfa = log_nfa_new;
            }
        }
    }

    if (log_nfa > log_eps)
        return log_nfa;

    /* try even finer precisions */
    r = rec;
    for (int32_t n = 0; n < 5; n++)
    {
        r.p /= 2.0;
        r.prec = r.p * M_PI;
        log_nfa_new = rect_nfa(r, angles, logNT);
        if (log_nfa_new > log_nfa)
        {
            log_nfa = log_nfa_new;
            rec = r;
        }
    }

    return log_nfa;
}

bool lsd::refine(point *reg, size_t *reg_size, image<double> &gradient_img,
                 double reg_angle, rect *rec,
                 image<uint8_t> &used, image<double> &angles)
{
    double angle, ang_d, mean_angle, tau, density, xc, yc, ang_c, sum, s_sum;
    size_t i, n;

    density = static_cast<double>(*reg_size) /
              (length(rec->x1 - rec->x2, rec->y1 - rec->y2) * rec->width);

    if (density >= m_lsd_parameter.density_th)
        return true;

    xc = static_cast<double>(reg[0].x);
    yc = static_cast<double>(reg[0].y);
    ang_c = angles[reg[0].x + reg[0].y * angles.x()];
    sum = s_sum = 0.0;
    n = 0;
    for (i = 0; i < *reg_size; i++)
    {
        used[reg[i].x + reg[i].y * used.x()] = NOTUSED;
        if (length(xc - static_cast<double>(reg[i].x), yc - static_cast<double>(reg[i].y)) < rec->width)
        {
            angle = angles[reg[i].x + reg[i].y * angles.x()];
            ang_d = angle_diff_withoutcheck(angle, ang_c);
            sum += ang_d;
            s_sum += ang_d * ang_d;
            ++n;
        }
    }
    mean_angle = sum / static_cast<double>(n);
    tau = 2.0 * sqrt((s_sum - 2.0 * mean_angle * sum) / static_cast<double>(n) + mean_angle * mean_angle);

    region_grow(reg[0].x, reg[0].y, angles, reg, reg_size, &reg_angle, used, tau);

    if (*reg_size < 2)
        return false;

    region2rect(reg, *reg_size, gradient_img, reg_angle, rec);

    return true;
}

PYBIND11_MODULE(_lsd, m)
{
    // for testing
    m.def("gaussian_kernel", &gaussian_kernel_withoutpassref);
    m.def("is_similiar_direction", &is_similiar_direction);
    m.def("dot", &dot);
    m.def("cross", &cross);
    m.def("length", &length);
    m.def("angle_diff", &angle_diff);

    py::class_<lsd>(m, "lsd")
        .def(py::init<>())
        .def("read_img", &lsd::read_img)
        // for user usage
        .def("line_segment_detector", &lsd::line_segment_detector)
        // for user usage
        .def("width", [](lsd &c)
             { return c.origin_image.x(); })
        // for user usage
        .def("height", [](lsd &c)
             { return c.origin_image.y(); })
        // for testing
        .def(
            "__getitem__", [](lsd &c, std::tuple<size_t, size_t> a)
            { return c.origin_image(std::get<0>(a), std::get<1>(a)); },
            py::is_operator())
        // for testing
        .def(
            "__setitem__", [](lsd &c, std::tuple<size_t, size_t> a, double value)
            { c.origin_image(std::get<0>(a), std::get<1>(a)) = value; },
            py::is_operator());
}