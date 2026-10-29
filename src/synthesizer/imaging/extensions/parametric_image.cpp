/******************************************************************************
 * Images of parametric populations.
 *
 * Each population has an elliptical radial profile (Sersic, Gaussian or a
 * point source), optionally restricted to an annulus, and a signal per
 * image (e.g. its luminosity in each filter, or its spectrum for a data
 * cube). The image is the sum over populations of each population's light
 * in each pixel, normalised by the profile's (analytic) light within its
 * annulus, times its signal. Light falling outside the image is lost.
 *
 * Each pixel is integrated to a requested relative tolerance:
 *   - where the profile is smooth on the scale of a pixel the pixel centre
 *     sample is used (its error, the midpoint rule's res^2 / 24 times the
 *     Laplacian, is estimated from the neighbouring centre samples);
 *   - where it isn't, the pixel is integrated with adaptive Gauss-Legendre
 *     cubature;
 *   - pixels straddling an annulus edge, or near the centre of a profile
 *     with a cusp, are integrated in thin elliptical shells whose light
 *     comes from the profile's enclosed fraction, each spread over the pixel
 *     by the exact fraction of its ellipse inside the pixel. This makes the
 *     split between annuli and the light of a cusp exact.
 *****************************************************************************/

/* C includes. */
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

/* Python includes. */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "../../extensions/numpy_init.h"

#include <Python.h>

/* Local includes. */
#include "../../extensions/property_funcs.h"
#include "../../extensions/python_to_cpp.h"
#include "../../extensions/timers.h"
#ifdef ATOMIC_TIMING
#include "../../extensions/timers_init.h"
#endif

#ifdef WITH_OPENMP
#include <omp.h>
#endif

/* The layout of a population's profile parameters. */
enum ProfileParam {
  P_TYPE = 0, /* 0: Sersic, 1: Gaussian, 2: point source. */
  P_X0,       /* The centre (in units of the resolution's units). */
  P_Y0,       /* */
  P_COS,      /* The cosine and sine of the major axis' angle. */
  P_SIN,      /* */
  P_Q,        /* The minor to major axis ratio. */
  P_SCALE,    /* The effective radius (Sersic) or major sigma (Gaussian). */
  P_INDEX,    /* The Sersic index. */
  P_BN,       /* The Sersic b_n. */
  P_RLO,      /* The inner and outer (elliptical) radii of the annulus
                 (an outer radius of UNBOUNDED or more is unbounded). */
  P_RHI,      /* */
  P_CUSP,     /* Whether the centre is integrated in shells. */
  NPARAMS
};

enum ProfileType { SERSIC = 0, GAUSSIAN = 1, POINT = 2 };

/* An outer annulus radius at or beyond this is unbounded. Infinities are
 * avoided because the extensions build with -ffast-math, under which the
 * compiler may assume every value is finite (see floating_point_utils.h). */
static const double UNBOUNDED = 1e300;

/* Pixels within this many pixels of a cusp's centre are integrated in
 * shells. */
static const double CORE_PIXELS = 2.0;

/* The profile is resolved where it changes by a factor e over at least this
 * many pixels. Elsewhere the centre samples are too far apart to estimate a
 * pixel's integral from, so the pixel is integrated with cubature. */
static const double RESOLVED_PIXELS = 2.0;

/* The deepest the adaptive cubature subdivides a pixel. */
static const int MAX_DEPTH = 10;

/* The number of (linearly spaced) shells across a pixel's radial span, the
 * number of log spaced shells below the first when a pixel reaches the
 * centre, and the number of geometrically graded shells above each radius
 * where a shell becomes tangent to one of the pixel's edges. */
static const int LINEAR_SHELLS = 24;
static const int CENTRAL_SHELLS = 64;
static const int TANGENT_GRADING = 8;
static const double SHELLS_PER_PIXEL = 16.0;

/* The largest change in the log of the surface brightness across a shell,
 * and the most pieces a shell is split into to achieve it. */
static const double MAX_LOG_STEP = 0.05;
static const int MAX_SPLITS = 256;

/**
 * @brief The regularised lower incomplete gamma function P(a, x).
 *
 * @param a: The shape (> 0).
 * @param x: The argument (>= 0).
 * @return P(a, x).
 */
static double gamma_p(double a, double x) {
  if (x <= 0.0) return 0.0;
  if (x >= UNBOUNDED) return 1.0;
  const double gln = std::lgamma(a);
  if (x < a + 1.0) {
    /* Series representation. */
    double ap = a, sum = 1.0 / a, del = sum;
    for (int n = 0; n < 1000; n++) {
      ap += 1.0;
      del *= x / ap;
      sum += del;
      if (std::fabs(del) < std::fabs(sum) * 1e-16) break;
    }
    return sum * std::exp(-x + a * std::log(x) - gln);
  }
  /* Continued fraction representation (modified Lentz). */
  const double tiny = 1e-300;
  double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
  for (int i = 1; i < 1000; i++) {
    const double an = -i * (i - a);
    b += 2.0;
    d = an * d + b;
    if (std::fabs(d) < tiny) d = tiny;
    c = b + an / c;
    if (std::fabs(c) < tiny) c = tiny;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) < 1e-16) break;
  }
  return 1.0 - std::exp(-x + a * std::log(x) - gln) * h;
}

/**
 * @brief A population's elliptical radial profile.
 */
struct Profile {
  int type;
  double x0, y0, cos, sin, q, scale, index, bn, rlo, rhi;
  bool cusp;

  /* The profile's (unnormalised) light over the whole plane. */
  double total;

  explicit Profile(const double *p) {
    type = static_cast<int>(p[P_TYPE]);
    x0 = p[P_X0];
    y0 = p[P_Y0];
    cos = p[P_COS];
    sin = p[P_SIN];
    q = p[P_Q];
    scale = p[P_SCALE];
    index = p[P_INDEX];
    bn = p[P_BN];
    rlo = p[P_RLO];
    rhi = p[P_RHI];
    cusp = p[P_CUSP] != 0.0;
    if (type == SERSIC) {
      total =
          2.0 * M_PI * index * q * scale * scale *
          std::exp(bn - 2.0 * index * std::log(bn) + std::lgamma(2.0 * index));
    } else if (type == GAUSSIAN) {
      total = 2.0 * M_PI * q * scale * scale;
    } else {
      total = 1.0;
    }
  }

  /* The elliptical radius (the semi-major axis of the contour) at (x, y). */
  inline double radius(double x, double y) const {
    const double dx = x - x0, dy = y - y0;
    const double a = dx * cos + dy * sin;
    const double b = (-dx * sin + dy * cos) / q;
    return std::sqrt(a * a + b * b);
  }

  /* The (unnormalised) surface brightness at a radius. */
  inline double intensity(double r) const {
    if (type == SERSIC) {
      return std::exp(-bn * (std::pow(r / scale, 1.0 / index) - 1.0));
    }
    const double u = r / scale;
    return std::exp(-0.5 * u * u);
  }

  /* The fraction of the light within a radius. */
  inline double enclosed(double r) const {
    if (r <= 0.0) return 0.0;
    if (r >= UNBOUNDED) return 1.0;
    if (type == SERSIC) {
      return gamma_p(2.0 * index, bn * std::pow(r / scale, 1.0 / index));
    }
    const double u = r / scale;
    return 1.0 - std::exp(-0.5 * u * u);
  }

  /* The length over which the surface brightness changes by a factor e at
   * a radius, along the minor axis (where it changes fastest). */
  inline double local_scale(double r) const {
    if (type == SERSIC) {
      const double rr = std::max(r, 1e-300);
      return q * index * std::pow(scale, 1.0 / index) *
             std::pow(rr, 1.0 - 1.0 / index) / bn;
    }
    return q * scale * scale / std::max(r, scale);
  }

  /* The surface brightness of the population (zero outside its annulus). */
  inline double value(double x, double y) const {
    const double r = radius(x, y);
    if (r < rlo || r >= rhi) return 0.0;
    return intensity(r);
  }
};

/**
 * @brief The arcs of an ellipse (in its parametric angle) inside an
 * interval of one coordinate.
 *
 * Along the ellipse of radius r the coordinate is
 * centre + r * amp * cos(t - phase), so it lies in [lo, hi] on (at most)
 * two arcs with closed form ends.
 *
 * @param r: The ellipse's radius.
 * @param centre: The coordinate of the ellipse's centre.
 * @param amp: The amplitude of the coordinate per unit radius.
 * @param phase: The phase of the coordinate.
 * @param lo: The lower edge of the interval.
 * @param hi: The upper edge of the interval.
 * @param starts: The output start angles of the two arcs.
 * @param lengths: The output lengths of the two arcs.
 */
static inline void get_arcs(double r, double centre, double amp, double phase,
                            double lo, double hi, double starts[2],
                            double lengths[2]) {
  const double a = r * amp;
  if (a <= 0.0) {
    /* A point lies wholly in the interval containing it. */
    const double len = (lo <= centre && centre < hi) ? M_PI : 0.0;
    starts[0] = phase;
    starts[1] = phase - M_PI;
    lengths[0] = lengths[1] = len;
    return;
  }
  const double c_lo = std::clamp((lo - centre) / a, -1.0, 1.0);
  const double c_hi = std::clamp((hi - centre) / a, -1.0, 1.0);
  const double u_hi = std::acos(c_lo);
  const double u_lo = std::acos(c_hi);
  const double len = std::max(u_hi - u_lo, 0.0);
  starts[0] = u_lo + phase;
  starts[1] = -u_hi + phase;
  lengths[0] = lengths[1] = len;
}

/**
 * @brief The overlap of two arcs on a circle (each at most half of it).
 */
static inline double arc_overlap(double s1, double l1, double s2, double l2) {
  if (l1 <= 0.0 || l2 <= 0.0) return 0.0;
  double d = std::fmod(s2 - s1, 2.0 * M_PI);
  if (d < 0.0) d += 2.0 * M_PI;
  return std::max(std::min(l1, d + l2) - d, 0.0) +
         std::max(std::min(l1, d - 2.0 * M_PI + l2), 0.0);
}

/**
 * @brief Integrate a population's light over a pixel in elliptical shells.
 *
 * The shells span the pixel's radial range. Each carries the light between
 * its edges (from the enclosed fraction), spread around its ellipse, of
 * which the pixel gets the fraction of the ellipse inside it. Shell edges
 * are placed at the annulus edges, at the pixel's corners (where the
 * fraction has a kink) and graded towards the radii where a shell becomes
 * tangent to one of the pixel's edges (where the fraction grows as a
 * square root).
 *
 * @return The pixel's (unnormalised) light.
 */
static double integrate_pixel_shells(const Profile &prof, double xl, double xh,
                                     double yl, double yh, double res) {
  /* The coordinates along the ellipse of radius r at parametric angle t:
   * x = x0 + r (cos(t) cos(phi) - q sin(t) sin(phi)), and
   * y = y0 + r (cos(t) sin(phi) + q sin(t) cos(phi)). */
  const double xa = prof.cos, xb = -prof.q * prof.sin;
  const double ya = prof.sin, yb = prof.q * prof.cos;
  const double xamp = std::hypot(xa, xb), xphase = std::atan2(xb, xa);
  const double yamp = std::hypot(ya, yb), yphase = std::atan2(yb, ya);

  /* The pixel's radial range: the radius is largest at a corner, and
   * changes by at most 1 / q per unit distance. */
  const double xc = 0.5 * (xl + xh), yc = 0.5 * (yl + yh);
  const double rc = prof.radius(xc, yc);
  const double half_diag = 0.5 * std::hypot(xh - xl, yh - yl);
  double rmin = std::max(rc - half_diag / prof.q, 0.0);
  double rmax = 0.0;
  const double cx[4] = {xl, xh, xl, xh}, cy[4] = {yl, yl, yh, yh};
  double corners[4];
  for (int i = 0; i < 4; i++) {
    corners[i] = prof.radius(cx[i], cy[i]);
    rmax = std::max(rmax, corners[i]);
  }
  rmin = std::max(rmin, prof.rlo);
  rmax = std::min(rmax, prof.rhi);
  if (rmax <= rmin) return 0.0;

  /* Collect the shell edges. */
  std::vector<double> edges;
  edges.reserve(LINEAR_SHELLS + CENTRAL_SHELLS + 4 * TANGENT_GRADING + 16);
  /* The shells are at most 1 / SHELLS_PER_PIXEL of a pixel apart, since
   * the part of a shell in the pixel changes on the scale of the pixel
   * (an elongated profile's pixels span many pixels in radius). */
  const int nlinear = std::max(
      LINEAR_SHELLS, (int)std::ceil((rmax - rmin) / res * SHELLS_PER_PIXEL));
  const double step = (rmax - rmin) / nlinear;
  for (int i = 0; i <= nlinear; i++) edges.push_back(rmin + i * step);
  if (rmin == 0.0 || rmin <= prof.rlo) {
    /* Resolve a cusp with log spaced shells below the first. */
    const double top = rmin + step;
    const double bottom = std::max(top * 1e-8, 1e-300);
    for (int i = 0; i < CENTRAL_SHELLS; i++) {
      edges.push_back(
          rmin + bottom * std::pow(top / bottom, (double)i / CENTRAL_SHELLS));
    }
  }
  for (int i = 0; i < 4; i++) edges.push_back(corners[i]);
  const double tangents[4] = {
      std::fabs(xl - prof.x0) / xamp, std::fabs(xh - prof.x0) / xamp,
      std::fabs(yl - prof.y0) / yamp, std::fabs(yh - prof.y0) / yamp};
  for (int i = 0; i < 4; i++) {
    for (int k = 0; k <= TANGENT_GRADING; k++) {
      const double grade = k == 0 ? 0.0 : step * std::pow(0.5, k - 1);
      edges.push_back(tangents[i] + grade);
    }
  }
  edges.push_back(prof.rlo);
  if (prof.rhi < UNBOUNDED) edges.push_back(prof.rhi);
  std::sort(edges.begin(), edges.end());

  /* The light a shell puts in the pixel. */
  auto shell_light = [&](double lo, double hi, double mass) {
    const double r = lo > 0.0 ? std::sqrt(lo * hi) : 0.5 * hi;
    double xs[2], xl2[2], ys[2], yl2[2];
    get_arcs(r, prof.x0, xamp, xphase, xl, xh, xs, xl2);
    get_arcs(r, prof.y0, yamp, yphase, yl, yh, ys, yl2);
    double overlap = 0.0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        overlap += arc_overlap(xs[i], xl2[i], ys[j], yl2[j]);
      }
    }
    return mass * overlap / (2.0 * M_PI);
  };

  /* Sum the light each shell puts in the pixel. A shell's light is
   * placed at its middle radius, so a shell across which the profile
   * changes steeply (e.g. a narrow Gaussian along its minor axis) is
   * split until it changes by at most MAX_LOG_STEP across each piece. */
  double light = 0.0;
  double lo = rmin, f_lo = prof.enclosed(rmin);
  for (const double hi : edges) {
    if (hi <= lo) continue;
    if (hi > rmax) break;
    const double log_step =
        std::fabs(std::log(prof.intensity(lo)) - std::log(prof.intensity(hi)));
    const int nsub = std::min(
        std::max((int)std::ceil(log_step / MAX_LOG_STEP), 1), MAX_SPLITS);
    double sub_lo = lo, sub_f_lo = f_lo;
    for (int k = 1; k <= nsub; k++) {
      const double sub_hi = k == nsub ? hi : lo + (hi - lo) * k / nsub;
      const double sub_f_hi = prof.enclosed(sub_hi);
      const double mass = sub_f_hi - sub_f_lo;
      if (mass > 0.0) light += shell_light(sub_lo, sub_hi, mass);
      sub_lo = sub_hi;
      sub_f_lo = sub_f_hi;
    }
    lo = hi;
    f_lo = sub_f_lo;
  }
  return light * prof.total;
}

/* The 3 point Gauss-Legendre nodes and weights on [-1, 1]. */
static const double GL_NODES[3] = {-0.7745966692414834, 0.0,
                                   0.7745966692414834};
static const double GL_WEIGHTS[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};

/**
 * @brief Integrate a population's light over a rectangle with the 3x3
 * Gauss-Legendre rule.
 */
static inline double gauss_legendre(const Profile &prof, double xl, double xh,
                                    double yl, double yh) {
  const double hx = 0.5 * (xh - xl), hy = 0.5 * (yh - yl);
  const double mx = 0.5 * (xh + xl), my = 0.5 * (yh + yl);
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      sum += GL_WEIGHTS[i] * GL_WEIGHTS[j] *
             prof.value(mx + hx * GL_NODES[i], my + hy * GL_NODES[j]);
    }
  }
  return sum * hx * hy;
}

/**
 * @brief Integrate a population's light over a rectangle with adaptive
 * Gauss-Legendre cubature.
 *
 * The rectangle's estimate is compared with the sum of its four quarters'
 * and each quarter subdivided further until they agree to the tolerance.
 */
static double adaptive_cubature(const Profile &prof, double xl, double xh,
                                double yl, double yh, double coarse,
                                double tol, int depth) {
  const double xm = 0.5 * (xl + xh), ym = 0.5 * (yl + yh);
  const double q[4] = {gauss_legendre(prof, xl, xm, yl, ym),
                       gauss_legendre(prof, xm, xh, yl, ym),
                       gauss_legendre(prof, xl, xm, ym, yh),
                       gauss_legendre(prof, xm, xh, ym, yh)};
  const double fine = q[0] + q[1] + q[2] + q[3];
  if (std::fabs(fine - coarse) <= tol || depth >= MAX_DEPTH) return fine;
  const double sub_tol = 0.5 * tol;
  return adaptive_cubature(prof, xl, xm, yl, ym, q[0], sub_tol, depth + 1) +
         adaptive_cubature(prof, xm, xh, yl, ym, q[1], sub_tol, depth + 1) +
         adaptive_cubature(prof, xl, xm, ym, yh, q[2], sub_tol, depth + 1) +
         adaptive_cubature(prof, xm, xh, ym, yh, q[3], sub_tol, depth + 1);
}

/**
 * @brief Compute a population's normalised light in every pixel.
 *
 * @param prof: The population's profile.
 * @param res: The pixel size.
 * @param npix_x, npix_y: The number of pixels along x and y.
 * @param tol: The relative tolerance.
 * @param nthreads: The number of threads.
 * @param centres: Scratch space for the centre samples,
 *                 (npix_y + 4) * (npix_x + 4).
 * @param weights: The output normalised light in each pixel (row major in
 *                 y, x).
 */
static void population_weights(const Profile &prof, double res, int npix_x,
                               int npix_y, double tol, int nthreads,
                               std::vector<double> &centres,
                               std::vector<double> &weights) {
  std::fill(weights.begin(), weights.end(), 0.0);
  const double norm =
      prof.total * (prof.enclosed(prof.rhi) - prof.enclosed(prof.rlo));
  if (!(norm > 0.0)) return;

  /* A point source puts all its light in the pixel containing it. */
  if (prof.type == POINT) {
    const int ix = (int)std::floor(prof.x0 / res + 0.5 * npix_x);
    const int iy = (int)std::floor(prof.y0 / res + 0.5 * npix_y);
    if (ix >= 0 && ix < npix_x && iy >= 0 && iy < npix_y) {
      weights[(size_t)iy * npix_x + ix] = 1.0;
    }
    return;
  }

  const int mx = npix_x + 4;
  const double area = res * res;
  const double x_first = -res * (npix_x - 1) / 2.0;
  const double y_first = -res * (npix_y - 1) / 2.0;
  const double core = CORE_PIXELS * res;
  const double reach = 0.5 * M_SQRT2 * res / prof.q;
  const double floor_value = tol * 1e-6 * prof.total / area;

  /* Sample the profile at every pixel centre, with a margin of two pixels
   * for the stencils below. */
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int j = 0; j < npix_y + 4; j++) {
    const double y = y_first + (j - 2) * res;
    for (int i = 0; i < mx; i++) {
      centres[(size_t)j * mx + i] = prof.value(x_first + (i - 2) * res, y);
    }
  }

  /* The discrete Laplacian (times res^2) of the centre samples. */
  auto laplacian = [&](size_t c) {
    return centres[c - 1] + centres[c + 1] + centres[c - mx] +
           centres[c + mx] - 4.0 * centres[c];
  };

  /* Integrate each pixel. */
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, 4)
  for (int j = 0; j < npix_y; j++) {
    const double y = y_first + j * res;
    const double yl = y - 0.5 * res, yh = y + 0.5 * res;
    for (int i = 0; i < npix_x; i++) {
      const double x = x_first + i * res;
      const double xl = x - 0.5 * res, xh = x + 0.5 * res;
      const double r = prof.radius(x, y);
      double light;

      /* Pixels straddling an annulus edge or near a cusp's centre are
       * integrated in shells. */
      const bool at_edge =
          (prof.rlo > 0.0 && std::fabs(r - prof.rlo) < reach) ||
          (prof.rhi < UNBOUNDED && std::fabs(r - prof.rhi) < reach);
      if (at_edge || (prof.cusp && r < core + reach)) {
        light = integrate_pixel_shells(prof, xl, xh, yl, yh, res);
      } else {
        /* Otherwise use the centre sample if the midpoint rule's error
         * (res^2 / 24 times the Laplacian, estimated from the neighbouring
         * centre samples) is small enough. Failing that use the centre
         * sample corrected by that error (exact for a quadratic profile)
         * if its own error (estimated from the Laplacian of the Laplacian)
         * is small enough, or else adaptive cubature. */
        const size_t c = (size_t)(j + 2) * mx + (i + 2);
        const bool resolved = prof.local_scale(r) >= RESOLVED_PIXELS * res;
        const double f = centres[c];
        const double lap = laplacian(c);
        const double error = std::fabs(lap) / 24.0;
        const double corrected = f + lap / 24.0;
        const double error4 =
            std::fabs(laplacian(c - 1) + laplacian(c + 1) + laplacian(c - mx) +
                      laplacian(c + mx) - 4.0 * lap) /
            300.0;
        /* (the fourth order estimate guards against the Laplacian
         * vanishing by chance, e.g. at an inflection of a profile
         * narrower than a pixel) */
        if (std::max(error, error4) <= floor_value ||
            (resolved && std::max(error, error4) <= tol * std::fabs(f))) {
          light = f * area;
        } else if (resolved && error4 <= tol * std::fabs(corrected)) {
          light = corrected * area;
        } else {
          const double coarse = gauss_legendre(prof, xl, xh, yl, yh);
          light = adaptive_cubature(
              prof, xl, xh, yl, yh, coarse,
              std::max(tol * std::fabs(coarse), floor_value * area), 0);
        }
      }
      weights[(size_t)j * npix_x + i] = light / norm;
    }
  }
}

/**
 * @brief Make the images of a set of parametric populations.
 *
 * @param np_profiles: The (npop, NPARAMS) profile parameters (float64).
 * @param np_signals: The (npop, nsignal) signal of each population (float32
 *                    or float64, which sets the image dtype).
 * @param res: The pixel size.
 * @param npix_x, npix_y: The number of pixels along x and y.
 * @param tol: The relative tolerance of each pixel's integral.
 * @param nthreads: The number of threads.
 * @return The (npix_y, npix_x, nsignal) images.
 */
PyObject *make_parametric_images(PyObject *self, PyObject *args) {
  (void)self;
  tic("make_parametric_images");

  PyArrayObject *np_profiles, *np_signals;
  double res, tol;
  int npix_x, npix_y, nthreads;
  if (!PyArg_ParseTuple(args, "OOdiidi", &np_profiles, &np_signals, &res,
                        &npix_x, &npix_y, &tol, &nthreads)) {
    return NULL;
  }
  if (PyArray_TYPE(np_profiles) != NPY_FLOAT64 ||
      !is_c_contiguous(np_profiles, "profiles") ||
      PyArray_NDIM(np_profiles) != 2 ||
      PyArray_DIM(np_profiles, 1) != NPARAMS) {
    PyErr_SetString(PyExc_ValueError,
                    "profiles must be a C contiguous float64 array of shape "
                    "(npop, NPARAMS).");
    return NULL;
  }
  if (!is_float32_or_float64(np_signals, "signals") ||
      !is_c_contiguous(np_signals, "signals") ||
      PyArray_NDIM(np_signals) != 2 ||
      PyArray_DIM(np_signals, 0) != PyArray_DIM(np_profiles, 0)) {
    PyErr_SetString(PyExc_ValueError,
                    "signals must be a C contiguous float array of shape "
                    "(npop, nsignal).");
    return NULL;
  }
  if (npix_x <= 0 || npix_y <= 0 || !(res > 0.0)) {
    PyErr_SetString(PyExc_ValueError,
                    "The image needs a positive resolution and size.");
    return NULL;
  }

  const int npop = (int)PyArray_DIM(np_profiles, 0);
  const int nsig = (int)PyArray_DIM(np_signals, 1);
  const double *params = data_ptr<const double>(np_profiles);
  const int sig_typenum = PyArray_TYPE(np_signals);
  const size_t npix = (size_t)npix_x * npix_y;

  npy_intp dims[3] = {npix_y, npix_x, nsig};
  PyArrayObject *np_img =
      (PyArrayObject *)PyArray_ZEROS(3, dims, sig_typenum, 0);
  if (np_img == NULL) return NULL;

  dispatch_float(sig_typenum, [&](auto s) {
    using SigReal = decltype(s);
    const SigReal *signals = data_ptr<const SigReal>(np_signals);
    SigReal *img = data_ptr<SigReal>(np_img);

    std::vector<double> centres((size_t)(npix_x + 4) * (npix_y + 4));
    std::vector<double> weights(npix);

    Py_BEGIN_ALLOW_THREADS;
    for (int p = 0; p < npop; p++) {
      const Profile prof(params + (size_t)p * NPARAMS);
      population_weights(prof, res, npix_x, npix_y, tol, nthreads, centres,
                         weights);
      const SigReal *sig = signals + (size_t)p * nsig;

      /* Add this population's light times its signal to every image. */
#pragma omp parallel for num_threads(nthreads) schedule(static)
      for (long long k = 0; k < (long long)npix; k++) {
        const double w = weights[k];
        if (w == 0.0) continue;
        SigReal *out = img + (size_t)k * nsig;
        for (int s_i = 0; s_i < nsig; s_i++) {
          out[s_i] += static_cast<SigReal>(w * sig[s_i]);
        }
      }
    }
    Py_END_ALLOW_THREADS;
  });

  toc("make_parametric_images");
  return Py_BuildValue("N", np_img);
}

static PyMethodDef ParametricImageMethods[] = {
    {"make_parametric_images", (PyCFunction)make_parametric_images,
     METH_VARARGS, "Make the images of a set of parametric populations."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "parametric_image",
    "A module to make images of parametric populations",
    -1,
    ParametricImageMethods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit_parametric_image(void) {
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL) return NULL;
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    Py_DECREF(m);
    return NULL;
  }
#ifdef ATOMIC_TIMING
  if (import_toc_capsule() < 0) {
    Py_DECREF(m);
    return NULL;
  }
#endif
  return m;
}
