// SGD-based palette optimizer
// Ported from palettequant.c (tiled palette quantization)

#include "Optimizer.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fmt/core.h>

namespace sfc {

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr unsigned MAX_PALS = 16;
static constexpr unsigned MAX_COLS = 256;


// ── Floating-point color for SGD ──────────────────────────────────────────────

struct OColor {
  double r, g, b;
};

static inline OColor oc(double r, double g, double b) { return {r, g, b}; }
static inline bool oc_eq(OColor a, OColor b) { return a.r == b.r && a.g == b.g && a.b == b.b; }
static inline double js_round(double x) { return floor(x + 0.5); }

// ── CIE Lab color space ───────────────────────────────────────────────────────

static bool g_use_lab = false;
static double g_max_val = 31.0;

// Lab cache for mode-reduced colors (up to 32768 entries for 5-bit modes)
static OColor lab_cache[32768];
static bool lab_computed[32768];
static bool lab_cache_initialized = false;

static void reset_lab_cache() {
  if (lab_cache_initialized) {
    memset(lab_computed, 0, sizeof(lab_computed));
    lab_cache_initialized = false;
  }
}

static OColor rgb_to_lab(OColor rgb, double max_val) {
  double r = rgb.r / max_val;
  double g = rgb.g / max_val;
  double b = rgb.b / max_val;

  // sRGB → linear
  r = (r > 0.04045) ? pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
  g = (g > 0.04045) ? pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
  b = (b > 0.04045) ? pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

  // Linear RGB → XYZ (D65)
  double x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) * 100.0;
  double y = (r * 0.2126729 + g * 0.7151522 + b * 0.0721750) * 100.0;
  double z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) * 100.0;

  // XYZ → Lab (illuminant D65)
  auto f = [](double t) { return t > 0.008856 ? cbrt(t) : 7.787 * t + 16.0/116.0; };
  double fx = f(x / 95.047);
  double fy = f(y / 100.000);
  double fz = f(z / 108.883);

  return {116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)};
}

static inline OColor cached_rgb_to_lab(OColor rgb) {
  int ri = (int)js_round(rgb.r);
  int gi = (int)js_round(rgb.g);
  int bi = (int)js_round(rgb.b);
  if (ri < 0) ri = 0; if (gi < 0) gi = 0; if (bi < 0) bi = 0;
  int max_i = (int)g_max_val;
  if (ri > max_i) ri = max_i;
  if (gi > max_i) gi = max_i;
  if (bi > max_i) bi = max_i;

  // For modes with <= 5 bits per channel (max 32768 entries), use static cache
  if (max_i <= 31) {
    int idx = (ri << 10) | (gi << 5) | bi;
    if (!lab_computed[idx]) {
      lab_cache[idx] = rgb_to_lab(oc(ri, gi, bi), g_max_val);
      lab_computed[idx] = true;
      lab_cache_initialized = true;
    }
    return lab_cache[idx];
  }
  // For larger color spaces, convert on the fly
  return rgb_to_lab(rgb, g_max_val);
}

static inline double oc_dist(OColor a, OColor b) {
  if (!g_use_lab) {
    double dr = a.r - b.r, dg = a.g - b.g, db = a.b - b.b;
    return 2 * dr * dr + 4 * dg * dg + db * db;
  }
  OColor la = cached_rgb_to_lab(a);
  OColor lb = cached_rgb_to_lab(b);
  double dL = la.r - lb.r, da = la.g - lb.g, db = la.b - lb.b;
  return dL * dL + da * da + db * db;
}
static inline void oc_add(OColor& a, OColor b) { a.r += b.r; a.g += b.g; a.b += b.b; }
static inline void oc_sub(OColor& a, OColor b) { a.r -= b.r; a.g -= b.g; a.b -= b.b; }
static inline void oc_scl(OColor& c, double f) { c.r *= f; c.g *= f; c.b *= f; }
static inline void oc_clamp(OColor& c, double lo, double hi) {
  if (c.r < lo) c.r = lo; else if (c.r > hi) c.r = hi;
  if (c.g < lo) c.g = lo; else if (c.g > hi) c.g = hi;
  if (c.b < lo) c.b = lo; else if (c.b > hi) c.b = hi;
}
static inline void oc_move(OColor& c, OColor p, double a) {
  double inv = 1.0 - a;
  c.r = inv * c.r + a * p.r;
  c.g = inv * c.g + a * p.g;
  c.b = inv * c.b + a * p.b;
}

static inline double oc_brightness(OColor c) {
  return 0.299 * c.r * c.r + 0.587 * c.g * c.g + 0.114 * c.b * c.b;
}

// ── Linear/sRGB conversion (for dithering error diffusion) ───────────────────

static inline void oc_to_linear(OColor& c) { c.r *= c.r; c.g *= c.g; c.b *= c.b; }
static inline void oc_to_srgb(OColor& c) { c.r = sqrt(c.r); c.g = sqrt(c.g); c.b = sqrt(c.b); }

// ── Dither patterns (2x2 ordered dither, from palettequant) ──────────────────

static const int DITHER_PATTERNS[6][2][2] = {
    {{0,2},{3,1}}, {{0,3},{1,2}}, {{0,1},{3,2}},  // 4-candidate
    {{0,1},{1,0}}, {{0,1},{0,1}}, {{0,0},{1,1}}   // 2-candidate
};

static int dither_pattern_candidates(DitherPattern p) {
  return (int)p < 3 ? 4 : 2;
}

static const int (*dither_pattern_matrix(DitherPattern p))[2] {
  return DITHER_PATTERNS[(int)p];
}

static inline OColor oc_round(OColor c) {
  return {js_round(c.r), js_round(c.g), js_round(c.b)};
}

// ── PRNG (Mulberry32, identical to palettequant) ──────────────────────────────

struct Mulberry32 {
  uint32_t a;
};

static inline double m32_next(Mulberry32& r) {
  r.a += 0x6D2B79F5;
  uint32_t a = r.a;
  uint32_t t = (a ^ (a >> 15)) * (1 | a);
  t = (t + ((t ^ (t >> 7)) * (61 | t))) ^ t;
  return (double)((t ^ (t >> 14)) >> 0) / 4294967296.0;
}

// ── Random shuffle (Fisher-Yates, identical to palettequant) ──────────────────

struct RandShuf {
  std::vector<int> vals;
  int n, cur;
  Mulberry32* rng;
};

static void rs_init(RandShuf& s, int n, Mulberry32* rng) {
  s.vals.resize(n);
  s.n = n; s.cur = n - 1; s.rng = rng;
  for (int i = 0; i < n; i++) s.vals[i] = i;
}

static void rs_shuffle(RandShuf& s) {
  for (int i = 0; i < s.n; i++) {
    int j = i + (int)(m32_next(*s.rng) * (s.n - i));
    std::swap(s.vals[i], s.vals[j]);
  }
}

static int rs_next(RandShuf& s) {
  if (++s.cur >= s.n) { rs_shuffle(s); s.cur = 0; }
  return s.vals[s.cur];
}

// ── Tile data ─────────────────────────────────────────────────────────────────

struct OTile {
  std::vector<OColor> colors;
  std::vector<double> counts;
  int num_colors = 0;
  std::vector<int> px, py;
  std::vector<OColor> pcols;
  int num_pixels = 0;
};

struct OPixel {
  int tile_idx;
  OColor color;
  int x, y;
};

// ── Utility ───────────────────────────────────────────────────────────────────

static int max_idx(const double* v, int n) {
  int m = 0;
  for (int i = 1; i < n; i++) if (v[i] > v[m]) m = i;
  return m;
}

static int min_idx(const double* v, int n) {
  int m = 0;
  for (int i = 1; i < n; i++) if (v[i] < v[m]) m = i;
  return m;
}

static int iter_count(double it) {
  int i = (int)it;
  return (it != (double)i) ? (int)ceil(it) : i;
}

// ── Closest color / palette ───────────────────────────────────────────────────

struct CRes { int idx; double dist; };

static CRes closest_color(const OColor* pal, int nc, OColor c) {
  int mi = nc - 1;
  double md = oc_dist(pal[mi], c);
  for (int i = nc - 2; i >= 0; i--) {
    double d = oc_dist(pal[i], c);
    if (d < md) { mi = i; md = d; }
  }
  return {mi, md};
}

static double pal_dist(const OColor* pal, int nc, const OTile& t) {
  double tot = 0;
  for (int i = 0; i < t.num_colors; i++) {
    CRes r = closest_color(pal, nc, t.colors[i]);
    tot += t.counts[i] * r.dist;
  }
  return tot;
}

static int closest_pal_idx(const OColor pals[][MAX_COLS], int np, int nc, const OTile& t) {
  if (np == 1) return 0;
  int best = 0;
  double bd = pal_dist(pals[0], nc, t);
  for (int i = 1; i < np; i++) {
    double d = pal_dist(pals[i], nc, t);
    if (d < bd) { best = i; bd = d; }
  }
  return best;
}

struct PRes { int idx; double dist; };

static PRes closest_pal_dist(const OColor pals[][MAX_COLS], int np, int nc, const OTile& t) {
  PRes r;
  r.idx = 0; r.dist = pal_dist(pals[0], nc, t);
  for (int i = 1; i < np; i++) {
    double d = pal_dist(pals[i], nc, t);
    if (d < r.dist) { r.idx = i; r.dist = d; }
  }
  return r;
}

// ── Dithered color/palette selection ──────────────────────────────────────────

struct DRes { int idx; double dist; OColor target; };

static DRes closest_color_dither(const OColor* pal, int nc, OColor pcol, int px, int py,
    double dither_weight, const int dp[2][2], int dpx, double max_val) {
  OColor error = {0, 0, 0};
  OColor lp = pcol; oc_to_linear(lp);

  struct Cand { int ci; double cd; double br; };
  Cand cands[4];
  OColor c = {};

  for (int i = 0; i < dpx; i++) {
    c = lp;
    OColor err = error; oc_scl(err, dither_weight);
    oc_add(c, err);
    oc_clamp(c, 0, max_val * max_val); // linear space: max_val^2
    oc_to_srgb(c);
    CRes cr = closest_color(pal, nc, c);
    cands[i] = {cr.idx, cr.dist, oc_brightness(pal[cr.idx])};
    OColor red = oc_round(pal[cr.idx]);
    oc_to_linear(red);
    oc_add(error, lp);
    oc_sub(error, red);
  }

  // Sort candidates by brightness
  for (int i = 0; i < dpx - 1; i++)
    for (int j = i + 1; j < dpx; j++)
      if (cands[i].br > cands[j].br) std::swap(cands[i], cands[j]);

  int idx = dp[px & 1][py & 1];
  return {cands[idx].ci, cands[idx].cd, c};
}

static double pal_dist_d(const OColor* pal, int nc, const OTile& t,
    double dw, const int dp[2][2], int dpx, double max_val) {
  double tot = 0;
  for (int i = 0; i < t.num_pixels; i++) {
    DRes r = closest_color_dither(pal, nc, t.pcols[i], t.px[i], t.py[i], dw, dp, dpx, max_val);
    tot += r.dist;
  }
  return tot;
}

static int closest_pal_idx_d(const OColor pals[][MAX_COLS], int np, int nc, const OTile& t,
    double dw, const int dp[2][2], int dpx, double max_val) {
  if (np == 1) return 0;
  int best = 0;
  double bd = pal_dist_d(pals[0], nc, t, dw, dp, dpx, max_val);
  for (int i = 1; i < np; i++) {
    double d = pal_dist_d(pals[i], nc, t, dw, dp, dpx, max_val);
    if (d < bd) { best = i; bd = d; }
  }
  return best;
}

static PRes closest_pal_dist_d(const OColor pals[][MAX_COLS], int np, int nc, const OTile& t,
    double dw, const int dp[2][2], int dpx, double max_val) {
  PRes r;
  r.idx = 0; r.dist = pal_dist_d(pals[0], nc, t, dw, dp, dpx, max_val);
  for (int i = 1; i < np; i++) {
    double d = pal_dist_d(pals[i], nc, t, dw, dp, dpx, max_val);
    if (d < r.dist) { r.idx = i; r.dist = d; }
  }
  return r;
}

// ── MSE ───────────────────────────────────────────────────────────────────────

static double mse(const OColor pals[][MAX_COLS], int np, int nc,
                  const std::vector<OTile>& tiles) {
  double tot = 0;
  int cnt = 0;
  for (const auto& t : tiles) {
    int pi = closest_pal_idx(pals, np, nc, t);
    for (int i = 0; i < t.num_colors; i++) {
      CRes r = closest_color(pals[pi], nc, t.colors[i]);
      tot += r.dist * t.counts[i];
      cnt += (int)t.counts[i];
    }
  }
  return cnt > 0 ? tot / cnt : 0;
}

// ── Palette manipulation ──────────────────────────────────────────────────────

static void reduce_pals(OColor out[][MAX_COLS], const OColor in[][MAX_COLS], int np, int nc) {
  for (int p = 0; p < np; p++)
    for (int c = 0; c < nc; c++)
      out[p][c] = oc_round(in[p][c]);
}

static void move_closer(OColor pals[][MAX_COLS], int np, int nc,
                        const OPixel& pixel, const std::vector<OTile>& tiles, double alpha,
                        int shared_color_idx,
                        bool use_dither = false, double dw = 0, const int (*dp)[2] = nullptr,
                        int dpx = 0, double max_val = 0) {
  const OTile& tile = tiles[pixel.tile_idx];
  int pi, ci;
  OColor target;
  if (use_dither) {
    pi = closest_pal_idx_d(pals, np, nc, tile, dw, dp, dpx, max_val);
    DRes dr = closest_color_dither(pals[pi], nc, pixel.color, pixel.x, pixel.y, dw, dp, dpx, max_val);
    ci = dr.idx; target = dr.target;
  } else {
    pi = closest_pal_idx(pals, np, nc, tile);
    CRes r = closest_color(pals[pi], nc, pixel.color);
    ci = r.idx; target = pixel.color;
  }
  if (ci != shared_color_idx)
    oc_move(pals[pi][ci], target, alpha);
}

// ── Phase 1: Initialize palettes with 1 color ────────────────────────────────

static int cq1(OColor pals[][MAX_COLS], int* out_nc,
               const std::vector<OTile>& tiles, const std::vector<OPixel>& pixels,
               RandShuf& rs, unsigned num_palettes, int shared_color_idx,
               double fraction_of_pixels, OColor col0_val,
               bool slow_dither = false, double dw = 0, const int (*dp)[2] = nullptr,
               int dpx = 0, double max_val = 0) {
  int npx = (int)pixels.size();
  double iterations = fraction_of_pixels * npx;
  double alpha = 0.3;
  if (slow_dither) { iterations /= 5; alpha = 0.1; }

  OColor avg = {0, 0, 0};
  for (const auto& p : pixels) oc_add(avg, p.color);
  oc_scl(avg, 1.0 / npx);

  int np = 1, nc = 1;
  pals[0][0] = avg;
  if (shared_color_idx >= 0) {
    pals[0][1] = avg;
    pals[0][0] = col0_val;
    nc = 2;
  }

  int si = 0;
  for (unsigned num_p = 2; num_p <= num_palettes; num_p++) {
    memcpy(pals[num_p - 1], pals[si], nc * sizeof(OColor));
    np = num_p;
    int it = iter_count(iterations);
    for (int i = 0; i < it; i++)
      move_closer(pals, np, nc, pixels[rs_next(rs)], tiles, alpha, shared_color_idx,
                  slow_dither, dw, dp, dpx, max_val);

    double pd[MAX_PALS] = {};
    for (const auto& t : tiles) {
      PRes r = closest_pal_dist(pals, np, nc, t);
      pd[r.idx] += r.dist;
    }
    si = max_idx(pd, np);
  }
  *out_nc = nc;
  return np;
}

// ── Phase 2: Expand palettes by one color ─────────────────────────────────────

static void expand1(OColor pals[][MAX_COLS], int np, int* nc_p,
                    const std::vector<OTile>& tiles, const std::vector<OPixel>& pixels,
                    RandShuf& rs, double fraction_of_pixels, int shared_color_idx,
                    bool slow_dither = false, double dw = 0, const int (*dp)[2] = nullptr,
                    int dpx = 0, double max_val = 0) {
  int nc = *nc_p;
  int npx = (int)pixels.size();
  double iterations = fraction_of_pixels * npx;
  double alpha = 0.3;
  if (slow_dither) { iterations /= 5; alpha = 0.1; }
  int ncn = nc + 1;
  int si[MAX_PALS] = {};

  if (ncn > 2) {
    double tcd[MAX_PALS][MAX_COLS] = {};
    for (const auto& t : tiles) {
      int pi = closest_pal_idx(pals, np, nc, t);
      for (int i = 0; i < t.num_colors; i++) {
        CRes r = closest_color(pals[pi], nc, t.colors[i]);
        tcd[pi][r.idx] += t.counts[i] * r.dist;
      }
    }
    for (int p = 0; p < np; p++)
      si[p] = max_idx(tcd[p], ncn);
  }

  for (int p = 0; p < np; p++)
    pals[p][nc] = pals[p][si[p]];
  *nc_p = ncn;
  nc = ncn;

  int it = iter_count(iterations);
  for (int i = 0; i < it; i++)
    move_closer(pals, np, nc, pixels[rs_next(rs)], tiles, alpha, shared_color_idx,
                slow_dither, dw, dp, dpx, max_val);
}

// ── Phase 3: Replace weakest colors ───────────────────────────────────────────

static void replace_weak(OColor out[][MAX_COLS], const OColor in[][MAX_COLS],
                        int np, int nc, const std::vector<OTile>& tiles,
                        double mcf, double mpf, int rep_pal, int shared_color_idx,
                        bool slow_dither = false, double dw = 0,
                        const int (*dp)[2] = nullptr, int dpx = 0, double max_val = 0) {
  int nt = (int)tiles.size();
  std::vector<int> cpi(nt, 0);
  double tpm[MAX_PALS] = {};
  double rpm[MAX_PALS] = {};
  int mxpi = 0, mnpi = 0;

  if (np > 1) {
    for (int j = 0; j < nt; j++) {
      PRes r = slow_dither
        ? closest_pal_dist_d(in, np, nc, tiles[j], dw, dp, dpx, max_val)
        : closest_pal_dist(in, np, nc, tiles[j]);
      tpm[r.idx] += r.dist;
      cpi[j] = r.idx;

      OColor rem[MAX_PALS][MAX_COLS];
      int rn = 0;
      for (int p = 0; p < np; p++)
        if (p != r.idx) { memcpy(rem[rn], in[p], nc * sizeof(OColor)); rn++; }
      if (rn > 0) {
        PRes r2 = slow_dither
          ? closest_pal_dist_d(rem, rn, nc, tiles[j], dw, dp, dpx, max_val)
          : closest_pal_dist(rem, rn, nc, tiles[j]);
        rpm[r.idx] += r2.dist;
      }
    }
    mxpi = max_idx(tpm, np);
    mnpi = min_idx(rpm, np);
  }

  if (nc > 1) {
    double tcm[MAX_PALS][MAX_COLS] = {};
    double scm[MAX_PALS][MAX_COLS] = {};
    for (int j = 0; j < nt; j++) {
      int mpi = cpi[j];
      const OColor* pal = in[mpi];
      if (slow_dither) {
        // SLOW: evaluate each pixel individually with dithering
        for (int p = 0; p < tiles[j].num_pixels; p++) {
          DRes r = closest_color_dither(pal, nc, tiles[j].pcols[p],
              tiles[j].px[p], tiles[j].py[p], dw, dp, dpx, max_val);
          tcm[mpi][r.idx] += r.dist;
          OColor rm[MAX_COLS]; int rn = 0;
          for (int k = 0; k < nc; k++)
            if (k != r.idx) rm[rn++] = pal[k];
          DRes r2 = closest_color_dither(rm, rn, tiles[j].pcols[p],
              tiles[j].px[p], tiles[j].py[p], dw, dp, dpx, max_val);
          scm[mpi][r.idx] += r2.dist;
        }
      } else {
        for (int i = 0; i < tiles[j].num_colors; i++) {
          CRes r = closest_color(pal, nc, tiles[j].colors[i]);
          tcm[mpi][r.idx] += r.dist * tiles[j].counts[i];
          OColor rm[MAX_COLS]; int rn = 0;
          for (int k = 0; k < nc; k++)
            if (k != r.idx) rm[rn++] = pal[k];
          CRes r2 = closest_color(rm, rn, tiles[j].colors[i]);
          scm[mpi][r.idx] += r2.dist * tiles[j].counts[i];
        }
      }
    }

    for (int p = 0; p < np; p++) {
      int mxc = max_idx(tcm[p], nc);
      int mnc = min_idx(scm[p], nc);
      int rep = (mnc != mxc && mnc != shared_color_idx && scm[p][mnc] < mcf * tcm[p][mxc]);
      for (int c = 0; c < nc; c++)
        out[p][c] = (c == mnc && rep) ? in[p][mxc] : in[p][c];
    }
  } else {
    for (int p = 0; p < np; p++)
      for (int c = 0; c < nc; c++)
        out[p][c] = in[p][c];
  }

  if (rep_pal && mnpi != mxpi && rpm[mnpi] < mpf * tpm[mxpi])
    memcpy(out[mnpi], out[mxpi], nc * sizeof(OColor));
}

// ── K-means ───────────────────────────────────────────────────────────────────

static void kmeans(OColor out[][MAX_COLS], const OColor in[][MAX_COLS],
                  int np, int nc, const std::vector<OTile>& tiles, int shared_color_idx) {
  int counts[MAX_PALS][MAX_COLS] = {};
  OColor sums[MAX_PALS][MAX_COLS];
  for (int p = 0; p < np; p++)
    for (int c = 0; c < nc; c++)
      sums[p][c] = {0, 0, 0};

  for (const auto& t : tiles) {
    int pi = closest_pal_idx(in, np, nc, t);
    for (int i = 0; i < t.num_colors; i++) {
      CRes r = closest_color(in[pi], nc, t.colors[i]);
      counts[pi][r.idx] += (int)t.counts[i];
      OColor w = t.colors[i];
      oc_scl(w, t.counts[i]);
      oc_add(sums[pi][r.idx], w);
    }
  }

  for (int p = 0; p < np; p++)
    for (int c = 0; c < nc; c++) {
      if (counts[p][c] == 0 || c == shared_color_idx)
        out[p][c] = in[p][c];
      else {
        out[p][c] = sums[p][c];
        oc_scl(out[p][c], 1.0 / counts[p][c]);
      }
    }
}

// ── Greedy palette selection ─────────────────────────────────────────────────

static void greedy_palette_select(
    const std::vector<OColor>& colors,
    const std::vector<double>& counts,
    OColor* out_pal, int n_out,
    int start_idx  // 0 or 1 if col0 is shared
) {
  int nc = (int)colors.size();
  if (nc == 0) return;

  // Start with most frequent color
  int best_init = 0;
  for (int i = 1; i < nc; i++)
    if (counts[i] > counts[best_init]) best_init = i;

  out_pal[start_idx] = colors[best_init];
  std::vector<double> best_dist(nc);
  for (int i = 0; i < nc; i++)
    best_dist[i] = oc_dist(colors[i], out_pal[start_idx]);

  std::vector<bool> used(nc, false);
  used[best_init] = true;

  for (int slot = start_idx + 1; slot < n_out; slot++) {
    int best_c = -1;
    double best_reduction = -1;

    for (int c = 0; c < nc; c++) {
      if (used[c]) continue;
      double reduction = 0;
      for (int other = 0; other < nc; other++) {
        double d = oc_dist(colors[c], colors[other]);
        if (d < best_dist[other])
          reduction += (best_dist[other] - d) * counts[other];
      }
      if (reduction > best_reduction) {
        best_reduction = reduction;
        best_c = c;
      }
    }

    if (best_c < 0 || best_reduction <= 0) break;
    out_pal[slot] = colors[best_c];
    used[best_c] = true;

    for (int i = 0; i < nc; i++) {
      double d = oc_dist(colors[i], out_pal[slot]);
      if (d < best_dist[i]) best_dist[i] = d;
    }
  }
}

// ── Sort palettes (TSP-like, identical to palettequant) ──────────────────────

static void rev_sub(int* a, int l, int r) {
  double mid = (l + r) / 2.0;
  while (l < mid) { std::swap(a[l], a[r]); l++; r--; }
}

static void sort_palettes(OColor out[][MAX_COLS], const OColor in[][MAX_COLS],
                         int np, int nc, int si, Mulberry32& rng) {
  if (nc == 2 && si == 1) {
    for (int p = 0; p < np; p++) memcpy(out[p], in[p], nc * sizeof(OColor));
    return;
  }

  int pds = np + 2;
  std::vector<double> pd(pds * pds, 0);
  auto PD = [&](int i, int j) -> double& { return pd[i * pds + j]; };

  std::vector<int> ci(np * np * nc);
  auto CI = [&](int a, int b, int k) -> int& { return ci[(a * np + b) * nc + k]; };
  for (int a = 0; a < np; a++)
    for (int b = 0; b < np; b++)
      for (int k = 0; k < nc; k++)
        CI(a, b, k) = k;

  for (int p1 = 0; p1 < np - 1; p1++)
    for (int p2 = p1 + 1; p2 < np; p2++) {
      for (int it = 0; it < 2000; it++) {
        int i1 = si + (int)(m32_next(rng) * (nc - si - 1));
        int i2 = i1 + 1 + (int)(m32_next(rng) * (nc - i1 - 1));
        if (m32_next(rng) < 0.5) std::swap(i1, i2);
        double st = oc_dist(in[p1][i1], in[p2][CI(p1, p2, i1)]) + oc_dist(in[p1][i2], in[p2][CI(p1, p2, i2)]);
        double sw = oc_dist(in[p1][i1], in[p2][CI(p1, p2, i2)]) + oc_dist(in[p1][i2], in[p2][CI(p1, p2, i1)]);
        if (sw < st) std::swap(CI(p1, p2, i1), CI(p1, p2, i2));
      }
      double tot = 0;
      for (int k = 0; k < nc; k++) tot += oc_dist(in[p1][k], in[p2][CI(p1, p2, k)]);
      PD(p1 + 1, p2 + 1) = tot;
      PD(p2 + 1, p1 + 1) = tot;
    }

  // Reverse mapping
  for (int p1 = 1; p1 < np; p1++)
    for (int p2 = 0; p2 < p1; p2++)
      for (int i = 0; i < nc; i++)
        for (int k = 0; k < nc; k++)
          if (CI(p2, p1, k) == i) { CI(p1, p2, i) = k; break; }

  std::vector<int> pidx(np + 2);
  for (int i = 0; i < np + 2; i++) pidx[i] = i;
  if (np > 2)
    for (int it = 0; it < 100000; it++) {
      int x1 = (int)(m32_next(rng) * np); if (x1 < 1) x1 = 1;
      int x2 = x1 + 1 + (int)(m32_next(rng) * np); if (x2 > np) x2 = np;
      double st = PD(pidx[x1 - 1], pidx[x1]) + PD(pidx[x2], pidx[x2 + 1]);
      double sw = PD(pidx[x1 - 1], pidx[x2]) + PD(pidx[x1], pidx[x2 + 1]);
      if (sw < st) rev_sub(pidx.data(), x1, x2);
    }

  // Color order in first palette
  std::vector<int> p1i(nc + 2);
  for (int i = 0; i < nc + 2; i++) p1i[i] = i;
  int pal1 = pidx[1] - 1;
  std::vector<double> p1d((nc + 2) * (nc + 2), 0);
  auto P1D = [&](int i, int j) -> double& { return p1d[i * (nc + 2) + j]; };
  for (int i = 1; i <= nc; i++)
    for (int j = 1; j <= nc; j++)
      P1D(i, j) = oc_dist(in[pal1][i - 1], in[pal1][j - 1]);
  if (nc > 2)
    for (int it = 0; it < 100000; it++) {
      int x1 = (int)(m32_next(rng) * nc); if (x1 < 1 + si) x1 = 1 + si;
      int x2 = x1 + 1 + (int)(m32_next(rng) * nc); if (x2 > nc) x2 = nc;
      double st = P1D(p1i[x1 - 1], p1i[x1]) + P1D(p1i[x2], p1i[x2 + 1]);
      double sw = P1D(p1i[x1 - 1], p1i[x2]) + P1D(p1i[x1], p1i[x2 + 1]);
      if (sw < st) rev_sub(p1i.data(), x1, x2);
    }

  // Build final mapping
  std::vector<int> pi(np * nc);
  auto PI = [&](int p, int c) -> int& { return pi[p * nc + c]; };
  for (int i = 0; i < nc; i++) PI(0, i) = p1i[i + 1] - 1;
  for (int i = 1; i < np; i++)
    for (int j = 0; j < nc; j++) {
      int pp1 = pidx[i] - 1, pp2 = pidx[i + 1] - 1;
      PI(i, j) = CI(pp1, pp2, PI(i - 1, j));
    }

  // Refine subsequent rows
  if (nc >= 4)
    for (int i = 1; i < np; i++) {
      int pp1 = pidx[i] - 1, pp2 = pidx[i + 1] - 1;
      for (int it = 0; it < 10000; it++) {
        int x1 = (int)(m32_next(rng) * nc); if (x1 < si) x1 = si;
        int x2 = (int)(m32_next(rng) * nc); if (x2 < si) x2 = si;
        if (x1 == x2) continue;
        int u1 = PI(i - 1, x1), c1 = PI(i, x1), l1 = x1 > 0 ? PI(i, x1 - 1) : -1, r1 = x1 + 1 < nc ? PI(i, x1 + 1) : nc;
        int u2 = PI(i - 1, x2), c2 = PI(i, x2), l2 = x2 > 0 ? PI(i, x2 - 1) : -1, r2 = x2 + 1 < nc ? PI(i, x2 + 1) : nc;
        double st = 2 * oc_dist(in[pp2][c1], in[pp1][u1]);
        if (l1 >= 0) st += oc_dist(in[pp2][c1], in[pp2][l1]);
        if (r1 < nc) st += oc_dist(in[pp2][c1], in[pp2][r1]);
        st += 2 * oc_dist(in[pp2][c2], in[pp1][u2]);
        if (l2 >= 0) st += oc_dist(in[pp2][c2], in[pp2][l2]);
        if (r2 < nc) st += oc_dist(in[pp2][c2], in[pp2][r2]);
        double sw = 2 * oc_dist(in[pp2][c2], in[pp1][u1]);
        if (l1 >= 0) sw += oc_dist(in[pp2][c2], in[pp2][l1]);
        if (r1 < nc) sw += oc_dist(in[pp2][c2], in[pp2][r1]);
        sw += 2 * oc_dist(in[pp2][c1], in[pp1][u2]);
        if (l2 >= 0) sw += oc_dist(in[pp2][c1], in[pp2][l2]);
        if (r2 < nc) sw += oc_dist(in[pp2][c1], in[pp2][r2]);
        if (sw < st) std::swap(PI(i, x1), PI(i, x2));
      }
    }

  for (int i = 0; i < np; i++) {
    int pp = pidx[i + 1] - 1;
    for (int j = 0; j < nc; j++)
      out[i][j] = in[pp][PI(i, j)];
  }
}

// ── Convert OColor to rgba_t in reduced color space ───────────────────────────

static rgba_t oc_to_rgba(OColor c) {
  int r = (int)js_round(c.r);
  int g = (int)js_round(c.g);
  int b = (int)js_round(c.b);
  if (r < 0) r = 0;
  if (g < 0) g = 0;
  if (b < 0) b = 0;
  return (rgba_t)r | ((rgba_t)g << 8) | ((rgba_t)b << 16) | 0xff000000u;
}

static OColor rgba_to_oc(rgba_t c) {
  rgba_color rc(c);
  return {(double)rc.r, (double)rc.g, (double)rc.b};
}

// ── Extract tiles from reduced image ──────────────────────────────────────────

static void extract_tiles(std::vector<OTile>& tiles,
                          const std::vector<rgba_t>& reduced_pixels,
                          unsigned width, unsigned height,
                          unsigned tile_w, unsigned tile_h) {
  for (unsigned sy = 0; sy < height; sy += tile_h) {
    for (unsigned sx = 0; sx < width; sx += tile_w) {
      OTile tile;
      unsigned ex = std::min(sx + tile_w, width);
      unsigned ey = std::min(sy + tile_h, height);

      for (unsigned y = sy; y < ey; y++) {
        for (unsigned x = sx; x < ex; x++) {
          rgba_t pixel = reduced_pixels[x + width * y];
          if (pixel == transparent_color) continue;

          OColor c = rgba_to_oc(pixel);
          tile.px.push_back(x);
          tile.py.push_back(y);
          tile.pcols.push_back(c);
          tile.num_pixels++;

          // Check if color already exists in tile
          int found = -1;
          for (int i = 0; i < tile.num_colors; i++)
            if (oc_eq(tile.colors[i], c)) { found = i; break; }
          if (found >= 0)
            tile.counts[found] += 1;
          else {
            tile.colors.push_back(c);
            tile.counts.push_back(1);
            tile.num_colors++;
          }
        }
      }

      if (tile.num_colors > 0)
        tiles.push_back(std::move(tile));
    }
  }
}

static void extract_pixels(std::vector<OPixel>& pixels, const std::vector<OTile>& tiles) {
  for (int t = 0; t < (int)tiles.size(); t++) {
    for (int p = 0; p < tiles[t].num_pixels; p++) {
      pixels.push_back({t, tiles[t].pcols[p], tiles[t].px[p], tiles[t].py[p]});
    }
  }
}

// ── Public API ────────────────────────────────────────────────────────────────

unsigned max_channel_value_for_mode(Mode mode) {
  switch (mode) {
  case Mode::snes: case Mode::snes_mode7:
  case Mode::gbc: case Mode::gba: case Mode::gba_affine:
    return 31;   // 5-bit
  case Mode::md: case Mode::pce: case Mode::pce_sprite:
    return 7;    // 3-bit
  case Mode::sms:
    return 3;    // 2-bit
  case Mode::wsc: case Mode::wsc_packed:
  case Mode::ngpc: case Mode::gg:
    return 15;   // 4-bit
  case Mode::gb:
    return 3;    // 2-bit grayscale
  case Mode::ws: case Mode::ngp:
    return 7;    // 3-bit grayscale
  default:
    return 255;
  }
}

OptimizedResult sgd_optimize(
    const channel_vec_t& image_data,
    unsigned width, unsigned height,
    unsigned tile_width, unsigned tile_height,
    unsigned num_palettes,
    unsigned colors_per_palette,
    Mode mode,
    double fraction_of_pixels,
    bool col0_is_shared,
    rgba_t col0_value,
    uint32_t seed,
    const DitherOptions& dither,
    const std::vector<rgba_vec_t>* initial_palettes,
    bool use_lab)
{
  if (seed == 0) seed = (uint32_t)time(nullptr);
  Mulberry32 rng = {seed};

  // Set Lab mode globals
  unsigned max_val = max_channel_value_for_mode(mode);
  g_use_lab = use_lab;
  g_max_val = (double)max_val;
  if (use_lab) reset_lab_cache();

  // Dither config
  bool use_dither = dither.mode != DitherMode::off;
  bool slow_dither = dither.mode == DitherMode::slow;
  const int (*dp)[2] = nullptr;
  int dpx = 0;
  double dw = 0;

  // Reduce image colors to mode-specific color space
  std::vector<rgba_t> reduced(width * height);
  for (unsigned i = 0; i < width * height; i++) {
    rgba_t pixel = image_data[i * 4]
                 | (image_data[i * 4 + 1] << 8)
                 | (image_data[i * 4 + 2] << 16)
                 | (image_data[i * 4 + 3] << 24);
    reduced[i] = reduce_color(pixel, mode);
  }

  // Extract tiles
  std::vector<OTile> tiles;
  extract_tiles(tiles, reduced, width, height, tile_width, tile_height);
  if (tiles.empty()) {
    return {{}, {}};
  }

  // Extract pixels
  std::vector<OPixel> pixels;
  extract_pixels(pixels, tiles);
  int npx = (int)pixels.size();
  if (npx == 0) {
    return {{}, {}};
  }

  RandShuf rs;
  rs_init(rs, npx, &rng);

  double iterations = fraction_of_pixels * npx;
  double alpha = 0.3;
  double final_alpha = 0.05;

  if (use_dither) {
    dp = dither_pattern_matrix(dither.pattern);
    dpx = dither_pattern_candidates(dither.pattern);
    dw = dither.weight;
  }
  if (slow_dither) final_alpha = 0.02;

  int shared_color_idx = col0_is_shared ? 0 : -1;
  OColor col0_oc = rgba_to_oc(col0_value);

  OColor pals[MAX_PALS][MAX_COLS];
  int np, nc;

  if (initial_palettes && !initial_palettes->empty()) {
    // Use provided initial palettes (skip Phase 1-2)
    np = (int)initial_palettes->size();
    nc = np > 0 ? (int)(*initial_palettes)[0].size() : 0;
    for (int p = 0; p < np; p++)
      for (int c = 0; c < nc && c < (int)(*initial_palettes)[p].size(); c++)
        pals[p][c] = rgba_to_oc((*initial_palettes)[p][c]);
  } else {
    // Phase 1: Initialize palettes
    np = cq1(pals, &nc, tiles, pixels, rs, num_palettes, shared_color_idx,
             fraction_of_pixels, col0_oc,
             slow_dither, dw, dp, dpx, (double)max_val);

    int si2 = 2, ei = colors_per_palette;
    if (col0_is_shared) si2++;

    // Phase 2: Expand palettes
    for (int num_c = si2; num_c <= ei; num_c++)
      expand1(pals, np, &nc, tiles, pixels, rs, fraction_of_pixels, shared_color_idx,
              slow_dither, dw, dp, dpx, (double)max_val);
  }

  // Phase 3: Replace weak colors (10 iterations)
  double min_mse_val = mse(pals, np, nc, tiles);
  OColor min_pals[MAX_PALS][MAX_COLS];
  for (int p = 0; p < np; p++) memcpy(min_pals[p], pals[p], nc * sizeof(OColor));

  for (int i = 0; i < 10; i++) {
    OColor tmp[MAX_PALS][MAX_COLS];
    replace_weak(tmp, pals, np, nc, tiles, 0.5, 0.5, 1, shared_color_idx,
                 slow_dither, dw, dp, dpx, (double)max_val);
    for (int p = 0; p < np; p++) memcpy(pals[p], tmp[p], nc * sizeof(OColor));
    int it = iter_count(iterations);
    for (int j = 0; j < it; j++)
      move_closer(pals, np, nc, pixels[rs_next(rs)], tiles, alpha, shared_color_idx,
                  slow_dither, dw, dp, dpx, (double)max_val);
    double m = mse(pals, np, nc, tiles);
    if (m < min_mse_val) {
      min_mse_val = m;
      for (int p = 0; p < np; p++) memcpy(min_pals[p], pals[p], nc * sizeof(OColor));
    }
  }
  for (int p = 0; p < np; p++) memcpy(pals[p], min_pals[p], nc * sizeof(OColor));

  OColor rp[MAX_PALS][MAX_COLS];

  if (!use_dither) {
    // Round to valid values before fine-tune (non-dithered only)
    reduce_pals(rp, pals, np, nc);
    for (int p = 0; p < np; p++) memcpy(pals[p], rp[p], nc * sizeof(OColor));
  }

  // Phase 4: Fine-tune
  int fit = iter_count(iterations * 10);
  for (int i = 0; i < fit; i++)
    move_closer(pals, np, nc, pixels[rs_next(rs)], tiles, final_alpha, shared_color_idx,
                slow_dither, dw, dp, dpx, (double)max_val);

  if (!use_dither) {
    // Phase 5: K-means (non-dithered only)
    reduce_pals(rp, pals, np, nc);
    for (int p = 0; p < np; p++) memcpy(pals[p], rp[p], nc * sizeof(OColor));
    for (int i = 0; i < 3; i++) {
      OColor km[MAX_PALS][MAX_COLS];
      kmeans(km, pals, np, nc, tiles, shared_color_idx);
      for (int p = 0; p < np; p++) memcpy(pals[p], km[p], nc * sizeof(OColor));
    }
  }

  // Final round to valid values
  reduce_pals(rp, pals, np, nc);
  for (int p = 0; p < np; p++) memcpy(pals[p], rp[p], nc * sizeof(OColor));

  // Clamp to valid range
  for (int p = 0; p < np; p++)
    for (int c = 0; c < nc; c++)
      oc_clamp(pals[p][c], 0, max_val);

  // Phase 6: Sort for display
  int dnc = nc;
  OColor sinp[MAX_PALS][MAX_COLS];
  for (int p = 0; p < np; p++) memcpy(sinp[p], pals[p], nc * sizeof(OColor));
  int ss = 0;
  if (col0_is_shared) ss = 1;
  OColor disp_pals[MAX_PALS][MAX_COLS];
  sort_palettes(disp_pals, sinp, np, dnc, ss, rng);

  // Build result
  OptimizedResult result;
  for (int p = 0; p < np; p++) {
    rgba_vec_t pal_colors;
    for (int c = 0; c < nc; c++)
      pal_colors.push_back(oc_to_rgba(pals[p][c]));
    result.palettes.push_back(pal_colors);
  }
  for (int p = 0; p < np; p++) {
    rgba_vec_t pal_colors;
    for (int c = 0; c < dnc; c++)
      pal_colors.push_back(oc_to_rgba(disp_pals[p][c]));
    result.display_palettes.push_back(pal_colors);
  }

  return result;
}

// ── Quantize image using palettes ─────────────────────────────────────────────

channel_vec_t sgd_quantize(
    const channel_vec_t& image_data,
    unsigned width, unsigned height,
    unsigned tile_width, unsigned tile_height,
    const std::vector<rgba_vec_t>& palettes,
    Mode mode,
    const DitherOptions& dither,
    bool use_lab)
{
  unsigned max_val = max_channel_value_for_mode(mode);

  // Set Lab mode globals
  g_use_lab = use_lab;
  g_max_val = (double)max_val;
  if (use_lab) reset_lab_cache();

  // Reduce image to mode-specific colors
  std::vector<rgba_t> reduced(width * height);
  for (unsigned i = 0; i < width * height; i++) {
    rgba_t pixel = image_data[i * 4]
                 | (image_data[i * 4 + 1] << 8)
                 | (image_data[i * 4 + 2] << 16)
                 | (image_data[i * 4 + 3] << 24);
    reduced[i] = reduce_color(pixel, mode);
  }

  // Convert palettes to OColor format
  int np = (int)palettes.size();
  int nc = np > 0 ? (int)palettes[0].size() : 0;
  OColor opals[MAX_PALS][MAX_COLS];
  for (int p = 0; p < np; p++)
    for (int c = 0; c < nc && c < (int)palettes[p].size(); c++)
      opals[p][c] = rgba_to_oc(palettes[p][c]);

  // Dither config
  bool use_dither = dither.mode != DitherMode::off;
  const int (*dp)[2] = nullptr;
  int dpx = 0;
  double dw = 0;
  if (use_dither) {
    dp = dither_pattern_matrix(dither.pattern);
    dpx = dither_pattern_candidates(dither.pattern);
    dw = dither.weight;
  }

  // Initialize output: transparent pixels stay transparent
  channel_vec_t output(width * height * 4, 0);
  for (unsigned i = 0; i < width * height; i++) {
    if (reduced[i] == transparent_color) {
      // Keep transparent (all zeros)
    } else {
      // Will be overwritten by quantized color below
      rgba_t norm = normalize_color(reduced[i], mode);
      output[i * 4]     = norm & 0xff;
      output[i * 4 + 1] = (norm >> 8) & 0xff;
      output[i * 4 + 2] = (norm >> 16) & 0xff;
      output[i * 4 + 3] = (norm >> 24) & 0xff;
    }
  }

  // For each tile region, find best palette and quantize pixels
  for (unsigned sy = 0; sy < height; sy += tile_height) {
    for (unsigned sx = 0; sx < width; sx += tile_width) {
      unsigned ex = std::min(sx + tile_width, width);
      unsigned ey = std::min(sy + tile_height, height);

      // Build OTile for palette matching
      OTile tile;
      for (unsigned y = sy; y < ey; y++) {
        for (unsigned x = sx; x < ex; x++) {
          rgba_t pixel = reduced[x + width * y];
          if (pixel == transparent_color) continue;
          OColor c = rgba_to_oc(pixel);
          tile.px.push_back(x);
          tile.py.push_back(y);
          tile.pcols.push_back(c);
          tile.num_pixels++;
          int found = -1;
          for (int i = 0; i < tile.num_colors; i++)
            if (oc_eq(tile.colors[i], c)) { found = i; break; }
          if (found >= 0) tile.counts[found] += 1;
          else { tile.colors.push_back(c); tile.counts.push_back(1); tile.num_colors++; }
        }
      }

      if (tile.num_colors == 0) continue;

      // Find best palette for this tile
      int pi;
      if (use_dither)
        pi = closest_pal_idx_d(opals, np, nc, tile, dw, dp, dpx, (double)max_val);
      else
        pi = closest_pal_idx(opals, np, nc, tile);

      // Quantize each pixel
      for (int p = 0; p < tile.num_pixels; p++) {
        unsigned px = tile.px[p];
        unsigned py = tile.py[p];
        OColor pcol = tile.pcols[p];

        OColor qcol;
        if (use_dither) {
          DRes dr = closest_color_dither(opals[pi], nc, pcol, px, py, dw, dp, dpx, (double)max_val);
          qcol = opals[pi][dr.idx];
        } else {
          CRes cr = closest_color(opals[pi], nc, pcol);
          qcol = opals[pi][cr.idx];
        }

        // Round and clamp
        qcol = oc_round(qcol);
        oc_clamp(qcol, 0, max_val);

        // Convert to normalized 8-bit RGBA
        rgba_t qrgba = oc_to_rgba(qcol);
        rgba_t norm = normalize_color(qrgba, mode);
        unsigned idx = px + width * py;
        output[idx * 4]     = norm & 0xff;
        output[idx * 4 + 1] = (norm >> 8) & 0xff;
        output[idx * 4 + 2] = (norm >> 16) & 0xff;
        output[idx * 4 + 3] = (norm >> 24) & 0xff;
      }
    }
  }

  return output;
}

// ── Tile-clustering palette optimization ──────────────────────────────────────

// Helper: collect unique colors from tiles assigned to a cluster
static void collect_cluster_colors(
    const std::vector<OTile>& tiles, const std::vector<int>& assignment,
    int cluster_id, int nt,
    std::vector<OColor>& ccolors, std::vector<double>& ccounts)
{
  ccolors.clear();
  ccounts.clear();
  for (int t = 0; t < nt; t++) {
    if (assignment[t] != cluster_id) continue;
    for (int i = 0; i < tiles[t].num_colors; i++) {
      OColor c = tiles[t].colors[i];
      int found = -1;
      for (int j = 0; j < (int)ccolors.size(); j++)
        if (oc_eq(ccolors[j], c)) { found = j; break; }
      if (found >= 0) ccounts[found] += tiles[t].counts[i];
      else { ccolors.push_back(c); ccounts.push_back(tiles[t].counts[i]); }
    }
  }
}

// Helper: build palette for a cluster using greedy or k-means++ initialization
static void build_cluster_palette(
    OColor* pal, int nc, int start, OColor col0_oc,
    const std::vector<OColor>& ccolors, const std::vector<double>& ccounts,
    OColor default_color, bool use_greedy, Mulberry32& rng)
{
  int ncc = (int)ccolors.size();
  int slots = nc - start;
  if (start > 0) pal[0] = col0_oc;

  if (ncc == 0) {
    for (int c = 0; c < slots; c++) pal[start + c] = default_color;
  } else if (use_greedy) {
    // Fill with default first (in case greedy doesn't fill all slots)
    for (int c = 0; c < slots; c++) pal[start + c] = ccolors[0];
    greedy_palette_select(ccolors, ccounts, pal, nc, start);
  } else if (ncc <= slots) {
    for (int c = 0; c < ncc; c++) pal[start + c] = ccolors[c];
    for (int c = ncc; c < slots; c++) pal[start + c] = ccolors[ncc - 1];
  } else {
    // K-means++ on cluster colors
    std::vector<double> cdists(ncc, 1e30);
    int pick = (int)(m32_next(rng) * ncc);
    pal[start] = ccolors[pick];
    for (int ci = 1; ci < slots; ci++) {
      for (int j = 0; j < ncc; j++) {
        double d = oc_dist(ccolors[j], pal[start + ci - 1]);
        if (d < cdists[j]) cdists[j] = d;
      }
      double total = 0;
      for (int j = 0; j < ncc; j++) total += cdists[j] * ccounts[j];
      double r = m32_next(rng) * total;
      double cum = 0;
      int sel = ncc - 1;
      for (int j = 0; j < ncc; j++) {
        cum += cdists[j] * ccounts[j];
        if (cum >= r) { sel = j; break; }
      }
      pal[start + ci] = ccolors[sel];
    }
  }
}

OptimizedResult cluster_optimize(
    const channel_vec_t& image_data,
    unsigned width, unsigned height,
    unsigned tile_width, unsigned tile_height,
    unsigned num_palettes,
    unsigned colors_per_palette,
    Mode mode,
    unsigned max_iterations,
    bool col0_is_shared,
    rgba_t col0_value,
    uint32_t seed,
    bool use_lab,
    bool use_greedy,
    bool hierarchical)
{
  if (seed == 0) seed = (uint32_t)time(nullptr);
  Mulberry32 rng = {seed};

  // Set Lab mode globals
  unsigned max_val = max_channel_value_for_mode(mode);
  g_use_lab = use_lab;
  g_max_val = (double)max_val;
  if (use_lab) reset_lab_cache();

  // Reduce image colors to mode-specific color space
  std::vector<rgba_t> reduced(width * height);
  for (unsigned i = 0; i < width * height; i++) {
    rgba_t pixel = image_data[i * 4]
                 | (image_data[i * 4 + 1] << 8)
                 | (image_data[i * 4 + 2] << 16)
                 | (image_data[i * 4 + 3] << 24);
    reduced[i] = reduce_color(pixel, mode);
  }

  // Extract tiles
  std::vector<OTile> tiles;
  extract_tiles(tiles, reduced, width, height, tile_width, tile_height);
  if (tiles.empty()) return {{}, {}};

  int nt = (int)tiles.size();
  int np = std::min((int)num_palettes, nt);
  int nc = (int)colors_per_palette;
  int shared_color_idx = col0_is_shared ? 0 : -1;
  OColor col0_oc = rgba_to_oc(col0_value);
  int start = col0_is_shared ? 1 : 0;

  // Compute tile centroids (weighted average color, always in RGB for averaging)
  std::vector<OColor> centroids(nt);
  for (int t = 0; t < nt; t++) {
    OColor sum = {0, 0, 0};
    double total = 0;
    for (int i = 0; i < tiles[t].num_colors; i++) {
      OColor w = tiles[t].colors[i];
      oc_scl(w, tiles[t].counts[i]);
      oc_add(sum, w);
      total += tiles[t].counts[i];
    }
    if (total > 0) oc_scl(sum, 1.0 / total);
    centroids[t] = sum;
  }

  OColor pals[MAX_PALS][MAX_COLS];
  std::vector<int> assignment(nt, 0);

  if (hierarchical && np > 1) {
    // ── Hierarchical divisive clustering ──────────────────────────────────
    int current_k = 1;

    // Phase 1: Build initial global palette (all tiles in cluster 0)
    {
      std::vector<OColor> ccolors;
      std::vector<double> ccounts;
      collect_cluster_colors(tiles, assignment, 0, nt, ccolors, ccounts);
      build_cluster_palette(pals[0], nc, start, col0_oc, ccolors, ccounts,
                           centroids[0], use_greedy, rng);
    }

    // Phase 2: Progressive split until we have np clusters
    while (current_k < np) {
      // Find worst cluster by total error
      int worst = 0;
      double worst_err = 0;
      for (int k = 0; k < current_k; k++) {
        double err = 0;
        for (int t = 0; t < nt; t++) {
          if (assignment[t] != k) continue;
          err += pal_dist(pals[k], nc, tiles[t]);
        }
        if (err > worst_err) { worst_err = err; worst = k; }
      }

      // Collect tiles in worst cluster
      std::vector<int> worst_tiles;
      for (int t = 0; t < nt; t++)
        if (assignment[t] == worst) worst_tiles.push_back(t);

      if ((int)worst_tiles.size() < 2) {
        // Can't split a single tile; find another cluster
        // Fill remaining palettes with the worst cluster's palette
        for (int k = current_k; k < np; k++)
          memcpy(pals[k], pals[worst], nc * sizeof(OColor));
        current_k = np;
        break;
      }

      // 2-means split on centroids (using oc_dist which respects Lab mode)
      // Init: pick the two most distant centroids in worst cluster
      int c1_idx = worst_tiles[0], c2_idx = worst_tiles[0];
      double max_dist = 0;
      for (int i = 0; i < (int)worst_tiles.size(); i++) {
        for (int j = i + 1; j < (int)worst_tiles.size(); j++) {
          double d = oc_dist(centroids[worst_tiles[i]], centroids[worst_tiles[j]]);
          if (d > max_dist) {
            max_dist = d;
            c1_idx = worst_tiles[i];
            c2_idx = worst_tiles[j];
          }
        }
      }

      OColor center1 = centroids[c1_idx];
      OColor center2 = centroids[c2_idx];

      // Run 2-means for a few iterations
      for (int iter = 0; iter < 20; iter++) {
        OColor sum1 = {0,0,0}, sum2 = {0,0,0};
        int cnt1 = 0, cnt2 = 0;
        for (int ti : worst_tiles) {
          double d1 = oc_dist(centroids[ti], center1);
          double d2 = oc_dist(centroids[ti], center2);
          if (d1 <= d2) { oc_add(sum1, centroids[ti]); cnt1++; }
          else { oc_add(sum2, centroids[ti]); cnt2++; }
        }
        if (cnt1 > 0) { oc_scl(sum1, 1.0/cnt1); center1 = sum1; }
        if (cnt2 > 0) { oc_scl(sum2, 1.0/cnt2); center2 = sum2; }
      }

      // Final assignment of tiles in worst cluster to group_a (stays) or group_b (new)
      int new_k = current_k;
      for (int ti : worst_tiles) {
        double d1 = oc_dist(centroids[ti], center1);
        double d2 = oc_dist(centroids[ti], center2);
        if (d2 < d1) assignment[ti] = new_k;
        // else stays at 'worst'
      }

      // Build palettes for both sub-groups
      {
        std::vector<OColor> cc; std::vector<double> cn;
        collect_cluster_colors(tiles, assignment, worst, nt, cc, cn);
        OColor def = cc.empty() ? center1 : cc[0];
        build_cluster_palette(pals[worst], nc, start, col0_oc, cc, cn, def, use_greedy, rng);
      }
      {
        std::vector<OColor> cc; std::vector<double> cn;
        collect_cluster_colors(tiles, assignment, new_k, nt, cc, cn);
        OColor def = cc.empty() ? center2 : cc[0];
        build_cluster_palette(pals[new_k], nc, start, col0_oc, cc, cn, def, use_greedy, rng);
      }

      current_k++;
    }

    // Phase 3: Iterative refinement
    for (unsigned iter = 0; iter < 20; iter++) {
      int changes = 0;
      for (int t = 0; t < nt; t++) {
        int best = closest_pal_idx(pals, np, nc, tiles[t]);
        if (best != assignment[t]) { assignment[t] = best; changes++; }
      }
      if (changes == 0) break;

      // Rebuild palettes
      for (int k = 0; k < np; k++) {
        std::vector<OColor> cc; std::vector<double> cn;
        collect_cluster_colors(tiles, assignment, k, nt, cc, cn);
        if (!cc.empty())
          build_cluster_palette(pals[k], nc, start, col0_oc, cc, cn, cc[0], use_greedy, rng);
      }
    }

  } else {
    // ── Flat k-means clustering (original algorithm) ─────────────────────

    // K-means++ initialization on tile centroids
    std::vector<int> centers(np);
    centers[0] = (int)(m32_next(rng) * nt);
    std::vector<double> kpp_dists(nt, 1e30);
    for (int k = 1; k < np; k++) {
      for (int t = 0; t < nt; t++) {
        double d = oc_dist(centroids[t], centroids[centers[k - 1]]);
        if (d < kpp_dists[t]) kpp_dists[t] = d;
      }
      double total = 0;
      for (int t = 0; t < nt; t++) total += kpp_dists[t];
      double r = m32_next(rng) * total;
      double cum = 0;
      int sel = nt - 1;
      for (int t = 0; t < nt; t++) {
        cum += kpp_dists[t];
        if (cum >= r) { sel = t; break; }
      }
      centers[k] = sel;
    }

    // Initial tile-to-cluster assignment
    for (int t = 0; t < nt; t++) {
      double best = 1e30;
      for (int k = 0; k < np; k++) {
        double d = oc_dist(centroids[t], centroids[centers[k]]);
        if (d < best) { best = d; assignment[t] = k; }
      }
    }

    // Initialize palettes from cluster color data
    for (int p = 0; p < np; p++) {
      std::vector<OColor> ccolors;
      std::vector<double> ccounts;
      collect_cluster_colors(tiles, assignment, p, nt, ccolors, ccounts);
      build_cluster_palette(pals[p], nc, start, col0_oc, ccolors, ccounts,
                           centroids[centers[p]], use_greedy, rng);
    }

    // Main iteration loop: assign tiles → rebuild palettes → repeat
    for (unsigned iter = 0; iter < max_iterations; iter++) {
      bool changed = false;

      // Assign tiles to closest palette
      for (int t = 0; t < nt; t++) {
        int pi = closest_pal_idx(pals, np, nc, tiles[t]);
        if (pi != assignment[t]) { assignment[t] = pi; changed = true; }
      }

      if (!changed && iter > 0) break;

      if (use_greedy) {
        // Greedy palette rebuild
        for (int p = 0; p < np; p++) {
          std::vector<OColor> cc; std::vector<double> cn;
          collect_cluster_colors(tiles, assignment, p, nt, cc, cn);
          if (!cc.empty())
            build_cluster_palette(pals[p], nc, start, col0_oc, cc, cn, cc[0], true, rng);
        }
      } else {
        // Inner k-means: rebuild palette colors from assigned tiles
        int kcounts[MAX_PALS][MAX_COLS] = {};
        OColor ksums[MAX_PALS][MAX_COLS];
        for (int p = 0; p < np; p++)
          for (int c = 0; c < nc; c++)
            ksums[p][c] = {0, 0, 0};

        for (int t = 0; t < nt; t++) {
          int p = assignment[t];
          for (int i = 0; i < tiles[t].num_colors; i++) {
            CRes r = closest_color(pals[p], nc, tiles[t].colors[i]);
            kcounts[p][r.idx] += (int)tiles[t].counts[i];
            OColor w = tiles[t].colors[i];
            oc_scl(w, tiles[t].counts[i]);
            oc_add(ksums[p][r.idx], w);
          }
        }

        for (int p = 0; p < np; p++)
          for (int c = 0; c < nc; c++) {
            if (c == shared_color_idx || kcounts[p][c] == 0) continue;
            pals[p][c] = ksums[p][c];
            oc_scl(pals[p][c], 1.0 / kcounts[p][c]);
          }
      }
    }
  }

  // Round to valid values
  OColor rp[MAX_PALS][MAX_COLS];
  reduce_pals(rp, pals, np, nc);
  for (int p = 0; p < np; p++) memcpy(pals[p], rp[p], nc * sizeof(OColor));

  // Clamp to valid range
  for (int p = 0; p < np; p++)
    for (int c = 0; c < nc; c++)
      oc_clamp(pals[p][c], 0, max_val);

  // Sort for display
  OColor sinp[MAX_PALS][MAX_COLS];
  for (int p = 0; p < np; p++) memcpy(sinp[p], pals[p], nc * sizeof(OColor));
  int ss = col0_is_shared ? 1 : 0;
  OColor disp_pals[MAX_PALS][MAX_COLS];
  sort_palettes(disp_pals, sinp, np, nc, ss, rng);

  // Build result
  OptimizedResult result;
  for (int p = 0; p < np; p++) {
    rgba_vec_t pal_colors;
    for (int c = 0; c < nc; c++)
      pal_colors.push_back(oc_to_rgba(pals[p][c]));
    result.palettes.push_back(pal_colors);
  }
  for (int p = 0; p < np; p++) {
    rgba_vec_t pal_colors;
    for (int c = 0; c < nc; c++)
      pal_colors.push_back(oc_to_rgba(disp_pals[p][c]));
    result.display_palettes.push_back(pal_colors);
  }

  return result;
}

// ── Quality assessment ─────────────────────────────────────────────────────────

QualityReport compute_quality(
    const channel_vec_t& image_data,
    const channel_vec_t& quantized_data,
    unsigned width, unsigned height,
    Mode mode,
    bool use_lab)
{
  unsigned mv = max_channel_value_for_mode(mode);
  double max_val_d = (double)mv;

  // Set Lab globals for cache
  if (use_lab) {
    g_use_lab = true;
    g_max_val = max_val_d;
    reset_lab_cache();
  }

  double sum_error = 0;
  double max_error = 0;
  unsigned exact = 0;
  unsigned de_lt5 = 0;
  unsigned total = 0;

  for (unsigned i = 0; i < width * height; i++) {
    rgba_t orig_pixel = image_data[i * 4]
                      | (image_data[i * 4 + 1] << 8)
                      | (image_data[i * 4 + 2] << 16)
                      | (image_data[i * 4 + 3] << 24);
    rgba_t orig_reduced = reduce_color(orig_pixel, mode);
    if (orig_reduced == transparent_color) continue;

    rgba_t quant_pixel = quantized_data[i * 4]
                       | (quantized_data[i * 4 + 1] << 8)
                       | (quantized_data[i * 4 + 2] << 16)
                       | (quantized_data[i * 4 + 3] << 24);
    rgba_t quant_reduced = reduce_color(quant_pixel, mode);

    double or_ = orig_reduced & 0xff, og = (orig_reduced >> 8) & 0xff, ob = (orig_reduced >> 16) & 0xff;
    double qr = quant_reduced & 0xff, qg = (quant_reduced >> 8) & 0xff, qb = (quant_reduced >> 16) & 0xff;

    double dist;
    if (use_lab) {
      OColor lab_orig = cached_rgb_to_lab(oc(or_, og, ob));
      OColor lab_quant = cached_rgb_to_lab(oc(qr, qg, qb));
      double dL = lab_orig.r - lab_quant.r;
      double da = lab_orig.g - lab_quant.g;
      double db2 = lab_orig.b - lab_quant.b;
      dist = dL*dL + da*da + db2*db2;
      if (sqrt(dist) < 5.0) de_lt5++;
    } else {
      double dr = or_ - qr, dg = og - qg, db = ob - qb;
      dist = 2 * dr * dr + 4 * dg * dg + db * db;
    }

    sum_error += dist;
    if (dist > max_error) max_error = dist;
    if (dist == 0) exact++;
    total++;
  }

  QualityReport report;
  report.total_pixels = total;
  report.mse = total > 0 ? sum_error / total : 0;

  if (use_lab) {
    // For Lab: PSNR based on Lab range (L: 0-100, a/b: ~-128 to 127)
    // Max possible Delta-E² ~ 100² + 256² + 256² ≈ 141000
    double max_possible = 100.0*100.0 + 256.0*256.0 + 256.0*256.0;
    report.psnr = report.mse > 0 ? 10.0 * log10(max_possible / report.mse) : 99.99;
  } else {
    double max_possible = 7.0 * mv * mv; // 2*M^2 + 4*M^2 + M^2
    report.psnr = report.mse > 0 ? 10.0 * log10(max_possible / report.mse) : 99.99;
  }
  report.exact_match_pct = total > 0 ? 100.0 * exact / total : 100.0;
  report.max_error = max_error;
  report.pct_de_lt5 = total > 0 ? 100.0 * de_lt5 / total : 100.0;
  return report;
}

} /* namespace sfc */
