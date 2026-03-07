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
static inline double oc_dist(OColor a, OColor b) {
  double dr = a.r - b.r, dg = a.g - b.g, db = a.b - b.b;
  return 2 * dr * dr + 4 * dg * dg + db * db;
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

static inline double js_round(double x) { return floor(x + 0.5); }

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
                        int shared_color_idx) {
  const OTile& tile = tiles[pixel.tile_idx];
  int pi = closest_pal_idx(pals, np, nc, tile);
  CRes r = closest_color(pals[pi], nc, pixel.color);
  if (r.idx != shared_color_idx)
    oc_move(pals[pi][r.idx], pixel.color, alpha);
}

// ── Phase 1: Initialize palettes with 1 color ────────────────────────────────

static int cq1(OColor pals[][MAX_COLS], int* out_nc,
               const std::vector<OTile>& tiles, const std::vector<OPixel>& pixels,
               RandShuf& rs, unsigned num_palettes, int shared_color_idx,
               double fraction_of_pixels, OColor col0_val) {
  int npx = (int)pixels.size();
  double iterations = fraction_of_pixels * npx;
  double alpha = 0.3;

  // Average color
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
      move_closer(pals, np, nc, pixels[rs_next(rs)], tiles, alpha, shared_color_idx);

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
                    RandShuf& rs, double fraction_of_pixels, int shared_color_idx) {
  int nc = *nc_p;
  int npx = (int)pixels.size();
  double iterations = fraction_of_pixels * npx;
  double alpha = 0.3;
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
    move_closer(pals, np, nc, pixels[rs_next(rs)], tiles, alpha, shared_color_idx);
}

// ── Phase 3: Replace weakest colors ───────────────────────────────────────────

static void replace_weak(OColor out[][MAX_COLS], const OColor in[][MAX_COLS],
                        int np, int nc, const std::vector<OTile>& tiles,
                        double mcf, double mpf, int rep_pal, int shared_color_idx) {
  int nt = (int)tiles.size();
  std::vector<int> cpi(nt, 0);
  double tpm[MAX_PALS] = {};
  double rpm[MAX_PALS] = {};
  int mxpi = 0, mnpi = 0;

  if (np > 1) {
    for (int j = 0; j < nt; j++) {
      PRes r = closest_pal_dist(in, np, nc, tiles[j]);
      tpm[r.idx] += r.dist;
      cpi[j] = r.idx;

      // Remaining palettes (exclude closest)
      OColor rem[MAX_PALS][MAX_COLS];
      int rn = 0;
      for (int p = 0; p < np; p++)
        if (p != r.idx) { memcpy(rem[rn], in[p], nc * sizeof(OColor)); rn++; }
      if (rn > 0) {
        PRes r2 = closest_pal_dist(rem, rn, nc, tiles[j]);
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
      for (int i = 0; i < tiles[j].num_colors; i++) {
        CRes r = closest_color(pal, nc, tiles[j].colors[i]);
        tcm[mpi][r.idx] += r.dist * tiles[j].counts[i];
        // Distance without this color
        OColor rm[MAX_COLS];
        int rn = 0;
        for (int k = 0; k < nc; k++)
          if (k != r.idx) rm[rn++] = pal[k];
        CRes r2 = closest_color(rm, rn, tiles[j].colors[i]);
        scm[mpi][r.idx] += r2.dist * tiles[j].counts[i];
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
    uint32_t seed)
{
  if (seed == 0) seed = (uint32_t)time(nullptr);
  Mulberry32 rng = {seed};

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

  int shared_color_idx = col0_is_shared ? 0 : -1;
  OColor col0_oc = rgba_to_oc(col0_value);
  unsigned max_val = max_channel_value_for_mode(mode);

  // Phase 1: Initialize palettes
  OColor pals[MAX_PALS][MAX_COLS];
  int np, nc;
  np = cq1(pals, &nc, tiles, pixels, rs, num_palettes, shared_color_idx,
           fraction_of_pixels, col0_oc);

  int si2 = 2, ei = colors_per_palette;
  if (col0_is_shared) si2++;

  // Phase 2: Expand palettes
  for (int num_c = si2; num_c <= ei; num_c++)
    expand1(pals, np, &nc, tiles, pixels, rs, fraction_of_pixels, shared_color_idx);

  // Phase 3: Replace weak colors (10 iterations)
  double min_mse_val = mse(pals, np, nc, tiles);
  OColor min_pals[MAX_PALS][MAX_COLS];
  for (int p = 0; p < np; p++) memcpy(min_pals[p], pals[p], nc * sizeof(OColor));

  for (int i = 0; i < 10; i++) {
    OColor tmp[MAX_PALS][MAX_COLS];
    replace_weak(tmp, pals, np, nc, tiles, 0.5, 0.5, 1, shared_color_idx);
    for (int p = 0; p < np; p++) memcpy(pals[p], tmp[p], nc * sizeof(OColor));
    int it = iter_count(iterations);
    for (int j = 0; j < it; j++)
      move_closer(pals, np, nc, pixels[rs_next(rs)], tiles, alpha, shared_color_idx);
    double m = mse(pals, np, nc, tiles);
    if (m < min_mse_val) {
      min_mse_val = m;
      for (int p = 0; p < np; p++) memcpy(min_pals[p], pals[p], nc * sizeof(OColor));
    }
  }
  for (int p = 0; p < np; p++) memcpy(pals[p], min_pals[p], nc * sizeof(OColor));

  // Round to valid values
  OColor rp[MAX_PALS][MAX_COLS];
  reduce_pals(rp, pals, np, nc);
  for (int p = 0; p < np; p++) memcpy(pals[p], rp[p], nc * sizeof(OColor));

  // Phase 4: Fine-tune
  int fit = iter_count(iterations * 10);
  for (int i = 0; i < fit; i++)
    move_closer(pals, np, nc, pixels[rs_next(rs)], tiles, final_alpha, shared_color_idx);

  // Phase 5: K-means
  reduce_pals(rp, pals, np, nc);
  for (int p = 0; p < np; p++) memcpy(pals[p], rp[p], nc * sizeof(OColor));
  for (int i = 0; i < 3; i++) {
    OColor km[MAX_PALS][MAX_COLS];
    kmeans(km, pals, np, nc, tiles, shared_color_idx);
    for (int p = 0; p < np; p++) memcpy(pals[p], km[p], nc * sizeof(OColor));
  }
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

} /* namespace sfc */
