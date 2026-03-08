// Palette optimizer
// Ported from palettequant (C implementation)
//
// Provides alternatives to the greedy set-covering palette builder:
// - SGD: stochastic gradient descent + k-means refinement
// - Cluster: tile-level k-means clustering
// Both minimize color error across all tiles.

#pragma once

#include "Common.h"
#include "Mode.h"
#include "Color.h"
#include <cstdint>
#include <vector>

namespace sfc {

// Dithering mode
enum class DitherMode { off, fast, slow };

// Dithering pattern (2x2 ordered dither matrices)
enum class DitherPattern {
  diagonal4, horizontal4, vertical4,
  diagonal2, horizontal2, vertical2
};

// Dithering options
struct DitherOptions {
  DitherMode mode = DitherMode::off;
  DitherPattern pattern = DitherPattern::diagonal4;
  double weight = 0.5;
};

// Result of SGD palette optimization
struct OptimizedResult {
  // Optimized subpalettes in reduced color space (for quantization)
  std::vector<rgba_vec_t> palettes;
  // Display palettes (sorted for visual coherence)
  std::vector<rgba_vec_t> display_palettes;
};

// Run SGD palette optimization on image data.
// Returns palettes in mode-specific reduced color space.
// If initial_palettes is provided, skips Phase 1-2 (initialization/expansion)
// and starts refinement from the given palettes.
OptimizedResult sgd_optimize(
    const channel_vec_t& image_data,
    unsigned width, unsigned height,
    unsigned tile_width, unsigned tile_height,
    unsigned num_palettes,
    unsigned colors_per_palette,
    Mode mode,
    double fraction_of_pixels = 0.1,
    bool col0_is_shared = false,
    rgba_t col0_value = 0,
    uint32_t seed = 0,
    const DitherOptions& dither = {},
    const std::vector<rgba_vec_t>* initial_palettes = nullptr
);

// Run tile-clustering palette optimization.
// Groups tiles by color similarity using k-means, then builds
// one palette per cluster via inner k-means on cluster colors.
OptimizedResult cluster_optimize(
    const channel_vec_t& image_data,
    unsigned width, unsigned height,
    unsigned tile_width, unsigned tile_height,
    unsigned num_palettes,
    unsigned colors_per_palette,
    Mode mode,
    unsigned max_iterations = 50,
    bool col0_is_shared = false,
    rgba_t col0_value = 0,
    uint32_t seed = 0
);

// Quantize an image using the given palettes, with optional dithering.
// Returns RGBA image data normalized to 8-bit per channel.
// Each pixel is mapped to the closest color in the best-matching subpalette.
channel_vec_t sgd_quantize(
    const channel_vec_t& image_data,
    unsigned width, unsigned height,
    unsigned tile_width, unsigned tile_height,
    const std::vector<rgba_vec_t>& palettes,
    Mode mode,
    const DitherOptions& dither = {}
);

// Get the maximum channel value for a given mode (e.g., 31 for 5-bit SNES)
unsigned max_channel_value_for_mode(Mode mode);

// Quality assessment report
struct QualityReport {
  double mse;              // Mean squared error (weighted: 2*dR^2+4*dG^2+dB^2)
  double psnr;             // Peak signal-to-noise ratio (dB)
  double exact_match_pct;  // Percentage of pixels with exact color match
  double max_error;        // Worst single-pixel weighted error
  unsigned total_pixels;   // Total non-transparent pixels evaluated
};

// Compute quality metrics comparing original vs quantized image.
// Both inputs are raw RGBA channel vectors (8-bit per channel).
// Comparison is done in reduced color space for the given mode.
QualityReport compute_quality(
    const channel_vec_t& image_data,
    const channel_vec_t& quantized_data,
    unsigned width, unsigned height,
    Mode mode
);

} /* namespace sfc */
