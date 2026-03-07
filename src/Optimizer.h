// SGD-based palette optimizer
// Ported from palettequant (C implementation)
//
// Provides an alternative to the greedy set-covering palette builder:
// uses stochastic gradient descent + k-means refinement to minimize
// color error across all tiles.

#pragma once

#include "Common.h"
#include "Mode.h"
#include "Color.h"
#include <cstdint>
#include <vector>

namespace sfc {

// Result of SGD palette optimization
struct OptimizedResult {
  // Optimized subpalettes in reduced color space (for quantization)
  std::vector<rgba_vec_t> palettes;
  // Display palettes (sorted for visual coherence)
  std::vector<rgba_vec_t> display_palettes;
};

// Run SGD palette optimization on image data.
// Returns palettes in mode-specific reduced color space.
//
// Parameters:
//   image_data      - flat RGBA bytes (full 8-bit color)
//   width, height   - image dimensions
//   tile_width/height - tile size for palette assignment
//   num_palettes    - number of subpalettes to generate
//   colors_per_palette - colors per subpalette
//   mode            - target console mode (determines color space)
//   fraction_of_pixels - training intensity (0.01-10, higher = slower but better)
//   col0_is_shared  - whether color 0 is shared across all subpalettes
//   col0_value      - shared color 0 value (reduced color space, ignored if !col0_is_shared)
//   seed            - PRNG seed for reproducibility (0 = random)
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
    uint32_t seed = 0
);

// Get the maximum channel value for a given mode (e.g., 31 for 5-bit SNES)
unsigned max_channel_value_for_mode(Mode mode);

} /* namespace sfc */
