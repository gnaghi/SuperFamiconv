# New Features Documentation

This document describes the features added to SuperFamiconv from the palettequant and menuloader/tools projects.

---

## 1. SGD Palette Optimization (`-O` / `--optimize`)

### What it does

Replaces the original greedy set-covering palette builder with a stochastic gradient descent (SGD) optimizer that minimizes overall color error across all tiles.

### Why it's useful

The original SuperFamiconv palette algorithm works by exact color set packing: it finds which tiles can share a palette, then packs colors into palettes. This fails entirely on photographic or gradient-heavy images where tiles contain more unique colors than a palette can hold (e.g., 57 unique colors in an 8x8 tile vs 16 colors per SNES palette).

SGD optimization handles any image gracefully:
- Starts from the average color and progressively refines
- Minimizes perceptual error (weighted MSE: `2*dR^2 + 4*dG^2 + dB^2`)
- Iteratively replaces weak colors and palettes that contribute little
- Final k-means pass snaps palette colors to their optimal positions

### How it works

1. **Phase 1** (Initialize): Average all pixels, split into N palettes by duplicating the worst-performing one
2. **Phase 2** (Expand): Add one color at a time to each palette, training with SGD between each addition
3. **Phase 3** (Replace weak): 10 iterations of replacing the least-useful color in each palette with the most-needed one, training with SGD after each replacement
4. **Phase 4** (Fine-tune): Long SGD pass with low learning rate for final refinement
5. **Phase 5** (K-means): 3 passes of k-means to snap colors to cluster centers

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--seed N` | random | Random seed for reproducible results |
| `--fraction-of-pixels F` | 0.1 | Training intensity (0.01-10.0). Higher = more SGD iterations per phase |

### Example

```bash
superfamiconv palette -i photo.png -O --seed 42 -o quantized.png -d palette.bin -v
```

---

## 2. Dithering Support (`--dither`)

### What it does

Adds ordered dithering when mapping pixels to palette colors, creating the illusion of more colors by mixing nearby palette colors in a pattern.

### Why it's useful

Without dithering, each pixel is mapped to the single closest palette color. For smooth gradients this produces visible banding (flat color steps). Dithering distributes quantization error using a 2x2 ordered pattern, breaking up banding into a more natural-looking texture.

### How it works

For each pixel, the dithering algorithm:
1. Converts the pixel to linear color space
2. Iteratively finds the closest palette color and accumulates error
3. Produces 2 or 4 candidate colors per pixel (depending on pattern)
4. Sorts candidates by brightness and selects one based on the pixel's position in the 2x2 pattern matrix

This is an **ordered dither** (not error diffusion like Floyd-Steinberg), making it deterministic and parallelizable.

### Modes

| Mode | Training | Output | Speed | Quality |
|------|----------|--------|-------|---------|
| `off` | Standard SGD | Nearest color | Fast | Good for simple images |
| `fast` | Standard SGD | Dithered | Medium | Smooth gradients, fast training |
| `slow` | Dithered SGD (1/5 iterations, alpha=0.1) | Dithered | Slow | Best quality, palettes trained for dithering |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dither MODE` | off | Dithering mode: `off`, `fast`, `slow` |
| `--dither-weight W` | 0.5 | Error accumulation weight (0.01-1.0). Higher = more dithering |
| `--dither-pattern P` | diagonal4 | Pattern name (see below) |

### Patterns

All patterns use a 2x2 matrix. 4-candidate patterns produce finer dithering, 2-candidate patterns are coarser.

| Pattern | Candidates | Matrix |
|---------|-----------|--------|
| `diagonal4` | 4 | `[[0,2],[3,1]]` |
| `horizontal4` | 4 | `[[0,3],[1,2]]` |
| `vertical4` | 4 | `[[0,1],[3,2]]` |
| `diagonal2` | 2 | `[[0,1],[1,0]]` |
| `horizontal2` | 2 | `[[0,1],[0,1]]` |
| `vertical2` | 2 | `[[0,0],[1,1]]` |

### Example

```bash
superfamiconv palette -i photo.png -O --dither fast --dither-weight 0.5 -o dithered.png
```

---

## 3. Tile-Clustering Optimization (`-K` / `--cluster`)

### What it does

Groups tiles by color similarity using k-means clustering, then builds one palette per cluster. This ensures tiles with similar colors share a palette, which is more efficient than the greedy approach.

### Why it's useful

The greedy algorithm processes tiles in scan order and assigns them to palettes without considering the global picture. Tile-clustering considers all tiles simultaneously:
- Tiles with similar colors are grouped together
- Each group gets a palette tailored to its colors
- The result is better color distribution across palettes

### How it works

1. **Compute tile centroids**: Weighted average color of each tile
2. **K-means++ initialization**: Select K initial cluster centers with probability proportional to distance (avoids poor random initialization)
3. **Assign tiles to clusters**: Each tile goes to the cluster with the closest centroid
4. **Build palettes per cluster**: K-means++ on the unique colors within each cluster to pick diverse initial palette colors
5. **Iterate**: Re-assign tiles to closest palette, rebuild palette colors via k-means, repeat until convergence (max 50 iterations)

### Combined with SGD (`-K -O`)

When both `-K` and `-O` are specified:
1. Clustering runs first to produce initial palettes
2. These palettes are passed to SGD as starting points (skipping SGD's Phase 1-2)
3. SGD refines the cluster-initialized palettes through Phases 3-5

This combination often produces the best results: clustering provides good initial palette assignments, and SGD fine-tunes the individual colors.

### Example

```bash
# Cluster only
superfamiconv palette -i photo.png -K --seed 42 -o clustered.png

# Cluster + SGD (recommended for best quality)
superfamiconv palette -i photo.png -K -O --seed 42 -o optimized.png
```

---

## 4. Quality Assessment (`-Q` / `--quality`)

### What it does

After palette creation, computes and prints quality metrics comparing the original image against the palette-quantized output.

### Why it's useful

Without quality metrics, the only way to evaluate palette quality is visual inspection. Quality assessment provides objective numbers to:
- Compare different optimization strategies (greedy vs SGD vs cluster)
- Tune parameters (number of palettes, colors per palette, dithering)
- Track regressions when changing algorithms
- Automate quality gates in build pipelines

### Metrics

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error using weighted distance `2*dR^2 + 4*dG^2 + dB^2`. Lower is better. Computed in mode-reduced color space. |
| **PSNR** | Peak Signal-to-Noise Ratio in dB. Higher is better. Computed as `10*log10(7*M^2/MSE)` where M is the max channel value (e.g., 31 for SNES 5-bit). |
| **Exact match %** | Percentage of pixels whose quantized color exactly matches the reduced original. Higher means more faithful reproduction. |
| **Max error** | Worst single-pixel weighted error. Identifies outlier pixels that may be visually jarring. |

### Output format

```
Quality (10800 pixels):
  MSE:          1.9147
  PSNR:         35.46 dB
  Exact match:  38.9%
  Max error:    60.0
```

Output goes to stderr, so it doesn't interfere with piped data output.

### Example

```bash
# Compare SGD vs clustering
superfamiconv palette -i photo.png -O -Q --seed 42
superfamiconv palette -i photo.png -K -Q --seed 42
superfamiconv palette -i photo.png -K -O -Q --seed 42
```

---

## 5. Composite Preview Image (`--out-preview`)

### What it does

Generates a single PNG file showing the quantized result image stacked above a scaled palette swatch, separated by a thin line.

### Why it's useful

SuperFamiconv outputs native binary formats (palette, tiles, map). To see the visual result, you'd normally need an emulator or separate tools. The composite preview gives immediate visual feedback in a single image:
- **Top**: How the image will look on the target console (all quantization, dithering, and palette constraints applied)
- **Bottom**: The palette colors used, scaled up so each color is clearly visible

This is invaluable during parameter tuning: you can immediately see how changing the number of palettes, dithering mode, or optimization strategy affects the output.

### Details

- Each palette color cell is scaled to fill the image width (minimum 4px per cell, maximum 32px)
- The separator line is 2px dark gray (#404040)
- Works with all optimization modes (greedy, SGD, cluster, cluster+SGD)
- Available in both `palette` subcommand and shorthand mode

### Example

```bash
# Preview with SGD optimization and dithering
superfamiconv palette -i photo.png -O --dither fast --out-preview preview.png

# Shorthand mode with full pipeline + preview
superfamiconv -i photo.png -O -Q --out-preview preview.png -p pal.bin -t tiles.bin -m map.bin
```

---

## Feature Interaction Matrix

| | SGD (-O) | Cluster (-K) | Dither | Quality (-Q) | Preview |
|---|---|---|---|---|---|
| **Greedy** (default) | - | - | N/A | Yes | Yes |
| **SGD** | Yes | - | off/fast/slow | Yes | Yes |
| **Cluster** | - | Yes | N/A | Yes | Yes |
| **Cluster+SGD** | Yes | Yes | off/fast/slow | Yes | Yes |

Notes:
- Dithering only applies when using SGD (with or without cluster initialization)
- Quality assessment works with all modes
- Preview works with all modes
- Greedy mode fails on photographic images (too many colors per tile); use `-O` or `-K` instead
