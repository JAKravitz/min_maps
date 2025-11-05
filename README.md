# Hyperspectral Mineral Classification and Abundance Mapping

A Python tool for processing hyperspectral imagery to generate mineral abundance maps and hard classification maps using linear spectral unmixing.

## Overview

This project processes hyperspectral satellite/airborne imagery to identify and map the abundance of three iron-oxide minerals:
- **Jarosite** (yellow)
- **Hematite** (red)
- **Goethite** (orange)

## Features

- **Linear Spectral Unmixing**: Constrained non-negative least squares unmixing for abundance estimation
- **Hard Classification**: Winner-take-all classification from abundance maps
- **AOI Extraction**: Extract region of interest based on lat/lon coordinates and diameter
- **Visualization**: Generate publication-quality PNG outputs

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.21.0
- scipy >= 1.7.0
- rasterio >= 1.3.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- pyproj >= 3.3.0

## Input Data

### 1. Hyperspectral Image
- **Format**: GeoTIFF with ENVI header (.tif + .tif.hdr)
- **Location**: `/Users/jeremy/Desktop/ff3-1101/`
- **Files**:
  - `FF03_20250707_00501045_0000001101_L2A.tif`
  - `FF03_20250707_00501045_0000001101_L2A.tif.hdr`

### 2. Spectral Libraries
- **Format**: CSV files with averaged spectra
- **Location**: `/Users/jeremy/Desktop/`
- **Files**:
  - `jarosite_convolved_to_sensor.csv`
  - `hematite_convolved_to_sensor.csv`
  - `goethite_convolved_to_sensor.csv`

Each library contains multiple reference spectra and an averaged spectrum (used for unmixing).

### 3. Area of Interest (AOI)
- **Center**: Latitude -18.91386, Longitude 128.82889
- **Size**: 30 km diameter

## Usage

Simply run the main script:

```bash
python mineral_classifier.py
```

The script will:
1. Load the hyperspectral image
2. Parse wavelengths from the ENVI header
3. Load and average the spectral libraries
4. Extract the AOI based on lat/lon coordinates
5. Perform constrained linear spectral unmixing
6. Generate hard classification map
7. Create and save all visualizations

## Outputs

All outputs are saved as PNG files in the `outputs/` directory:

1. **`1_rgb_with_aoi.png`**
   - RGB composite of the full scene
   - Red box showing the AOI location

2. **`2_abundance_maps.png`**
   - Combined view of all three abundance maps
   - Color scale: 0 (black) to 1 (white/red)

3. **`2_jarosite_abundance.png`**
   - Individual abundance map for jarosite

4. **`2_hematite_abundance.png`**
   - Individual abundance map for hematite

5. **`2_goethite_abundance.png`**
   - Individual abundance map for goethite

6. **`3_classification_map.png`**
   - Hard classification map with legend
   - Colors:
     - Black: Background (low abundance)
     - Yellow: Jarosite
     - Red: Hematite
     - Orange: Goethite

## Methodology

### Linear Spectral Unmixing

The unmixing uses **constrained non-negative least squares (NNLS)** with the following assumptions:

1. **Linear mixing model**: Each pixel reflectance is a linear combination of endmember spectra
2. **Non-negativity constraint**: Abundances must be >= 0
3. **Sum-to-one constraint**: Abundances are normalized to sum to 1

For each pixel:
```
R(λ) = Σ(ai × Ei(λ))
```

Where:
- `R(λ)` = observed reflectance at wavelength λ
- `ai` = abundance of mineral i
- `Ei(λ)` = endmember spectrum for mineral i

### Hard Classification

Classification uses a **winner-take-all** approach:
1. For each pixel, find the mineral with maximum abundance
2. Assign pixel to that mineral class
3. Pixels with total abundance < 0.1 are classified as background

## Customization

To modify the processing, edit these variables in `mineral_classifier.py`:

```python
# File paths
HSI_PATH = "path/to/your/image.tif"
HDR_PATH = "path/to/your/image.tif.hdr"

SPECTRAL_LIBRARIES = {
    'mineral1': 'path/to/library1.csv',
    'mineral2': 'path/to/library2.csv',
    # ...
}

# AOI parameters
AOI_CENTER_LAT = -18.91386
AOI_CENTER_LON = 128.82889
AOI_DIAMETER_KM = 30

# Output directory
OUTPUT_DIR = 'outputs'
```

You can also modify:
- RGB band selection in `create_rgb_composite()` (default: bands 50, 30, 15)
- Background threshold in `classify_hard()` (default: 0.1)
- Classification colors in `visualize_results()`

## Project Structure

```
min_maps/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── mineral_classifier.py      # Main processing script
└── outputs/                   # Generated visualizations
    ├── 1_rgb_with_aoi.png
    ├── 2_abundance_maps.png
    ├── 2_jarosite_abundance.png
    ├── 2_hematite_abundance.png
    ├── 2_goethite_abundance.png
    └── 3_classification_map.png
```

## Notes

- The script automatically handles different numbers of bands between the image and spectral libraries
- Invalid pixels (NaN or zero values) are skipped during unmixing
- Georeferencing information is preserved but not used in outputs (PNG only)
- Processing time depends on AOI size; expect ~1-5 minutes for a 30km diameter area

## Troubleshooting

**Issue**: "Could not parse wavelengths from header"
- The script will use band indices instead; unmixing should still work if libraries match the sensor

**Issue**: "Endmember bands != image bands"
- The script automatically truncates to the minimum number of bands

**Issue**: Memory errors
- Reduce AOI diameter or process in smaller chunks

## Author

Created by Claude (Anthropic)
Date: 2025-11-05

## License

This code is provided as-is for research and educational purposes.
