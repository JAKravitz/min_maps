#!/usr/bin/env python3
"""
Hyperspectral Mineral Classification and Abundance Mapping
Author: Claude
Date: 2025-11-05

This script performs linear spectral unmixing and hard classification
on hyperspectral imagery for mineral mapping.
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from pyproj import Transformer
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class HyperspectralMineralClassifier:
    """
    Classifier for hyperspectral mineral abundance mapping and classification.
    """

    def __init__(self, hsi_path, hdr_path=None):
        """
        Initialize the classifier with hyperspectral image.

        Parameters:
        -----------
        hsi_path : str
            Path to hyperspectral image (GeoTIFF)
        hdr_path : str, optional
            Path to ENVI header file
        """
        self.hsi_path = hsi_path
        self.hdr_path = hdr_path
        self.image = None
        self.wavelengths = None
        self.transform_matrix = None
        self.crs = None
        self.endmembers = {}
        self.mineral_names = []

    def load_image(self):
        """Load the hyperspectral image."""
        print(f"Loading hyperspectral image: {self.hsi_path}")

        with rasterio.open(self.hsi_path) as src:
            self.image = src.read()  # Shape: (bands, rows, cols)
            self.transform_matrix = src.transform
            self.crs = src.crs
            self.n_bands = src.count
            self.height = src.height
            self.width = src.width

            print(f"Image shape: {self.image.shape}")
            print(f"CRS: {self.crs}")

        # Parse wavelengths from header if available
        if self.hdr_path:
            self.wavelengths = self._parse_wavelengths_from_hdr()

        return self.image

    def _parse_wavelengths_from_hdr(self):
        """Parse wavelengths from ENVI header file."""
        wavelengths = []
        try:
            with open(self.hdr_path, 'r') as f:
                content = f.read()

            # Find wavelength section
            if 'wavelength = {' in content:
                start = content.find('wavelength = {') + len('wavelength = {')
                end = content.find('}', start)
                wl_str = content[start:end]
                wavelengths = [float(x.strip()) for x in wl_str.split(',') if x.strip()]
                print(f"Parsed {len(wavelengths)} wavelengths from header")
        except Exception as e:
            print(f"Warning: Could not parse wavelengths from header: {e}")
            wavelengths = list(range(1, self.n_bands + 1))

        return np.array(wavelengths)

    def load_spectral_library(self, csv_path, mineral_name):
        """
        Load spectral library from CSV and extract average spectrum.

        Parameters:
        -----------
        csv_path : str
            Path to CSV file containing spectral library
        mineral_name : str
            Name of the mineral
        """
        print(f"Loading spectral library for {mineral_name}: {csv_path}")

        df = pd.read_csv(csv_path)

        # Look for average/mean spectrum column
        avg_col = None
        for col in df.columns:
            if 'average' in col.lower() or 'mean' in col.lower() or 'avg' in col.lower():
                avg_col = col
                break

        if avg_col:
            spectrum = df[avg_col].values
            print(f"Found average spectrum in column: {avg_col}")
        else:
            # If no average column, compute mean across all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            spectrum = df[numeric_cols].mean(axis=1).values
            print(f"Computed average from {len(numeric_cols)} spectra")

        self.endmembers[mineral_name] = spectrum
        self.mineral_names.append(mineral_name)
        print(f"Loaded spectrum with {len(spectrum)} bands")

        return spectrum

    def latlon_to_pixel(self, lat, lon):
        """
        Convert lat/lon coordinates to pixel coordinates.

        Parameters:
        -----------
        lat : float
            Latitude
        lon : float
            Longitude

        Returns:
        --------
        row, col : int
            Pixel coordinates
        """
        # Transform lat/lon to image CRS
        transformer = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)

        # Convert to pixel coordinates
        inv_transform = ~self.transform_matrix
        col, row = inv_transform * (x, y)

        return int(row), int(col)

    def extract_aoi(self, center_lat, center_lon, diameter_km):
        """
        Extract area of interest as a square box.

        Parameters:
        -----------
        center_lat : float
            Center latitude
        center_lon : float
            Center longitude
        diameter_km : float
            Diameter of AOI in kilometers

        Returns:
        --------
        aoi_image : ndarray
            Extracted AOI
        bounds : tuple
            (row_start, row_end, col_start, col_end)
        """
        print(f"Extracting AOI: {diameter_km}km diameter centered at ({center_lat}, {center_lon})")

        # Get center pixel
        center_row, center_col = self.latlon_to_pixel(center_lat, center_lon)
        print(f"Center pixel: ({center_row}, {center_col})")

        # Calculate approximate pixels for diameter
        # Rough approximation: 1 degree latitude ≈ 111 km
        # Get pixel size in meters
        pixel_size = abs(self.transform_matrix[0])  # Assumes square pixels

        # Convert diameter to pixels
        radius_m = (diameter_km * 1000) / 2
        radius_pixels = int(radius_m / pixel_size)
        print(f"Pixel size: {pixel_size:.2f}m, AOI radius: {radius_pixels} pixels")

        # Calculate bounds
        row_start = max(0, center_row - radius_pixels)
        row_end = min(self.height, center_row + radius_pixels)
        col_start = max(0, center_col - radius_pixels)
        col_end = min(self.width, center_col + radius_pixels)

        bounds = (row_start, row_end, col_start, col_end)
        print(f"AOI bounds: rows [{row_start}:{row_end}], cols [{col_start}:{col_end}]")

        # Extract AOI
        aoi_image = self.image[:, row_start:row_end, col_start:col_end]
        print(f"AOI shape: {aoi_image.shape}")

        return aoi_image, bounds

    def unmix_linear(self, image_data, constrained=True):
        """
        Perform linear spectral unmixing using non-negative least squares.

        Parameters:
        -----------
        image_data : ndarray
            Hyperspectral image data (bands, rows, cols)
        constrained : bool
            If True, applies non-negativity and sum-to-one constraints

        Returns:
        --------
        abundance_maps : dict
            Dictionary of abundance maps for each mineral
        """
        print("Performing linear spectral unmixing...")

        n_bands, n_rows, n_cols = image_data.shape
        n_minerals = len(self.mineral_names)

        # Stack endmembers into matrix
        endmember_matrix = np.column_stack([
            self.endmembers[name] for name in self.mineral_names
        ])

        # Ensure endmember matrix has the same number of bands as image
        if endmember_matrix.shape[0] != n_bands:
            print(f"Warning: Endmember bands ({endmember_matrix.shape[0]}) != image bands ({n_bands})")
            min_bands = min(endmember_matrix.shape[0], n_bands)
            endmember_matrix = endmember_matrix[:min_bands, :]
            image_data = image_data[:min_bands, :, :]
            n_bands = min_bands

        print(f"Endmember matrix shape: {endmember_matrix.shape}")

        # Reshape image for unmixing
        pixels = image_data.reshape(n_bands, -1).T  # (n_pixels, n_bands)

        # Initialize abundance array
        abundances = np.zeros((pixels.shape[0], n_minerals))

        # Unmix each pixel
        print("Unmixing pixels...")
        for i in range(pixels.shape[0]):
            if i % 10000 == 0:
                print(f"  Progress: {i}/{pixels.shape[0]} pixels")

            pixel = pixels[i]

            # Skip invalid pixels (zeros or NaNs)
            if np.any(np.isnan(pixel)) or np.all(pixel == 0):
                continue

            if constrained:
                # Non-negative least squares
                abundances[i], _ = nnls(endmember_matrix, pixel)
            else:
                # Ordinary least squares
                abundances[i] = np.linalg.lstsq(endmember_matrix, pixel, rcond=None)[0]

        # Normalize abundances to sum to 1 (if constrained)
        if constrained:
            row_sums = abundances.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            abundances = abundances / row_sums

        # Reshape back to image dimensions
        abundance_maps = {}
        for i, name in enumerate(self.mineral_names):
            abundance_map = abundances[:, i].reshape(n_rows, n_cols)
            abundance_maps[name] = abundance_map
            print(f"{name} abundance range: [{abundance_map.min():.4f}, {abundance_map.max():.4f}]")

        return abundance_maps

    def classify_hard(self, abundance_maps):
        """
        Create hard classification map using winner-take-all from abundance maps.

        Parameters:
        -----------
        abundance_maps : dict
            Dictionary of abundance maps for each mineral

        Returns:
        --------
        classification : ndarray
            Classification map (0=background, 1-N=minerals)
        class_names : list
            List of class names (including background)
        """
        print("Creating hard classification map...")

        # Stack abundance maps
        abundance_stack = np.stack([
            abundance_maps[name] for name in self.mineral_names
        ], axis=0)

        # Winner-take-all classification
        classification = np.argmax(abundance_stack, axis=0) + 1  # +1 to start from 1

        # Set background where all abundances are very low
        abundance_sum = abundance_stack.sum(axis=0)
        classification[abundance_sum < 0.1] = 0  # Background threshold

        class_names = ['Background'] + self.mineral_names

        print(f"Classification classes: {class_names}")
        for i, name in enumerate(class_names):
            count = np.sum(classification == i)
            percentage = (count / classification.size) * 100
            print(f"  {name}: {count} pixels ({percentage:.2f}%)")

        return classification, class_names

    def create_rgb_composite(self, image_data, red_band=50, green_band=30, blue_band=15):
        """
        Create RGB composite from hyperspectral image.

        Parameters:
        -----------
        image_data : ndarray
            Hyperspectral image data (bands, rows, cols)
        red_band, green_band, blue_band : int
            Band indices for RGB

        Returns:
        --------
        rgb : ndarray
            RGB image (rows, cols, 3)
        """
        n_bands = image_data.shape[0]

        # Adjust band indices if out of range
        red_band = min(red_band, n_bands - 1)
        green_band = min(green_band, n_bands - 1)
        blue_band = min(blue_band, n_bands - 1)

        print(f"Creating RGB composite using bands R={red_band}, G={green_band}, B={blue_band}")

        rgb = np.stack([
            image_data[red_band],
            image_data[green_band],
            image_data[blue_band]
        ], axis=-1)

        # Normalize to 0-1
        rgb = self._normalize_rgb(rgb)

        return rgb

    def _normalize_rgb(self, rgb, percentile_clip=2):
        """Normalize RGB image with percentile clipping."""
        rgb_norm = np.zeros_like(rgb, dtype=np.float32)

        for i in range(3):
            band = rgb[:, :, i]
            vmin, vmax = np.percentile(band[band > 0], [percentile_clip, 100 - percentile_clip])
            band_norm = (band - vmin) / (vmax - vmin)
            band_norm = np.clip(band_norm, 0, 1)
            rgb_norm[:, :, i] = band_norm

        return rgb_norm

    def visualize_results(self, full_image, aoi_image, aoi_bounds, abundance_maps,
                         classification, class_names, output_dir='outputs'):
        """
        Create all visualization outputs.

        Parameters:
        -----------
        full_image : ndarray
            Full hyperspectral image
        aoi_image : ndarray
            AOI subset
        aoi_bounds : tuple
            (row_start, row_end, col_start, col_end)
        abundance_maps : dict
            Abundance maps for each mineral
        classification : ndarray
            Classification map
        class_names : list
            Class names
        output_dir : str
            Output directory for PNG files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        print(f"Saving outputs to: {output_path}")

        # 1. RGB with AOI box
        print("Creating RGB with AOI box...")
        rgb_full = self.create_rgb_composite(full_image)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(rgb_full)

        # Draw AOI box
        row_start, row_end, col_start, col_end = aoi_bounds
        rect = plt.Rectangle((col_start, row_start),
                            col_end - col_start,
                            row_end - row_start,
                            fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        ax.set_title('RGB Composite with AOI', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path / '1_rgb_with_aoi.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path / '1_rgb_with_aoi.png'}")

        # 2. Abundance maps
        print("Creating abundance maps...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for i, (name, abundance) in enumerate(abundance_maps.items()):
            im = axes[i].imshow(abundance, cmap='hot', vmin=0, vmax=1)
            axes[i].set_title(f'{name.capitalize()} Abundance', fontsize=12, fontweight='bold')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(output_path / '2_abundance_maps.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path / '2_abundance_maps.png'}")

        # 3. Individual abundance maps
        for name, abundance in abundance_maps.items():
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(abundance, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'{name.capitalize()} Abundance', fontsize=14, fontweight='bold')
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Abundance', fontsize=11)
            plt.tight_layout()
            filename = f'2_{name}_abundance.png'
            plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_path / filename}")

        # 4. Hard classification map
        print("Creating classification map...")

        # Create colormap
        colors = ['black', 'yellow', 'red', 'orange']  # Background, jarosite, hematite, goethite
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors[:len(class_names)])

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(classification, cmap=cmap, vmin=0, vmax=len(class_names)-1)
        ax.set_title('Hard Classification Map', fontsize=14, fontweight='bold')
        ax.axis('off')

        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], label=class_names[i])
                          for i in range(len(class_names))]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path / '3_classification_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path / '3_classification_map.png'}")

        print("All visualizations complete!")


def main():
    """Main processing function."""

    # Configuration
    HSI_PATH = "/Users/jeremy/Desktop/ff3-1101/FF03_20250707_00501045_0000001101_L2A.tif"
    HDR_PATH = "/Users/jeremy/Desktop/ff3-1101/FF03_20250707_00501045_0000001101_L2A.tif.hdr"

    SPECTRAL_LIBRARIES = {
        'jarosite': '/Users/jeremy/Desktop/jarosite_convolved_to_sensor.csv',
        'hematite': '/Users/jeremy/Desktop/hematite_convolved_to_sensor.csv',
        'goethite': '/Users/jeremy/Desktop/goethite_convolved_to_sensor.csv'
    }

    AOI_CENTER_LAT = -18.91386
    AOI_CENTER_LON = 128.82889
    AOI_DIAMETER_KM = 30

    OUTPUT_DIR = '/Users/jeremy/Desktop/ff3-1101/'

    print("="*80)
    print("HYPERSPECTRAL MINERAL CLASSIFICATION AND ABUNDANCE MAPPING")
    print("="*80)
    print()

    # Initialize classifier
    classifier = HyperspectralMineralClassifier(HSI_PATH, HDR_PATH)

    # Load hyperspectral image
    full_image = classifier.load_image()
    print()

    # Load spectral libraries
    for mineral_name, library_path in SPECTRAL_LIBRARIES.items():
        classifier.load_spectral_library(library_path, mineral_name)
    print()

    # Extract AOI
    aoi_image, aoi_bounds = classifier.extract_aoi(
        AOI_CENTER_LAT,
        AOI_CENTER_LON,
        AOI_DIAMETER_KM
    )
    print()

    # Perform unmixing
    abundance_maps = classifier.unmix_linear(aoi_image, constrained=True)
    print()

    # Create hard classification
    classification, class_names = classifier.classify_hard(abundance_maps)
    print()

    # Visualize results
    classifier.visualize_results(
        full_image,
        aoi_image,
        aoi_bounds,
        abundance_maps,
        classification,
        class_names,
        OUTPUT_DIR
    )

    print()
    print("="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print(f"Outputs saved to: {OUTPUT_DIR}/")
    print("  1. 1_rgb_with_aoi.png - RGB composite with AOI box")
    print("  2. 2_abundance_maps.png - All abundance maps combined")
    print("  3. 2_<mineral>_abundance.png - Individual abundance maps")
    print("  4. 3_classification_map.png - Hard classification map")


if __name__ == "__main__":
    main()
