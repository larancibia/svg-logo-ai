"""
Behavior Characterization for MAP-Elites Logo Generation
=========================================================
Extracts behavioral features from SVG logos to map them into a multi-dimensional grid.

Behavior Dimensions:
1. Complexity: Number of SVG elements (10-15, 15-20, 20-25, 25-30, 30-35, 35-40, 40-45, 45-50, 50-55, 55+)
2. Style: Geometric vs Organic (0=pure geometric, 1=pure organic)
3. Symmetry: Asymmetric vs Symmetric (0=no symmetry, 1=perfect symmetry)
4. Color Richness: Monochrome vs Polychromatic (0=1 color, 1=many colors)
"""

import re
import math
from typing import Dict, Tuple, List
from xml.etree import ElementTree as ET
import numpy as np


class BehaviorCharacterizer:
    """Extracts behavioral features from SVG logos"""

    def __init__(self, num_bins: int = 10):
        """
        Initialize behavior characterizer

        Args:
            num_bins: Number of bins per dimension (default: 10 for 10x10x10x10 grid)
        """
        self.num_bins = num_bins

    def characterize(self, svg_code: str) -> Dict:
        """
        Compute all behavioral features for an SVG logo

        Args:
            svg_code: SVG code string

        Returns:
            Dict with:
                - raw_scores: {complexity, style, symmetry, color_richness}
                - bins: (complexity_bin, style_bin, symmetry_bin, color_bin)
                - details: Additional diagnostic info
        """
        complexity = self.compute_complexity(svg_code)
        style_score = self.compute_style_score(svg_code)
        symmetry_score = self.compute_symmetry_score(svg_code)
        color_richness = self.compute_color_richness(svg_code)

        # Discretize to bins
        bins = self.discretize_to_bins(complexity, style_score, symmetry_score, color_richness)

        return {
            'raw_scores': {
                'complexity': complexity,
                'style': style_score,
                'symmetry': symmetry_score,
                'color_richness': color_richness
            },
            'bins': bins,
            'details': {
                'complexity_rating': self._complexity_rating(complexity),
                'style_rating': self._style_rating(style_score),
                'symmetry_rating': self._symmetry_rating(symmetry_score),
                'color_rating': self._color_rating(color_richness)
            }
        }

    def compute_complexity(self, svg_code: str) -> int:
        """
        Count number of SVG geometric elements

        Returns:
            Integer count of elements (path, circle, rect, ellipse, polygon, line, polyline)
        """
        try:
            root = ET.fromstring(svg_code)

            geometric_tags = ['path', 'circle', 'rect', 'ellipse', 'polygon', 'line', 'polyline']
            count = 0

            for elem in root.iter():
                tag = elem.tag.split('}')[-1]  # Remove namespace
                if tag in geometric_tags:
                    count += 1

            return count

        except Exception as e:
            print(f"Warning: Error parsing SVG for complexity: {e}")
            return 0

    def compute_style_score(self, svg_code: str) -> float:
        """
        Measure geometric vs organic style

        0.0 = Pure geometric (only straight lines, circles, rectangles)
        1.0 = Pure organic (curves, bezier paths, complex shapes)

        Returns:
            Float between 0.0 and 1.0
        """
        try:
            root = ET.fromstring(svg_code)

            geometric_count = 0  # rect, line, polygon with straight edges
            organic_count = 0    # path with curves, ellipse

            for elem in root.iter():
                tag = elem.tag.split('}')[-1]

                # Pure geometric shapes
                if tag in ['rect', 'line', 'polygon']:
                    geometric_count += 1
                elif tag == 'circle':
                    geometric_count += 0.5  # Circles are somewhat geometric

                # Organic shapes
                elif tag == 'ellipse':
                    organic_count += 0.5
                elif tag == 'path':
                    d = elem.get('d', '')
                    # Check for curves (C, Q, S, T commands)
                    if re.search(r'[CcQqSsTt]', d):
                        organic_count += 1
                    else:
                        geometric_count += 0.5  # Path with only lines
                elif tag == 'polyline':
                    # Polyline is somewhat organic
                    organic_count += 0.3
                    geometric_count += 0.3

            total = geometric_count + organic_count
            if total == 0:
                return 0.5  # Neutral if no shapes

            # Normalize to 0-1
            style_score = organic_count / total
            return float(np.clip(style_score, 0.0, 1.0))

        except Exception as e:
            print(f"Warning: Error computing style score: {e}")
            return 0.5

    def compute_symmetry_score(self, svg_code: str) -> float:
        """
        Detect symmetry in the logo

        0.0 = Completely asymmetric
        1.0 = Perfect symmetry (rotational or reflective)

        Method: Extract element positions and check for mirroring/rotation patterns

        Returns:
            Float between 0.0 and 1.0
        """
        try:
            root = ET.fromstring(svg_code)

            # Get viewBox to understand coordinate system
            viewbox = root.get('viewBox')
            if viewbox:
                _, _, width, height = map(float, viewbox.split())
                center_x = width / 2
                center_y = height / 2
            else:
                center_x = 100
                center_y = 100

            # Extract positions of all elements
            positions = []

            for elem in root.iter():
                tag = elem.tag.split('}')[-1]

                if tag == 'circle':
                    cx = float(elem.get('cx', center_x))
                    cy = float(elem.get('cy', center_y))
                    positions.append((cx, cy))

                elif tag == 'rect':
                    x = float(elem.get('x', 0))
                    y = float(elem.get('y', 0))
                    w = float(elem.get('width', 0))
                    h = float(elem.get('height', 0))
                    # Use center of rectangle
                    positions.append((x + w/2, y + h/2))

                elif tag == 'ellipse':
                    cx = float(elem.get('cx', center_x))
                    cy = float(elem.get('cy', center_y))
                    positions.append((cx, cy))

            if len(positions) < 2:
                return 0.5  # Not enough elements to determine symmetry

            # Check for vertical reflection symmetry (most common in logos)
            symmetry_score = self._check_vertical_symmetry(positions, center_x)

            # Also check horizontal symmetry
            horizontal_score = self._check_horizontal_symmetry(positions, center_y)

            # Return the maximum symmetry found
            return float(max(symmetry_score, horizontal_score))

        except Exception as e:
            print(f"Warning: Error computing symmetry: {e}")
            return 0.5

    def _check_vertical_symmetry(self, positions: List[Tuple[float, float]], center_x: float) -> float:
        """Check for vertical reflection symmetry around center_x"""
        if not positions:
            return 0.0

        tolerance = 5.0  # Pixel tolerance for matching
        matched = 0
        total = len(positions)

        for x, y in positions:
            # Calculate mirrored position
            mirror_x = 2 * center_x - x

            # Check if there's a matching element
            for mx, my in positions:
                if abs(mx - mirror_x) < tolerance and abs(my - y) < tolerance:
                    matched += 1
                    break

        return matched / total if total > 0 else 0.0

    def _check_horizontal_symmetry(self, positions: List[Tuple[float, float]], center_y: float) -> float:
        """Check for horizontal reflection symmetry around center_y"""
        if not positions:
            return 0.0

        tolerance = 5.0
        matched = 0
        total = len(positions)

        for x, y in positions:
            mirror_y = 2 * center_y - y

            for mx, my in positions:
                if abs(mx - x) < tolerance and abs(my - mirror_y) < tolerance:
                    matched += 1
                    break

        return matched / total if total > 0 else 0.0

    def compute_color_richness(self, svg_code: str) -> float:
        """
        Measure color richness

        0.0 = Monochrome (1 color)
        1.0 = Polychromatic (many colors)

        Returns:
            Float between 0.0 and 1.0
        """
        try:
            # Extract all hex colors
            hex_pattern = r'#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})'
            matches = set(re.findall(hex_pattern, svg_code))

            # Remove white/black (they don't count toward "color richness")
            colors = set()
            for match in matches:
                normalized = match.lower()
                # Normalize 3-digit to 6-digit
                if len(normalized) == 3:
                    normalized = ''.join([c*2 for c in normalized])

                # Skip white and very light colors
                if normalized not in ['ffffff', 'fefefe', 'fdfdfd']:
                    colors.add(normalized)

            num_colors = len(colors)

            # Map to 0-1 scale
            # 1 color -> 0.0
            # 2 colors -> 0.25
            # 3 colors -> 0.5
            # 4 colors -> 0.75
            # 5+ colors -> 1.0
            if num_colors <= 1:
                return 0.0
            elif num_colors == 2:
                return 0.25
            elif num_colors == 3:
                return 0.5
            elif num_colors == 4:
                return 0.75
            else:
                return 1.0

        except Exception as e:
            print(f"Warning: Error computing color richness: {e}")
            return 0.0

    def discretize_to_bins(self, complexity: int, style: float, symmetry: float, color_richness: float) -> Tuple[int, int, int, int]:
        """
        Convert continuous/discrete features to bin indices

        Args:
            complexity: Integer count of elements
            style: Float 0-1 (geometric to organic)
            symmetry: Float 0-1 (asymmetric to symmetric)
            color_richness: Float 0-1 (mono to polychromatic)

        Returns:
            (complexity_bin, style_bin, symmetry_bin, color_bin) all in [0, num_bins-1]
        """
        # Complexity bins: 10-15, 15-20, 20-25, 25-30, 30-35, 35-40, 40-45, 45-50, 50-55, 55+
        complexity_bin = min(self.num_bins - 1, max(0, (complexity - 10) // 5))

        # Style/Symmetry/Color: continuous 0-1 -> bins
        style_bin = min(self.num_bins - 1, int(style * self.num_bins))
        symmetry_bin = min(self.num_bins - 1, int(symmetry * self.num_bins))
        color_bin = min(self.num_bins - 1, int(color_richness * self.num_bins))

        return (complexity_bin, style_bin, symmetry_bin, color_bin)

    def _complexity_rating(self, complexity: int) -> str:
        """Human-readable complexity rating"""
        if complexity < 15:
            return "very_simple"
        elif complexity < 25:
            return "simple"
        elif complexity < 35:
            return "moderate"
        elif complexity < 45:
            return "complex"
        else:
            return "very_complex"

    def _style_rating(self, style: float) -> str:
        """Human-readable style rating"""
        if style < 0.3:
            return "geometric"
        elif style < 0.7:
            return "mixed"
        else:
            return "organic"

    def _symmetry_rating(self, symmetry: float) -> str:
        """Human-readable symmetry rating"""
        if symmetry < 0.3:
            return "asymmetric"
        elif symmetry < 0.7:
            return "partial_symmetry"
        else:
            return "symmetric"

    def _color_rating(self, color_richness: float) -> str:
        """Human-readable color rating"""
        if color_richness < 0.2:
            return "monochrome"
        elif color_richness < 0.5:
            return "duotone"
        elif color_richness < 0.8:
            return "tritone"
        else:
            return "polychromatic"


def demo():
    """Demo behavior characterization"""
    characterizer = BehaviorCharacterizer(num_bins=10)

    # Example 1: Simple geometric logo
    svg1 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="60" fill="#2563eb"/>
  <rect x="80" y="80" width="40" height="40" fill="#ffffff"/>
</svg>"""

    print("="*60)
    print("Example 1: Simple Geometric Logo")
    print("="*60)
    result1 = characterizer.characterize(svg1)
    print(f"Complexity: {result1['raw_scores']['complexity']} ({result1['details']['complexity_rating']})")
    print(f"Style: {result1['raw_scores']['style']:.2f} ({result1['details']['style_rating']})")
    print(f"Symmetry: {result1['raw_scores']['symmetry']:.2f} ({result1['details']['symmetry_rating']})")
    print(f"Color Richness: {result1['raw_scores']['color_richness']:.2f} ({result1['details']['color_rating']})")
    print(f"Bins: {result1['bins']}")

    # Example 2: Complex organic logo
    svg2 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <path d="M50 100 Q75 50 100 100 T150 100" fill="none" stroke="#e74c3c" stroke-width="4"/>
  <path d="M60 120 Q85 140 110 120 T160 120" fill="none" stroke="#3498db" stroke-width="4"/>
  <ellipse cx="50" cy="100" rx="10" ry="15" fill="#f39c12"/>
  <ellipse cx="150" cy="100" rx="10" ry="15" fill="#f39c12"/>
  <circle cx="100" cy="80" r="5" fill="#2ecc71"/>
</svg>"""

    print("\n" + "="*60)
    print("Example 2: Complex Organic Logo")
    print("="*60)
    result2 = characterizer.characterize(svg2)
    print(f"Complexity: {result2['raw_scores']['complexity']} ({result2['details']['complexity_rating']})")
    print(f"Style: {result2['raw_scores']['style']:.2f} ({result2['details']['style_rating']})")
    print(f"Symmetry: {result2['raw_scores']['symmetry']:.2f} ({result2['details']['symmetry_rating']})")
    print(f"Color Richness: {result2['raw_scores']['color_richness']:.2f} ({result2['details']['color_rating']})")
    print(f"Bins: {result2['bins']}")


if __name__ == "__main__":
    demo()
