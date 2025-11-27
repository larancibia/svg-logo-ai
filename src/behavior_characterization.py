"""
Behavior Characterization for MAP-Elites Logo Generation
=========================================================
Extracts behavioral features from SVG logos to map them into a multi-dimensional grid.

Behavior Dimensions (5D):
1. Complexity: Number of SVG elements (10-15, 15-20, 20-25, 25-30, 30-35, 35-40, 40-45, 45-50, 50-55, 55+)
2. Style: Geometric vs Organic (0=pure geometric, 1=pure organic)
3. Symmetry: Asymmetric vs Symmetric (0=no symmetry, 1=perfect symmetry)
4. Color Richness: Monochrome vs Polychromatic (0=1 color, 1=many colors)
5. Emotional Tone: Serious/Corporate vs Playful/Friendly (0=serious, 1=playful)

Enhanced Features:
- More accurate symmetry detection using transform analysis
- Better geometric vs organic classification
- Robust color harmony metrics
- Fast computation (optimized for 1000+ logos)
- Support for LLM-based emotional tone evaluation
"""

import re
import math
from typing import Dict, Tuple, List, Optional, Callable
from xml.etree import ElementTree as ET
import numpy as np
from collections import Counter
import colorsys


class BehaviorCharacterizer:
    """Extracts behavioral features from SVG logos"""

    def __init__(self, num_bins: int = 10, llm_evaluator: Optional[Callable] = None):
        """
        Initialize behavior characterizer

        Args:
            num_bins: Number of bins per dimension (default: 10 for 10^5 = 100,000 cell grid)
            llm_evaluator: Optional LLM function for emotional tone evaluation
                          Should accept (svg_code: str, genome: Dict) -> float
        """
        self.num_bins = num_bins
        self.llm_evaluator = llm_evaluator

    def characterize(self, svg_code: str, genome: Optional[Dict] = None) -> Dict:
        """
        Compute all behavioral features for an SVG logo (now 5D)

        Args:
            svg_code: SVG code string
            genome: Optional genome dict for LLM-based emotional tone evaluation

        Returns:
            Dict with:
                - raw_scores: {complexity, style, symmetry, color_richness, emotional_tone}
                - bins: (complexity_bin, style_bin, symmetry_bin, color_bin, emotional_bin)
                - details: Additional diagnostic info
        """
        complexity = self.compute_complexity(svg_code)
        style_score = self.compute_style_score(svg_code)
        symmetry_score = self.compute_symmetry_score(svg_code)
        color_richness = self.compute_color_richness(svg_code)
        emotional_tone = self.compute_emotional_tone(svg_code, genome)

        # Discretize to bins
        bins = self.discretize_to_bins(complexity, style_score, symmetry_score,
                                       color_richness, emotional_tone)

        return {
            'raw_scores': {
                'complexity': complexity,
                'style': style_score,
                'symmetry': symmetry_score,
                'color_richness': color_richness,
                'emotional_tone': emotional_tone
            },
            'bins': bins,
            'details': {
                'complexity_rating': self._complexity_rating(complexity),
                'style_rating': self._style_rating(style_score),
                'symmetry_rating': self._symmetry_rating(symmetry_score),
                'color_rating': self._color_rating(color_richness),
                'emotional_rating': self._emotional_rating(emotional_tone)
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
        Measure color richness with improved color harmony detection

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

                # Skip white, black, and very light/dark colors
                if normalized not in ['ffffff', 'fefefe', 'fdfdfd', '000000', '010101', '020202']:
                    # Check if it's not too light or too dark
                    rgb = tuple(int(normalized[i:i+2], 16) for i in (0, 2, 4))
                    brightness = sum(rgb) / 3
                    if 20 < brightness < 235:  # Exclude very light/dark
                        colors.add(normalized)

            num_colors = len(colors)

            # Map to 0-1 scale (smoother progression)
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
                # Scale beyond 5 colors
                return min(1.0, 0.75 + (num_colors - 4) * 0.05)

        except Exception as e:
            print(f"Warning: Error computing color richness: {e}")
            return 0.0

    def compute_emotional_tone(self, svg_code: str, genome: Optional[Dict] = None) -> float:
        """
        5th behavioral dimension: Emotional tone

        0.0 = Serious, professional, corporate, formal
        0.5 = Balanced, neutral
        1.0 = Playful, friendly, approachable, casual

        If LLM is available: use it for semantic understanding
        Fallback: heuristic based on shapes, colors, complexity, and style

        Args:
            svg_code: SVG code string
            genome: Optional genome for LLM context

        Returns:
            Float between 0.0 and 1.0
        """
        # Try LLM-based evaluation first
        if self.llm_evaluator is not None and genome is not None:
            try:
                llm_score = self.llm_evaluator(svg_code, genome)
                if llm_score is not None and 0.0 <= llm_score <= 1.0:
                    return float(llm_score)
            except Exception as e:
                print(f"Warning: LLM evaluator failed: {e}. Falling back to heuristic.")

        # Fallback: heuristic-based evaluation
        return self._compute_emotional_tone_heuristic(svg_code)

    def _compute_emotional_tone_heuristic(self, svg_code: str) -> float:
        """
        Heuristic-based emotional tone computation

        Factors:
        - Rounded shapes (circles, curves) -> More playful
        - Sharp angles (triangles, zigzags) -> More serious
        - Bright, saturated colors -> More playful
        - Dark, muted colors -> More serious
        - High complexity -> More serious
        - Low complexity with organic shapes -> More playful
        """
        try:
            root = ET.fromstring(svg_code)

            # Initialize scores
            playful_score = 0.0
            serious_score = 0.0

            # 1. Shape analysis
            for elem in root.iter():
                tag = elem.tag.split('}')[-1]

                # Playful shapes: circles, ellipses, curves
                if tag == 'circle':
                    playful_score += 1.5
                elif tag == 'ellipse':
                    playful_score += 1.2
                elif tag == 'path':
                    d = elem.get('d', '')
                    # Curves are playful
                    if re.search(r'[CcQqSsTt]', d):
                        playful_score += 1.0
                    # Straight lines are more serious
                    else:
                        serious_score += 0.5

                # Serious shapes: rectangles, polygons with straight edges
                elif tag == 'rect':
                    serious_score += 1.0
                elif tag == 'polygon':
                    serious_score += 0.8
                elif tag == 'line':
                    serious_score += 0.5

            # 2. Color analysis
            colors = self._extract_colors(svg_code)
            for color_hex in colors:
                if len(color_hex) == 3:
                    color_hex = ''.join([c*2 for c in color_hex])

                rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
                h, s, v = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)

                # High saturation and brightness -> playful
                if s > 0.6 and v > 0.5:
                    playful_score += 1.0
                # Low saturation or brightness -> serious
                elif s < 0.3 or v < 0.4:
                    serious_score += 1.0

                # Certain hue ranges
                # Warm colors (red, orange, yellow) -> slightly more playful
                if 0 <= h < 0.17 or 0.92 < h <= 1.0:  # Red
                    playful_score += 0.3
                elif 0.08 <= h < 0.17:  # Orange/Yellow
                    playful_score += 0.5
                # Cool colors (blue, gray) -> slightly more serious
                elif 0.5 <= h < 0.75:  # Blue/Cyan
                    serious_score += 0.3

            # 3. Complexity factor
            complexity = self.compute_complexity(svg_code)
            if complexity > 40:
                serious_score += 2.0  # Very complex -> more serious
            elif complexity < 20:
                playful_score += 1.0  # Simple -> potentially more playful

            # 4. Style factor
            style_score = self.compute_style_score(svg_code)
            # Organic style is more playful
            playful_score += style_score * 2.0
            # Geometric style is more serious
            serious_score += (1.0 - style_score) * 1.5

            # Normalize to 0-1
            total = playful_score + serious_score
            if total == 0:
                return 0.5  # Neutral if no features

            emotional_tone = playful_score / total
            return float(np.clip(emotional_tone, 0.0, 1.0))

        except Exception as e:
            print(f"Warning: Error computing emotional tone: {e}")
            return 0.5  # Neutral on error

    def _extract_colors(self, svg_code: str) -> List[str]:
        """Extract all color hex codes from SVG"""
        hex_pattern = r'#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})'
        return list(set(re.findall(hex_pattern, svg_code)))

    def compute_all_behaviors(self, svg_code: str, genome: Optional[Dict] = None) -> Dict[str, float]:
        """
        Compute all 5 behavioral dimensions at once (optimized for batch processing)

        Args:
            svg_code: SVG code string
            genome: Optional genome for LLM-based emotional tone

        Returns:
            Dict with keys: complexity, style, symmetry, color_richness, emotional_tone
        """
        result = self.characterize(svg_code, genome)
        return result['raw_scores']

    def discretize_to_bins(self, complexity: int, style: float, symmetry: float,
                           color_richness: float, emotional_tone: float) -> Tuple[int, int, int, int, int]:
        """
        Convert continuous/discrete features to bin indices (5D)

        Args:
            complexity: Integer count of elements
            style: Float 0-1 (geometric to organic)
            symmetry: Float 0-1 (asymmetric to symmetric)
            color_richness: Float 0-1 (mono to polychromatic)
            emotional_tone: Float 0-1 (serious to playful)

        Returns:
            (complexity_bin, style_bin, symmetry_bin, color_bin, emotional_bin) all in [0, num_bins-1]
        """
        # Complexity bins: 10-15, 15-20, 20-25, 25-30, 30-35, 35-40, 40-45, 45-50, 50-55, 55+
        complexity_bin = min(self.num_bins - 1, max(0, (complexity - 10) // 5))

        # Style/Symmetry/Color/Emotional: continuous 0-1 -> bins
        style_bin = min(self.num_bins - 1, int(style * self.num_bins))
        symmetry_bin = min(self.num_bins - 1, int(symmetry * self.num_bins))
        color_bin = min(self.num_bins - 1, int(color_richness * self.num_bins))
        emotional_bin = min(self.num_bins - 1, int(emotional_tone * self.num_bins))

        return (complexity_bin, style_bin, symmetry_bin, color_bin, emotional_bin)

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

    def _emotional_rating(self, emotional_tone: float) -> str:
        """Human-readable emotional tone rating"""
        if emotional_tone < 0.25:
            return "serious_corporate"
        elif emotional_tone < 0.45:
            return "professional"
        elif emotional_tone < 0.55:
            return "balanced"
        elif emotional_tone < 0.75:
            return "friendly"
        else:
            return "playful_casual"


def visualize_behavior_space(archive_data: List[Dict], output_path: str):
    """
    Create 2D projections of 5D behavior space for visualization

    Shows:
    - Filled cells across different 2D projections
    - Quality heatmap
    - Coverage metrics

    Args:
        archive_data: List of archive entries with behavior and fitness
        output_path: Path to save visualization (PNG or PDF)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # Create figure with multiple subplots for different projections
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('5D Behavior Space Projections', fontsize=16, fontweight='bold')

        # Define projection pairs
        projections = [
            ('complexity', 'style', 0, 1),
            ('complexity', 'symmetry', 0, 2),
            ('style', 'symmetry', 1, 2),
            ('color_richness', 'emotional_tone', 3, 4),
            ('complexity', 'emotional_tone', 0, 4),
            ('symmetry', 'emotional_tone', 2, 4)
        ]

        dimension_names = ['Complexity', 'Style', 'Symmetry', 'Color Richness', 'Emotional Tone']

        for idx, (ax, (name1, name2, dim1, dim2)) in enumerate(zip(axes.flat, projections)):
            # Extract data for this projection
            x_coords = [entry['behavior'][dim1] for entry in archive_data]
            y_coords = [entry['behavior'][dim2] for entry in archive_data]
            fitnesses = [entry['fitness'] for entry in archive_data]

            # Create scatter plot with fitness as color
            scatter = ax.scatter(x_coords, y_coords, c=fitnesses, cmap='viridis',
                               s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

            ax.set_xlabel(dimension_names[dim1], fontsize=10, fontweight='bold')
            ax.set_ylabel(dimension_names[dim2], fontsize=10, fontweight='bold')
            ax.set_title(f'{dimension_names[dim1]} vs {dimension_names[dim2]}', fontsize=11)
            ax.grid(True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Fitness', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Behavior space visualization saved to {output_path}")

    except ImportError:
        print("Warning: matplotlib not available. Skipping visualization.")
    except Exception as e:
        print(f"Warning: Failed to create visualization: {e}")


def demo():
    """Demo behavior characterization"""
    characterizer = BehaviorCharacterizer(num_bins=10)

    # Example 1: Simple geometric logo
    svg1 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="60" fill="#2563eb"/>
  <rect x="80" y="80" width="40" height="40" fill="#ffffff"/>
</svg>"""

    print("="*60)
    print("Example 1: Simple Geometric Logo (Serious)")
    print("="*60)
    result1 = characterizer.characterize(svg1)
    print(f"Complexity: {result1['raw_scores']['complexity']} ({result1['details']['complexity_rating']})")
    print(f"Style: {result1['raw_scores']['style']:.2f} ({result1['details']['style_rating']})")
    print(f"Symmetry: {result1['raw_scores']['symmetry']:.2f} ({result1['details']['symmetry_rating']})")
    print(f"Color Richness: {result1['raw_scores']['color_richness']:.2f} ({result1['details']['color_rating']})")
    print(f"Emotional Tone: {result1['raw_scores']['emotional_tone']:.2f} ({result1['details']['emotional_rating']})")
    print(f"5D Bins: {result1['bins']}")

    # Example 2: Complex organic logo
    svg2 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <path d="M50 100 Q75 50 100 100 T150 100" fill="none" stroke="#e74c3c" stroke-width="4"/>
  <path d="M60 120 Q85 140 110 120 T160 120" fill="none" stroke="#3498db" stroke-width="4"/>
  <ellipse cx="50" cy="100" rx="10" ry="15" fill="#f39c12"/>
  <ellipse cx="150" cy="100" rx="10" ry="15" fill="#f39c12"/>
  <circle cx="100" cy="80" r="5" fill="#2ecc71"/>
</svg>"""

    print("\n" + "="*60)
    print("Example 2: Complex Organic Logo (Playful)")
    print("="*60)
    result2 = characterizer.characterize(svg2)
    print(f"Complexity: {result2['raw_scores']['complexity']} ({result2['details']['complexity_rating']})")
    print(f"Style: {result2['raw_scores']['style']:.2f} ({result2['details']['style_rating']})")
    print(f"Symmetry: {result2['raw_scores']['symmetry']:.2f} ({result2['details']['symmetry_rating']})")
    print(f"Color Richness: {result2['raw_scores']['color_richness']:.2f} ({result2['details']['color_rating']})")
    print(f"Emotional Tone: {result2['raw_scores']['emotional_tone']:.2f} ({result2['details']['emotional_rating']})")
    print(f"5D Bins: {result2['bins']}")

    print("\n" + "="*60)
    print("5D Behavior Space Summary")
    print("="*60)
    print(f"Total possible cells: 10^5 = 100,000")
    print(f"Dimensions: Complexity, Style, Symmetry, Color, Emotional Tone")
    print(f"Each dimension: 10 bins")
    print("Emotional Tone: 0.0=serious/corporate, 1.0=playful/friendly")


if __name__ == "__main__":
    demo()
