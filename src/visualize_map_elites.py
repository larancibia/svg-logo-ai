#!/usr/bin/env python3
"""
MAP-Elites Archive Visualization
=================================
Visualize the MAP-Elites archive to understand coverage and fitness distribution.

Visualizations:
1. 2D heatmaps for each pair of dimensions
2. Fitness distribution histogram
3. Coverage statistics over time
4. Behavioral diversity analysis
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class MAPElitesVisualizer:
    """Visualize MAP-Elites archive results"""

    def __init__(self, archive_path: str):
        """
        Initialize visualizer

        Args:
            archive_path: Path to archive.json
        """
        self.archive_path = Path(archive_path)

        # Load archive
        with open(self.archive_path, 'r') as f:
            self.data = json.load(f)

        self.dimensions = tuple(self.data['dimensions'])
        self.entries = self.data['entries']
        self.stats = self.data['statistics']

        print(f"Loaded archive with {len(self.entries)} logos")
        print(f"Grid dimensions: {self.dimensions}")
        print(f"Coverage: {self.stats['coverage']*100:.1f}%")

    def plot_2d_heatmaps(self, output_dir: str = None):
        """
        Create 2D heatmaps for each pair of behavioral dimensions

        Args:
            output_dir: Directory to save plots (default: same as archive)
        """
        if output_dir is None:
            output_dir = self.archive_path.parent

        dim_names = ['Complexity', 'Style', 'Symmetry', 'Color']

        # Create 2D projections for each pair
        pairs = [
            (0, 1, 'Complexity', 'Style'),
            (0, 2, 'Complexity', 'Symmetry'),
            (0, 3, 'Complexity', 'Color'),
            (1, 2, 'Style', 'Symmetry'),
            (1, 3, 'Style', 'Color'),
            (2, 3, 'Symmetry', 'Color')
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (dim1, dim2, name1, name2) in enumerate(pairs):
            ax = axes[idx]

            # Create grid for this 2D projection
            grid = np.zeros((self.dimensions[dim1], self.dimensions[dim2]))
            count_grid = np.zeros((self.dimensions[dim1], self.dimensions[dim2]))

            # Fill grid with max fitness values
            for entry in self.entries:
                behavior = entry['behavior']
                fitness = entry['fitness']
                i, j = behavior[dim1], behavior[dim2]

                # Keep max fitness
                if fitness > grid[i, j]:
                    grid[i, j] = fitness
                count_grid[i, j] += 1

            # Plot heatmap
            im = ax.imshow(grid.T, origin='lower', cmap='viridis', aspect='auto',
                          vmin=0, vmax=100, interpolation='nearest')

            # Add cell counts
            for i in range(self.dimensions[dim1]):
                for j in range(self.dimensions[dim2]):
                    if count_grid[i, j] > 0:
                        text_color = 'white' if grid[i, j] < 50 else 'black'
                        ax.text(i, j, f'{int(grid[i, j])}',
                               ha='center', va='center',
                               color=text_color, fontsize=8)

            ax.set_xlabel(name1, fontsize=12)
            ax.set_ylabel(name2, fontsize=12)
            ax.set_title(f'{name1} vs {name2}', fontsize=14)

            # Colorbar
            plt.colorbar(im, ax=ax, label='Fitness')

        plt.suptitle(f'MAP-Elites Archive Coverage\nTotal: {len(self.entries)} logos, '
                    f'Coverage: {self.stats["coverage"]*100:.1f}%',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = Path(output_dir) / 'map_elites_heatmaps.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmaps to: {output_path}")
        plt.close()

    def plot_fitness_distribution(self, output_dir: str = None):
        """Plot fitness distribution histogram"""
        if output_dir is None:
            output_dir = self.archive_path.parent

        fitnesses = [entry['fitness'] for entry in self.entries]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(fitnesses, bins=20, edgecolor='black', alpha=0.7, color='#2563eb')
        ax.axvline(self.stats['avg_fitness'], color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {self.stats["avg_fitness"]:.2f}')
        ax.axvline(self.stats['max_fitness'], color='green', linestyle='--',
                  linewidth=2, label=f'Max: {self.stats["max_fitness"]:.2f}')

        ax.set_xlabel('Fitness', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Fitness Distribution (n={len(fitnesses)})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        output_path = Path(output_dir) / 'fitness_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved fitness distribution to: {output_path}")
        plt.close()

    def plot_behavioral_space_3d(self, output_dir: str = None, dims=(0, 1, 2)):
        """
        Plot 3D scatter of behavioral space

        Args:
            output_dir: Output directory
            dims: Which 3 dimensions to plot (default: complexity, style, symmetry)
        """
        if output_dir is None:
            output_dir = self.archive_path.parent

        from mpl_toolkits.mplot3d import Axes3D

        dim_names = ['Complexity', 'Style', 'Symmetry', 'Color']

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        behaviors = [entry['behavior'] for entry in self.entries]
        fitnesses = [entry['fitness'] for entry in self.entries]

        x = [b[dims[0]] for b in behaviors]
        y = [b[dims[1]] for b in behaviors]
        z = [b[dims[2]] for b in behaviors]

        scatter = ax.scatter(x, y, z, c=fitnesses, cmap='viridis',
                           s=100, alpha=0.6, edgecolors='black', linewidth=0.5,
                           vmin=0, vmax=100)

        ax.set_xlabel(dim_names[dims[0]], fontsize=12)
        ax.set_ylabel(dim_names[dims[1]], fontsize=12)
        ax.set_zlabel(dim_names[dims[2]], fontsize=12)
        ax.set_title(f'MAP-Elites Behavioral Space\n{dim_names[dims[0]]} × {dim_names[dims[1]]} × {dim_names[dims[2]]}',
                    fontsize=14, fontweight='bold')

        plt.colorbar(scatter, ax=ax, label='Fitness', pad=0.1)

        output_path = Path(output_dir) / 'behavioral_space_3d.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D scatter to: {output_path}")
        plt.close()

    def plot_statistics_summary(self, output_dir: str = None):
        """Create comprehensive statistics summary"""
        if output_dir is None:
            output_dir = self.archive_path.parent

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Coverage by dimension
        ax = axes[0, 0]
        dim_names = ['Complexity', 'Style', 'Symmetry', 'Color']
        dim_coverage = []

        for dim in range(4):
            unique_bins = len(set(entry['behavior'][dim] for entry in self.entries))
            coverage = unique_bins / self.dimensions[dim]
            dim_coverage.append(coverage * 100)

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        bars = ax.bar(dim_names, dim_coverage, color=colors, edgecolor='black', alpha=0.7)
        ax.set_ylabel('Coverage (%)', fontsize=11)
        ax.set_title('Coverage by Dimension', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, dim_coverage):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

        # 2. Generation distribution
        ax = axes[0, 1]
        generations = [entry['generation'] for entry in self.entries]
        gen_counts = {}
        for gen in generations:
            gen_counts[gen] = gen_counts.get(gen, 0) + 1

        gen_nums = sorted(gen_counts.keys())
        counts = [gen_counts[g] for g in gen_nums]

        ax.bar(gen_nums, counts, color='#9b59b6', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Generation', fontsize=11)
        ax.set_ylabel('Number of Logos', fontsize=11)
        ax.set_title('Logos by Generation', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # 3. Fitness vs complexity
        ax = axes[1, 0]
        complexities = [entry['raw_behavior']['complexity'] for entry in self.entries]
        fitnesses = [entry['fitness'] for entry in self.entries]

        ax.scatter(complexities, fitnesses, alpha=0.6, s=50, color='#2563eb', edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Complexity (# elements)', fontsize=11)
        ax.set_ylabel('Fitness', fontsize=11)
        ax.set_title('Fitness vs Complexity', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

        # Add trend line
        z = np.polyfit(complexities, fitnesses, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(complexities), max(complexities), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        ax.legend()

        # 4. Key statistics table
        ax = axes[1, 1]
        ax.axis('off')

        stats_text = f"""
        ARCHIVE STATISTICS
        {'='*40}

        Total Logos:        {len(self.entries)}
        Grid Dimensions:    {' × '.join(map(str, self.dimensions))}
        Total Cells:        {np.prod(self.dimensions):,}
        Coverage:           {self.stats['coverage']*100:.2f}%

        FITNESS METRICS
        {'='*40}

        Average:            {self.stats['avg_fitness']:.2f}
        Maximum:            {self.stats['max_fitness']:.2f}
        Minimum:            {self.stats['min_fitness']:.2f}
        Std Dev:            {np.std(fitnesses):.2f}

        BEHAVIORAL DIVERSITY
        {'='*40}

        Unique Behaviors:   {len(set(tuple(e['behavior']) for e in self.entries))}
        Generations:        {min(generations)} - {max(generations)}
        """

        ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
               verticalalignment='center', transform=ax.transAxes)

        plt.tight_layout()
        output_path = Path(output_dir) / 'statistics_summary.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved statistics summary to: {output_path}")
        plt.close()

    def generate_all_visualizations(self, output_dir: str = None):
        """Generate all visualizations"""
        if output_dir is None:
            output_dir = self.archive_path.parent

        print("\nGenerating visualizations...")
        print("-" * 60)

        self.plot_2d_heatmaps(output_dir)
        self.plot_fitness_distribution(output_dir)
        self.plot_behavioral_space_3d(output_dir)
        self.plot_statistics_summary(output_dir)

        print("-" * 60)
        print(f"All visualizations saved to: {output_dir}")


def main():
    """CLI for visualization"""
    parser = argparse.ArgumentParser(description='Visualize MAP-Elites archive')
    parser.add_argument('archive_path', type=str, help='Path to archive.json')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same as archive)')

    args = parser.parse_args()

    # Create visualizer
    visualizer = MAPElitesVisualizer(args.archive_path)

    # Generate all visualizations
    visualizer.generate_all_visualizations(args.output_dir)


if __name__ == "__main__":
    main()
