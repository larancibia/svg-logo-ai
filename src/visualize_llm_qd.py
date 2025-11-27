#!/usr/bin/env python3
"""
Visualization System for LLM-QD Experiments
============================================
Creates publication-ready visualizations and charts.

Visualizations:
1. Coverage over time (line chart)
2. Fitness distribution (histogram)
3. Behavior space heatmap (2D projections of 4D space)
4. Convergence curves (comparison across methods)
5. Quality-Diversity scatter plots
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# For better plots
plt.style.use('seaborn-v0_8-darkgrid')


class LLMQDVisualizer:
    """
    Visualization system for LLM-QD experiments
    """

    def __init__(self, results_dir: str):
        """
        Initialize visualizer

        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "visualizations"
        self.output_dir.mkdir(exist_ok=True)

        # Load data
        self.aggregate_results = {}
        self.archive_data = {}
        self.history_data = []

        self._load_data()

    def _load_data(self):
        """Load all experiment data"""
        print(f"Loading data from {self.results_dir}")

        # Load aggregate results
        aggregate_file = self.results_dir / "aggregate_results.json"
        if aggregate_file.exists():
            with open(aggregate_file, 'r') as f:
                self.aggregate_results = json.load(f)
            print(f"  ✓ Loaded aggregate results")

        # Load archive if exists
        for method_dir in self.results_dir.glob("llm_qd/query_*"):
            archive_file = method_dir / "archive.json"
            if archive_file.exists():
                with open(archive_file, 'r') as f:
                    self.archive_data[method_dir.name] = json.load(f)

        # Load history files
        for method_dir in self.results_dir.glob("llm_qd/query_*"):
            history_file = method_dir / "history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.history_data.append(json.load(f))

        print(f"  ✓ Loaded {len(self.history_data)} history files")

    def plot_coverage_over_time(self):
        """
        Plot coverage progression over iterations
        """
        print("\nGenerating coverage over time plot...")

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.history_data)))

        for i, history in enumerate(self.history_data):
            if not history:
                continue

            iterations = [h['iteration'] for h in history]
            coverages = [h['coverage'] * 100 for h in history]

            ax.plot(iterations, coverages, label=f'Query {i+1}',
                   color=colors[i], linewidth=2, marker='o', markersize=4, alpha=0.7)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Coverage (%)', fontsize=12)
        ax.set_title('LLM-QD Coverage Progression', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "coverage_over_time.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved to {output_file}")

    def plot_fitness_distribution(self):
        """
        Plot fitness score distribution
        """
        print("\nGenerating fitness distribution plot...")

        if 'llm_qd' not in self.aggregate_results or not self.aggregate_results['llm_qd']:
            print("  ⚠ No LLM-QD results to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Collect all fitness scores
        avg_fitnesses = [r['avg_fitness'] for r in self.aggregate_results['llm_qd']]
        max_fitnesses = [r['max_fitness'] for r in self.aggregate_results['llm_qd']]

        # Plot 1: Histogram of average fitness
        ax1.hist(avg_fitnesses, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(avg_fitnesses), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(avg_fitnesses):.1f}')
        ax1.set_xlabel('Average Fitness', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Average Fitness', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Histogram of max fitness
        ax2.hist(max_fitnesses, bins=15, color='darkgreen', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(max_fitnesses), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(max_fitnesses):.1f}')
        ax2.set_xlabel('Maximum Fitness', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Maximum Fitness', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_file = self.output_dir / "fitness_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved to {output_file}")

    def plot_behavior_space_heatmap(self):
        """
        Plot 2D projections of 4D behavior space
        """
        print("\nGenerating behavior space heatmaps...")

        # Need archive data with individual entries
        if not self.archive_data:
            print("  ⚠ No archive data available")
            return

        # Take first archive for visualization
        first_archive_key = list(self.archive_data.keys())[0]
        archive = self.archive_data[first_archive_key]

        if 'entries' not in archive or not archive['entries']:
            print("  ⚠ No entries in archive")
            return

        # Extract behaviors and fitness
        behaviors = []
        fitnesses = []

        for entry in archive['entries']:
            behavior = entry['behavior']  # (complexity, style, symmetry, color)
            fitness = entry['fitness']

            behaviors.append(behavior)
            fitnesses.append(fitness)

        behaviors = np.array(behaviors)
        fitnesses = np.array(fitnesses)

        # Create 2D projections
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Behavior Space Heatmaps (2D Projections of 4D Space)',
                    fontsize=14, fontweight='bold')

        projections = [
            (0, 1, 'Complexity', 'Style'),
            (0, 2, 'Complexity', 'Symmetry'),
            (0, 3, 'Complexity', 'Color'),
            (1, 2, 'Style', 'Symmetry'),
            (1, 3, 'Style', 'Color'),
            (2, 3, 'Symmetry', 'Color')
        ]

        for idx, (dim1, dim2, label1, label2) in enumerate(projections):
            ax = axes[idx // 3, idx % 3]

            # Create 2D histogram
            x = behaviors[:, dim1]
            y = behaviors[:, dim2]

            # Scatter plot with fitness as color
            scatter = ax.scatter(x, y, c=fitnesses, cmap='viridis',
                               s=100, alpha=0.6, edgecolor='black', linewidth=0.5)

            ax.set_xlabel(label1, fontsize=10)
            ax.set_ylabel(label2, fontsize=10)
            ax.set_title(f'{label1} vs {label2}', fontsize=11)
            ax.grid(True, alpha=0.3)

            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Fitness', fontsize=9)

        plt.tight_layout()
        output_file = self.output_dir / "behavior_space_heatmaps.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved to {output_file}")

    def plot_convergence_comparison(self):
        """
        Plot convergence curves comparing multiple runs
        """
        print("\nGenerating convergence comparison plot...")

        if not self.history_data:
            print("  ⚠ No history data available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.history_data)))

        for i, history in enumerate(self.history_data):
            if not history:
                continue

            iterations = [h['iteration'] for h in history]
            coverages = [h['coverage'] for h in history]
            avg_fitnesses = [h['avg_fitness'] for h in history]

            # Plot 1: Coverage convergence
            ax1.plot(iterations, coverages, label=f'Query {i+1}',
                    color=colors[i], linewidth=2, alpha=0.7)

            # Plot 2: Fitness convergence
            ax2.plot(iterations, avg_fitnesses, label=f'Query {i+1}',
                    color=colors[i], linewidth=2, alpha=0.7)

        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Coverage (fraction)', fontsize=12)
        ax1.set_title('Coverage Convergence', fontsize=13, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Average Fitness', fontsize=12)
        ax2.set_title('Fitness Convergence', fontsize=13, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "convergence_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved to {output_file}")

    def plot_quality_diversity_scatter(self):
        """
        Plot Quality-Diversity scatter (coverage vs fitness)
        """
        print("\nGenerating Quality-Diversity scatter plot...")

        if 'llm_qd' not in self.aggregate_results or not self.aggregate_results['llm_qd']:
            print("  ⚠ No LLM-QD results to plot")
            return

        results = self.aggregate_results['llm_qd']

        coverages = [r['coverage'] * 100 for r in results]
        avg_fitnesses = [r['avg_fitness'] for r in results]
        max_fitnesses = [r['max_fitness'] for r in results]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot points
        scatter = ax.scatter(coverages, avg_fitnesses, s=200, c=max_fitnesses,
                           cmap='RdYlGn', alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Max Fitness', fontsize=11)

        # Add labels for each point
        for i, (cov, fit) in enumerate(zip(coverages, avg_fitnesses)):
            ax.annotate(f'Q{i+1}', (cov, fit), fontsize=9,
                       xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('Coverage (%)', fontsize=12)
        ax.set_ylabel('Average Fitness', fontsize=12)
        ax.set_title('Quality-Diversity Trade-off', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add ideal region indicator
        ax.axhline(y=75, color='gray', linestyle='--', alpha=0.5, label='Quality threshold')
        ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5, label='Diversity threshold')
        ax.legend(loc='lower right')

        plt.tight_layout()
        output_file = self.output_dir / "quality_diversity_scatter.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved to {output_file}")

    def plot_summary_dashboard(self):
        """
        Create comprehensive summary dashboard
        """
        print("\nGenerating summary dashboard...")

        if 'llm_qd' not in self.aggregate_results or not self.aggregate_results['llm_qd']:
            print("  ⚠ No LLM-QD results to plot")
            return

        results = self.aggregate_results['llm_qd']

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Coverage over time (large, top)
        ax1 = fig.add_subplot(gs[0, :])
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.history_data)))
        for i, history in enumerate(self.history_data):
            if history:
                iterations = [h['iteration'] for h in history]
                coverages = [h['coverage'] * 100 for h in history]
                ax1.plot(iterations, coverages, label=f'Q{i+1}',
                        color=colors[i], linewidth=2, marker='o', markersize=3)
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Coverage (%)', fontsize=11)
        ax1.set_title('Coverage Progression', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right', ncol=2, fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 2. Fitness distribution (middle left)
        ax2 = fig.add_subplot(gs[1, 0])
        avg_fits = [r['avg_fitness'] for r in results]
        ax2.hist(avg_fits, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(avg_fits), color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Avg Fitness', fontsize=10)
        ax2.set_ylabel('Count', fontsize=10)
        ax2.set_title('Avg Fitness Distribution', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Max fitness distribution (middle center)
        ax3 = fig.add_subplot(gs[1, 1])
        max_fits = [r['max_fitness'] for r in results]
        ax3.hist(max_fits, bins=10, color='darkgreen', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(max_fits), color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Max Fitness', fontsize=10)
        ax3.set_ylabel('Count', fontsize=10)
        ax3.set_title('Max Fitness Distribution', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. API usage (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        api_calls = [r['api_calls'] for r in results]
        queries = [f'Q{i+1}' for i in range(len(results))]
        ax4.bar(queries, api_calls, color='coral', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Query', fontsize=10)
        ax4.set_ylabel('API Calls', fontsize=10)
        ax4.set_title('API Usage per Query', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Quality-Diversity scatter (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        coverages = [r['coverage'] * 100 for r in results]
        avg_fits = [r['avg_fitness'] for r in results]
        ax5.scatter(coverages, avg_fits, s=150, c='purple', alpha=0.6, edgecolor='black')
        for i, (cov, fit) in enumerate(zip(coverages, avg_fits)):
            ax5.annotate(f'Q{i+1}', (cov, fit), fontsize=8, xytext=(3, 3), textcoords='offset points')
        ax5.set_xlabel('Coverage (%)', fontsize=10)
        ax5.set_ylabel('Avg Fitness', fontsize=10)
        ax5.set_title('Quality-Diversity Trade-off', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # 6. Time and cost (bottom center)
        ax6 = fig.add_subplot(gs[2, 1])
        times = [r['time_seconds'] / 60 for r in results]  # Convert to minutes
        ax6.bar(queries, times, color='teal', alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Query', fontsize=10)
        ax6.set_ylabel('Time (minutes)', fontsize=10)
        ax6.set_title('Execution Time', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

        # 7. Summary stats (bottom right)
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')

        summary_text = f"""
        SUMMARY STATISTICS

        Queries: {len(results)}

        Coverage:
          Mean: {np.mean(coverages):.1f}%
          Std: {np.std(coverages):.1f}%

        Avg Fitness:
          Mean: {np.mean(avg_fits):.1f}
          Std: {np.std(avg_fits):.1f}

        Max Fitness:
          Mean: {np.mean(max_fits):.1f}
          Std: {np.std(max_fits):.1f}

        Total API Calls: {sum(api_calls)}
        Total Time: {sum([r['time_seconds'] for r in results])/60:.1f} min
        Total Cost: ${sum([r['cost_estimate_usd'] for r in results]):.2f}
        """

        ax7.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
                verticalalignment='center')

        fig.suptitle('LLM-QD EXPERIMENT DASHBOARD', fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(self.output_dir / "summary_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved to {self.output_dir / 'summary_dashboard.png'}")

    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print(f"\n{'='*80}")
        print(f"GENERATING VISUALIZATIONS")
        print(f"{'='*80}")

        self.plot_coverage_over_time()
        self.plot_fitness_distribution()
        self.plot_behavior_space_heatmap()
        self.plot_convergence_comparison()
        self.plot_quality_diversity_scatter()
        self.plot_summary_dashboard()

        print(f"\n{'='*80}")
        print(f"ALL VISUALIZATIONS COMPLETE")
        print(f"{'='*80}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}\n")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python visualize_llm_qd.py <results_directory>")
        print("\nExample:")
        print("  python visualize_llm_qd.py ../experiments/comprehensive_20251127_120000")
        return

    results_dir = sys.argv[1]

    if not Path(results_dir).exists():
        print(f"Error: Directory not found: {results_dir}")
        return

    visualizer = LLMQDVisualizer(results_dir)
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
