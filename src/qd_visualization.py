"""
QD Visualization for MAP-Elites Archive
========================================
Interactive and static visualizations for Quality-Diversity archives.

Features:
- 2D projections of 5D behavior space
- Quality heatmaps
- Coverage analysis
- Behavioral distribution histograms
- Interactive HTML visualization
- Export to various formats
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict


try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Some visualizations will be skipped.")


class QDVisualizer:
    """Visualization tools for QD archives"""

    def __init__(self, archive):
        """
        Initialize visualizer

        Args:
            archive: EnhancedQDArchive instance
        """
        self.archive = archive
        self.dimension_names = ['Complexity', 'Style', 'Symmetry', 'Color Richness', 'Emotional Tone']

    def create_interactive_grid(self, output_html: str):
        """
        Create interactive HTML visualization

        Features:
        - 2D grid views
        - Click cells to see logo details
        - Filter by dimensions
        - Quality heatmap overlay

        Args:
            output_html: Path to save HTML file
        """
        # Export data
        data = {
            'dimensions': self.archive.dimensions,
            'dimension_names': self.dimension_names[:len(self.archive.dimensions)],
            'entries': [],
            'coverage_metrics': self.archive.compute_coverage_metrics()
        }

        for behavior, entry in self.archive.archive.items():
            data['entries'].append({
                'logo_id': entry.logo_id,
                'behavior': list(entry.behavior),
                'fitness': entry.fitness,
                'raw_behavior': entry.raw_behavior,
                'genome': entry.genome,
                'svg_code': entry.svg_code,
                'metadata': entry.metadata
            })

        # Create HTML
        html = self._generate_html_template(data)

        with open(output_html, 'w') as f:
            f.write(html)

        print(f"Interactive visualization saved to {output_html}")

    def _generate_html_template(self, data: Dict) -> str:
        """Generate HTML template for interactive visualization"""
        data_json = json.dumps(data, indent=2)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QD Archive Visualization</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .controls {{
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        select, button {{
            padding: 8px 15px;
            margin: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        button {{
            background: #4CAF50;
            color: white;
            cursor: pointer;
            border: none;
        }}
        button:hover {{
            background: #45a049;
        }}
        .grid-container {{
            display: flex;
            gap: 20px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        .grid-view {{
            flex: 1;
            min-width: 400px;
        }}
        canvas {{
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: crosshair;
        }}
        .info-panel {{
            min-width: 300px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .logo-preview {{
            margin-top: 15px;
            padding: 10px;
            background: white;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Quality-Diversity Archive Visualization</h1>
        <p>Interactive exploration of 5D behavior space</p>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Coverage</div>
                <div class="stat-value" id="coverage">0%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Occupied Cells</div>
                <div class="stat-value" id="occupied">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Fitness</div>
                <div class="stat-value" id="avg-fitness">0.0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Max Fitness</div>
                <div class="stat-value" id="max-fitness">0.0</div>
            </div>
        </div>

        <div class="controls">
            <label>X-Axis:</label>
            <select id="x-dim">
                <option value="0">Complexity</option>
                <option value="1">Style</option>
                <option value="2">Symmetry</option>
                <option value="3">Color Richness</option>
                <option value="4">Emotional Tone</option>
            </select>

            <label>Y-Axis:</label>
            <select id="y-dim">
                <option value="1">Style</option>
                <option value="0">Complexity</option>
                <option value="2">Symmetry</option>
                <option value="3">Color Richness</option>
                <option value="4">Emotional Tone</option>
            </select>

            <button onclick="renderGrid()">Update View</button>
            <button onclick="exportData()">Export Data</button>
        </div>

        <div class="grid-container">
            <div class="grid-view">
                <h3>Behavior Space Projection</h3>
                <canvas id="grid-canvas" width="500" height="500"></canvas>
            </div>
            <div class="info-panel">
                <h3>Selection Info</h3>
                <div id="selection-info">Click on a cell to see details</div>
                <div class="logo-preview" id="logo-preview"></div>
            </div>
        </div>
    </div>

    <script>
        const archiveData = {data_json};
        let currentXDim = 0;
        let currentYDim = 1;

        function initialize() {{
            // Update stats
            const metrics = archiveData.coverage_metrics;
            document.getElementById('coverage').textContent =
                (metrics.overall_coverage * 100).toFixed(1) + '%';
            document.getElementById('occupied').textContent =
                metrics.occupied_cells;
            document.getElementById('avg-fitness').textContent =
                metrics.quality_distribution.mean.toFixed(1);
            document.getElementById('max-fitness').textContent =
                metrics.quality_distribution.max.toFixed(1);

            renderGrid();
        }}

        function renderGrid() {{
            currentXDim = parseInt(document.getElementById('x-dim').value);
            currentYDim = parseInt(document.getElementById('y-dim').value);

            const canvas = document.getElementById('grid-canvas');
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;

            // Clear
            ctx.fillStyle = '#f0f0f0';
            ctx.fillRect(0, 0, width, height);

            // Get dimensions
            const xDimSize = archiveData.dimensions[currentXDim];
            const yDimSize = archiveData.dimensions[currentYDim];

            const cellWidth = width / xDimSize;
            const cellHeight = height / yDimSize;

            // Project entries to 2D
            const grid = {{}};
            archiveData.entries.forEach(entry => {{
                const x = entry.behavior[currentXDim];
                const y = entry.behavior[currentYDim];
                const key = `${{x}},${{y}}`;

                if (!grid[key] || entry.fitness > grid[key].fitness) {{
                    grid[key] = entry;
                }}
            }});

            // Render cells
            Object.entries(grid).forEach(([key, entry]) => {{
                const [x, y] = key.split(',').map(Number);

                // Color by fitness
                const fitness = entry.fitness;
                const normalized = (fitness - 60) / 40; // Assuming 60-100 range
                const color = getFitnessColor(normalized);

                ctx.fillStyle = color;
                ctx.fillRect(
                    x * cellWidth,
                    (yDimSize - 1 - y) * cellHeight,
                    cellWidth - 1,
                    cellHeight - 1
                );
            }});

            // Add grid lines
            ctx.strokeStyle = '#ddd';
            ctx.lineWidth = 0.5;
            for (let i = 0; i <= xDimSize; i++) {{
                ctx.beginPath();
                ctx.moveTo(i * cellWidth, 0);
                ctx.lineTo(i * cellWidth, height);
                ctx.stroke();
            }}
            for (let i = 0; i <= yDimSize; i++) {{
                ctx.beginPath();
                ctx.moveTo(0, i * cellHeight);
                ctx.lineTo(width, i * cellHeight);
                ctx.stroke();
            }}
        }}

        function getFitnessColor(normalized) {{
            // Viridis-like colormap
            const r = Math.floor(255 * (0.267 + 0.733 * normalized));
            const g = Math.floor(255 * (0.005 + 0.765 * normalized));
            const b = Math.floor(255 * (0.329 + 0.145 * (1 - normalized)));
            return `rgb(${{r}},${{g}},${{b}})`;
        }}

        // Canvas click handler
        document.getElementById('grid-canvas').addEventListener('click', (e) => {{
            const canvas = e.target;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const xDimSize = archiveData.dimensions[currentXDim];
            const yDimSize = archiveData.dimensions[currentYDim];
            const cellWidth = canvas.width / xDimSize;
            const cellHeight = canvas.height / yDimSize;

            const cellX = Math.floor(x / cellWidth);
            const cellY = yDimSize - 1 - Math.floor(y / cellHeight);

            // Find entry
            const key = `${{cellX}},${{cellY}}`;
            const entry = archiveData.entries.find(e =>
                e.behavior[currentXDim] === cellX &&
                e.behavior[currentYDim] === cellY
            );

            if (entry) {{
                showEntryDetails(entry);
            }} else {{
                document.getElementById('selection-info').innerHTML =
                    '<p>Empty cell at (' + cellX + ', ' + cellY + ')</p>';
                document.getElementById('logo-preview').innerHTML = '';
            }}
        }});

        function showEntryDetails(entry) {{
            const info = `
                <p><strong>Logo ID:</strong> ${{entry.logo_id}}</p>
                <p><strong>Fitness:</strong> ${{entry.fitness.toFixed(2)}}</p>
                <p><strong>Behavior:</strong> ${{entry.behavior.join(', ')}}</p>
                <p><strong>Company:</strong> ${{entry.genome.company || 'N/A'}}</p>
            `;
            document.getElementById('selection-info').innerHTML = info;

            // Show SVG preview (limited to avoid issues)
            const preview = `<div>${{entry.svg_code.substring(0, 500)}}...</div>`;
            document.getElementById('logo-preview').innerHTML = preview;
        }}

        function exportData() {{
            const dataStr = JSON.stringify(archiveData, null, 2);
            const blob = new Blob([dataStr], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'qd_archive_export.json';
            a.click();
        }}

        // Initialize on load
        initialize();
    </script>
</body>
</html>"""

        return html

    def create_behavior_distributions(self, output_path: str):
        """
        Create histograms showing distribution across each dimension

        Args:
            output_path: Path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available, skipping behavior distribution plots")
            return

        n_dims = len(self.archive.dimensions)
        fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 4))

        if n_dims == 1:
            axes = [axes]

        for dim_idx in range(n_dims):
            ax = axes[dim_idx]

            # Extract bin values for this dimension
            bins = [entry.behavior[dim_idx] for entry in self.archive.archive.values()]

            # Create histogram
            ax.hist(bins, bins=self.archive.dimensions[dim_idx],
                   range=(-0.5, self.archive.dimensions[dim_idx] - 0.5),
                   color='steelblue', alpha=0.7, edgecolor='black')

            dim_name = self.dimension_names[dim_idx] if dim_idx < len(self.dimension_names) else f'Dim {dim_idx}'
            ax.set_xlabel(f'{dim_name} Bin', fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')
            ax.set_title(f'{dim_name} Distribution')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Behavior distributions saved to {output_path}")

    def create_quality_heatmaps(self, output_path: str):
        """
        Create heatmaps showing quality in different 2D projections

        Args:
            output_path: Path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available, skipping quality heatmaps")
            return

        # Create 2D projections
        projections = [
            (0, 1, 'Complexity vs Style'),
            (0, 2, 'Complexity vs Symmetry'),
            (1, 2, 'Style vs Symmetry'),
            (3, 4, 'Color vs Emotional'),
            (0, 4, 'Complexity vs Emotional'),
            (2, 4, 'Symmetry vs Emotional')
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Quality Heatmaps - 2D Projections of 5D Space', fontsize=16, fontweight='bold')

        for idx, (dim1, dim2, title) in enumerate(projections):
            ax = axes[idx // 3, idx % 3]

            # Create 2D grid
            grid = np.zeros((self.archive.dimensions[dim2], self.archive.dimensions[dim1]))
            counts = np.zeros_like(grid)

            for entry in self.archive.archive.values():
                x = entry.behavior[dim1]
                y = entry.behavior[dim2]
                grid[y, x] = max(grid[y, x], entry.fitness)
                counts[y, x] += 1

            # Mask empty cells
            grid_masked = np.ma.masked_where(counts == 0, grid)

            # Plot heatmap
            im = ax.imshow(grid_masked, cmap='viridis', aspect='auto',
                          interpolation='nearest', origin='lower')

            dim1_name = self.dimension_names[dim1] if dim1 < len(self.dimension_names) else f'Dim {dim1}'
            dim2_name = self.dimension_names[dim2] if dim2 < len(self.dimension_names) else f'Dim {dim2}'

            ax.set_xlabel(dim1_name, fontweight='bold')
            ax.set_ylabel(dim2_name, fontweight='bold')
            ax.set_title(title)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Fitness', fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Quality heatmaps saved to {output_path}")

    def create_coverage_analysis(self, output_path: str):
        """
        Create comprehensive coverage analysis visualization

        Args:
            output_path: Path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available, skipping coverage analysis")
            return

        metrics = self.archive.compute_coverage_metrics()

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Overall coverage gauge
        ax1 = fig.add_subplot(gs[0, 0])
        coverage = metrics['overall_coverage']
        ax1.barh([0], [coverage], color='#4CAF50', height=0.5)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xlabel('Coverage', fontweight='bold')
        ax1.set_title(f'Overall Coverage: {coverage*100:.2f}%', fontweight='bold')
        ax1.set_yticks([])
        ax1.grid(True, alpha=0.3)

        # 2. Per-dimension coverage
        ax2 = fig.add_subplot(gs[0, 1:])
        per_dim = metrics['per_dimension_coverage']
        dims = list(per_dim.keys())
        values = [per_dim[d] for d in dims]
        colors = ['#FF6B6B' if v < 0.5 else '#4ECDC4' if v < 0.8 else '#95E1D3' for v in values]

        ax2.bar(dims, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Coverage', fontweight='bold')
        ax2.set_title('Per-Dimension Coverage', fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 3. Quality distribution
        ax3 = fig.add_subplot(gs[1, :])
        fitnesses = [e.fitness for e in self.archive.archive.values()]
        ax3.hist(fitnesses, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax3.axvline(metrics['quality_distribution']['mean'], color='red',
                   linestyle='--', linewidth=2, label=f"Mean: {metrics['quality_distribution']['mean']:.2f}")
        ax3.axvline(metrics['quality_distribution']['median'], color='green',
                   linestyle='--', linewidth=2, label=f"Median: {metrics['quality_distribution']['median']:.2f}")
        ax3.set_xlabel('Fitness', fontweight='bold')
        ax3.set_ylabel('Count', fontweight='bold')
        ax3.set_title('Fitness Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Statistics table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')

        stats_data = [
            ['Metric', 'Value'],
            ['Occupied Cells', f"{metrics['occupied_cells']:,}"],
            ['Total Cells', f"{metrics['total_cells']:,}"],
            ['Coverage', f"{metrics['overall_coverage']*100:.2f}%"],
            ['Min Fitness', f"{metrics['quality_distribution']['min']:.2f}"],
            ['Max Fitness', f"{metrics['quality_distribution']['max']:.2f}"],
            ['Mean Fitness', f"{metrics['quality_distribution']['mean']:.2f}"],
            ['Std Fitness', f"{metrics['quality_distribution']['std']:.2f}"],
            ['Avg Behavior Distance', f"{metrics['diversity_metrics']['average_behavior_distance']:.2f}"],
            ['Generation Diversity', f"{metrics['diversity_metrics']['generation_diversity']}"]
        ]

        table = ax4.table(cellText=stats_data, cellLoc='left', loc='center',
                         colWidths=[0.4, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Coverage analysis saved to {output_path}")

    def create_complete_report(self, output_dir: str):
        """
        Generate complete visualization report

        Args:
            output_dir: Directory to save all visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*60)
        print("Generating Complete QD Visualization Report")
        print("="*60)

        # 1. Interactive HTML
        self.create_interactive_grid(str(output_path / "interactive.html"))

        # 2. Behavior distributions
        self.create_behavior_distributions(str(output_path / "behavior_distributions.png"))

        # 3. Quality heatmaps
        self.create_quality_heatmaps(str(output_path / "quality_heatmaps.png"))

        # 4. Coverage analysis
        self.create_coverage_analysis(str(output_path / "coverage_analysis.png"))

        # 5. Export data
        self.archive.export_for_visualization(str(output_path / "archive_data.json"))

        print(f"\n{'='*60}")
        print(f"Complete report saved to: {output_path}")
        print(f"{'='*60}")


def demo():
    """Demo QD visualization"""
    from map_elites_archive import EnhancedQDArchive
    import random

    print("="*60)
    print("QD Visualization Demo")
    print("="*60)

    # Create and populate archive
    archive = EnhancedQDArchive(dimensions=(10, 10, 10, 10, 10))

    for i in range(200):
        behavior = tuple(random.randint(0, 9) for _ in range(5))
        archive.add(
            logo_id=f"logo_{i}",
            svg_code=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><circle cx="100" cy="100" r="50" fill="#4CAF50"/></svg>',
            genome={'company': f'Company {i}', 'industry': 'Tech'},
            fitness=random.uniform(60, 95),
            aesthetic_breakdown={'balance': 0.8, 'contrast': 0.7},
            behavior=behavior,
            raw_behavior={
                'complexity': random.randint(15, 50),
                'style': random.random(),
                'symmetry': random.random(),
                'color_richness': random.random(),
                'emotional_tone': random.random()
            },
            generation=random.randint(0, 10),
            metadata={'design_rationale': f'Test logo {i}'}
        )

    print(f"Archive populated with {len(archive.archive)} entries")

    # Create visualizer
    visualizer = QDVisualizer(archive)

    # Generate all visualizations
    output_dir = "/home/luis/svg-logo-ai/output/qd_viz_demo"
    visualizer.create_complete_report(output_dir)


if __name__ == "__main__":
    demo()
