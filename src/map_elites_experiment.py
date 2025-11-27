"""
MAP-Elites Logo Generation Experiment
======================================
LLM-Guided MAP-Elites for SVG Logo Generation

This is Research Idea #1 - a NOVEL contribution combining:
- MAP-Elites quality-diversity algorithm
- LLM-guided mutations toward behavioral targets
- Multi-dimensional behavior characterization

Expected outcomes:
- Diverse portfolio of 1,000-3,000 logos across behavioral space
- Systematic exploration of design space (complexity, style, symmetry, color)
- Better diversity than pure evolutionary approaches
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from behavior_characterization import BehaviorCharacterizer
from map_elites_archive import MAPElitesArchive, ArchiveEntry
from llm_guided_mutation import LLMGuidedMutator
from logo_validator import LogoValidator
from experiment_tracker import ExperimentTracker


class LLMMELogo:
    """
    LLM-Guided MAP-Elites for Logo Generation

    Main experiment class implementing the MAP-Elites algorithm with
    LLM-guided mutations.
    """

    def __init__(self,
                 company_name: str,
                 industry: str,
                 grid_dimensions: Tuple[int, int, int, int] = (10, 10, 10, 10),
                 use_llm: bool = True,
                 experiment_name: str = "llm_me_logo"):
        """
        Initialize LLM-ME-Logo experiment

        Args:
            company_name: Company name for logo generation
            industry: Industry sector
            grid_dimensions: MAP-Elites grid dimensions (complexity, style, symmetry, color)
            use_llm: Use actual LLM (requires API key) or mock mode
            experiment_name: Name for experiment tracking
        """
        self.company_name = company_name
        self.industry = industry
        self.grid_dimensions = grid_dimensions
        self.use_llm = use_llm

        # Initialize components
        self.characterizer = BehaviorCharacterizer(num_bins=grid_dimensions[0])
        self.archive = MAPElitesArchive(dimensions=grid_dimensions)
        self.validator = LogoValidator()
        self.tracker = ExperimentTracker(
            experiment_name=experiment_name,
            base_dir="/home/luis/svg-logo-ai"
        )

        # Initialize LLM mutator (if API key available)
        self.mutator = None
        if use_llm:
            try:
                self.mutator = LLMGuidedMutator()
                print("✓ LLM mutator initialized")
            except Exception as e:
                print(f"⚠ Could not initialize LLM mutator: {e}")
                print("  Falling back to mock mode")
                self.use_llm = False

        # Track statistics
        self.generation = 0
        self.total_evaluations = 0
        self.successful_additions = 0
        self.failed_additions = 0

        # Log initialization
        self.tracker.log_decision(
            decision="Use LLM-Guided MAP-Elites for logo generation",
            rationale="Combine quality-diversity with intelligent LLM-guided mutations for systematic design space exploration",
            alternatives=[
                "Pure evolutionary algorithm",
                "Random sampling",
                "RAG-only approach"
            ],
            metadata={
                "grid_dimensions": f"{grid_dimensions[0]}x{grid_dimensions[1]}x{grid_dimensions[2]}x{grid_dimensions[3]}",
                "total_cells": int(np.prod(grid_dimensions)),
                "use_llm": use_llm,
                "company": company_name,
                "industry": industry
            }
        )

    def create_random_genome(self) -> Dict:
        """Create random genome for logo generation"""
        style_options = [
            ['modern', 'minimal'],
            ['geometric', 'clean'],
            ['organic', 'flowing'],
            ['bold', 'dynamic'],
            ['elegant', 'refined']
        ]

        color_palettes = [
            ['#2563eb'],  # Blue monochrome
            ['#e74c3c'],  # Red monochrome
            ['#2563eb', '#1e40af'],  # Blue duotone
            ['#e74c3c', '#c0392b'],  # Red duotone
            ['#2563eb', '#10b981'],  # Blue-green
            ['#e74c3c', '#f39c12', '#2563eb'],  # Triadic
        ]

        design_principles = [
            ['simplicity', 'balance'],
            ['hierarchy', 'contrast'],
            ['symmetry', 'harmony'],
            ['rhythm', 'proportion']
        ]

        complexity_targets = [15, 20, 25, 30, 35, 40]

        return {
            'company': self.company_name,
            'industry': self.industry,
            'style_keywords': random.choice(style_options),
            'color_palette': random.choice(color_palettes),
            'design_principles': random.choice(design_principles),
            'complexity_target': random.choice(complexity_targets),
            'color_harmony_type': random.choice(['monochrome', 'analogous', 'complementary', 'triadic'])
        }

    def generate_logo(self, genome: Dict) -> str:
        """
        Generate logo from genome

        Args:
            genome: Genome dictionary

        Returns:
            SVG code
        """
        if self.use_llm and self.mutator:
            return self.mutator.generate_from_genome(genome)
        else:
            # Mock SVG for testing without API
            return self._generate_mock_svg(genome)

    def _generate_mock_svg(self, genome: Dict) -> str:
        """Generate mock SVG for testing (no LLM required)"""
        complexity = genome.get('complexity_target', 20)
        colors = genome.get('color_palette', ['#2563eb'])

        # Generate random elements
        elements = []
        for i in range(complexity):
            if random.random() < 0.3:
                # Circle
                cx = random.randint(20, 180)
                cy = random.randint(20, 180)
                r = random.randint(5, 30)
                color = random.choice(colors)
                elements.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>')
            elif random.random() < 0.5:
                # Rectangle
                x = random.randint(10, 150)
                y = random.randint(10, 150)
                w = random.randint(10, 40)
                h = random.randint(10, 40)
                color = random.choice(colors)
                elements.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}"/>')
            else:
                # Path (simple)
                x1, y1 = random.randint(20, 180), random.randint(20, 180)
                x2, y2 = random.randint(20, 180), random.randint(20, 180)
                color = random.choice(colors)
                elements.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{color}" stroke-width="2" fill="none"/>')

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
{chr(10).join('  ' + el for el in elements)}
</svg>"""
        return svg

    def mutate_logo(self,
                   source_svg: str,
                   current_behavior: Tuple[int, int, int, int],
                   target_behavior: Tuple[int, int, int, int],
                   genome: Dict) -> str:
        """
        Mutate logo toward target behavior

        Args:
            source_svg: Source SVG code
            current_behavior: Current behavior bins
            target_behavior: Target behavior bins
            genome: Genome context

        Returns:
            Mutated SVG code
        """
        if self.use_llm and self.mutator:
            return self.mutator.mutate_toward_target(
                source_svg,
                current_behavior,
                target_behavior,
                genome
            )
        else:
            # Mock mutation: slight modification
            return self._mock_mutate(source_svg, target_behavior)

    def _mock_mutate(self, svg: str, target: Tuple[int, int, int, int]) -> str:
        """Mock mutation for testing"""
        # Just add a random circle to increase complexity slightly
        colors = ['#2563eb', '#e74c3c', '#10b981']
        cx = random.randint(50, 150)
        cy = random.randint(50, 150)
        r = random.randint(5, 20)
        color = random.choice(colors)

        # Insert before </svg>
        new_element = f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>'
        return svg.replace('</svg>', f'{new_element}\n</svg>')

    def evaluate_logo(self, svg_code: str) -> Tuple[float, Dict]:
        """
        Evaluate logo fitness

        Args:
            svg_code: SVG code

        Returns:
            (fitness_score, aesthetic_breakdown)
        """
        results = self.validator.validate_all(svg_code)
        fitness = results['final_score']

        aesthetic_breakdown = {
            'aesthetic': results['level5_aesthetic']['score'],
            'professional': results['level4_professional']['score'],
            'technical': (results['level1_xml']['score'] + results['level2_svg']['score'] + results['level3_quality']['score']) / 3,
            'golden_ratio': results['level5_aesthetic']['golden_ratio'],
            'color_harmony': results['level5_aesthetic']['color_harmony'],
            'visual_interest': results['level5_aesthetic']['visual_interest']
        }

        self.total_evaluations += 1
        return fitness, aesthetic_breakdown

    def initialize_archive(self, n_random: int = 100):
        """
        Initialize archive with random logos

        Args:
            n_random: Number of random logos to generate
        """
        print(f"\nInitializing archive with {n_random} random logos...")

        for i in range(n_random):
            # Create random genome
            genome = self.create_random_genome()

            # Generate logo
            svg_code = self.generate_logo(genome)

            # Evaluate
            fitness, aesthetic_breakdown = self.evaluate_logo(svg_code)

            # Characterize behavior
            char_result = self.characterizer.characterize(svg_code)
            behavior = char_result['bins']
            raw_behavior = char_result['raw_scores']

            # Add to archive
            logo_id = f"init_{i:04d}"
            added = self.archive.add(
                logo_id=logo_id,
                svg_code=svg_code,
                genome=genome,
                fitness=fitness,
                aesthetic_breakdown=aesthetic_breakdown,
                behavior=behavior,
                raw_behavior=raw_behavior,
                generation=0
            )

            if added:
                self.successful_additions += 1

            if (i + 1) % 10 == 0:
                stats = self.archive.get_statistics()
                print(f"  [{i+1}/{n_random}] Coverage: {stats['coverage']*100:.1f}%, Avg Fitness: {stats['avg_fitness']:.1f}")

        # Log initialization results
        stats = self.archive.get_statistics()
        self.tracker.log_result(
            result_type="initialization",
            metrics={
                "n_generated": n_random,
                "n_added": self.successful_additions,
                "coverage": stats['coverage'],
                "avg_fitness": stats['avg_fitness'],
                "max_fitness": stats['max_fitness']
            },
            description="Archive initialized with random logos"
        )

        print(f"\n✓ Archive initialized:")
        print(f"  Occupied cells: {stats['num_occupied']}")
        print(f"  Coverage: {stats['coverage']*100:.1f}%")
        print(f"  Avg fitness: {stats['avg_fitness']:.2f}")

    def run_iterations(self, n_iterations: int = 1000):
        """
        Run MAP-Elites iterations

        Args:
            n_iterations: Number of iterations to run
        """
        print(f"\nRunning {n_iterations} MAP-Elites iterations...")

        for iteration in range(n_iterations):
            # 1. Select random occupied cell
            result = self.archive.get_random_occupied()
            if result is None:
                print("⚠ Archive is empty!")
                break

            source_behavior, source_entry = result

            # 2. Select random neighboring empty cell (or any neighbor if all occupied)
            empty_neighbors = self.archive.get_empty_neighbors(source_behavior, distance=1)

            if empty_neighbors:
                target_behavior = random.choice(empty_neighbors)
            else:
                # All neighbors occupied, try distance=2
                empty_neighbors = self.archive.get_empty_neighbors(source_behavior, distance=2)
                if empty_neighbors:
                    target_behavior = random.choice(empty_neighbors)
                else:
                    # Explore any empty cell
                    target_behavior = self._random_empty_cell()

            if target_behavior is None:
                print("⚠ Archive is full!")
                break

            # 3. Mutate logo toward target behavior
            mutated_svg = self.mutate_logo(
                source_entry.svg_code,
                source_behavior,
                target_behavior,
                source_entry.genome
            )

            # 4. Evaluate
            fitness, aesthetic_breakdown = self.evaluate_logo(mutated_svg)

            # 5. Characterize
            char_result = self.characterizer.characterize(mutated_svg)
            actual_behavior = char_result['bins']
            raw_behavior = char_result['raw_scores']

            # 6. Add to archive (may end up in different cell than target)
            logo_id = f"gen{self.generation:03d}_iter{iteration:04d}"
            added = self.archive.add(
                logo_id=logo_id,
                svg_code=mutated_svg,
                genome=source_entry.genome,  # Inherit genome
                fitness=fitness,
                aesthetic_breakdown=aesthetic_breakdown,
                behavior=actual_behavior,
                raw_behavior=raw_behavior,
                generation=self.generation,
                parent_ids=[source_entry.logo_id]
            )

            if added:
                self.successful_additions += 1
            else:
                self.failed_additions += 1

            # Progress report
            if (iteration + 1) % 50 == 0:
                stats = self.archive.get_statistics()
                print(f"  [{iteration+1}/{n_iterations}] Coverage: {stats['coverage']*100:.1f}%, "
                      f"Occupied: {stats['num_occupied']}, Avg: {stats['avg_fitness']:.1f}, "
                      f"Max: {stats['max_fitness']:.1f}")

        # Log iteration results
        stats = self.archive.get_statistics()
        self.tracker.log_result(
            result_type="map_elites_iterations",
            metrics={
                "n_iterations": n_iterations,
                "successful_additions": self.successful_additions,
                "failed_additions": self.failed_additions,
                "final_coverage": stats['coverage'],
                "final_avg_fitness": stats['avg_fitness'],
                "final_max_fitness": stats['max_fitness']
            },
            description=f"Completed {n_iterations} MAP-Elites iterations"
        )

    def _random_empty_cell(self) -> Optional[Tuple[int, int, int, int]]:
        """Find a random empty cell in the archive"""
        max_attempts = 100
        for _ in range(max_attempts):
            cell = tuple(random.randint(0, dim - 1) for dim in self.grid_dimensions)
            if cell not in self.archive.archive:
                return cell
        return None

    def get_results(self) -> Dict:
        """Get experiment results and statistics"""
        stats = self.archive.get_statistics()

        return {
            'experiment': 'LLM-Guided MAP-Elites',
            'grid_dimensions': list(self.grid_dimensions),
            'total_cells': int(np.prod(self.grid_dimensions)),
            'statistics': stats,
            'evaluations': self.total_evaluations,
            'successful_additions': self.successful_additions,
            'failed_additions': self.failed_additions,
            'best_logos': [
                {
                    'logo_id': entry.logo_id,
                    'fitness': float(entry.fitness),
                    'behavior': list(entry.behavior),
                    'raw_behavior': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                    for k, v in entry.raw_behavior.items()},
                    'generation': int(entry.generation)
                }
                for entry in self.archive.get_best_logos(10)
            ]
        }

    def save_results(self, output_dir: Optional[str] = None):
        """Save experiment results"""
        if output_dir is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_dir = f"/home/luis/svg-logo-ai/experiments/map_elites_{timestamp}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save archive
        self.archive.save_to_disk(str(output_path))

        # Save experiment summary
        results = self.get_results()
        with open(output_path / "experiment_summary.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {output_path}")

        self.tracker.log_step(
            step_type="save_results",
            description=f"Saved experiment results to {output_path}",
            metadata={"output_dir": str(output_path)}
        )

        return output_path

    def finalize(self):
        """Finalize experiment and export tracking"""
        trace_path = self.tracker.finalize()
        print(f"\n📊 Experiment trace: {trace_path}")
        return trace_path


def main():
    """Run full LLM-ME-Logo experiment"""
    print("="*80)
    print("LLM-GUIDED MAP-ELITES FOR LOGO GENERATION")
    print("="*80)

    # Create experiment
    experiment = LLMMELogo(
        company_name="InnovateTech",
        industry="technology",
        grid_dimensions=(10, 10, 10, 10),  # Full 4D grid
        use_llm=False,  # Set to True when API key available
        experiment_name="llm_me_logo_v1"
    )

    # Initialize archive
    experiment.initialize_archive(n_random=200)

    # Run iterations
    experiment.run_iterations(n_iterations=500)

    # Get and print results
    results = experiment.get_results()
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Grid: {results['grid_dimensions']} ({results['total_cells']:,} total cells)")
    print(f"Coverage: {results['statistics']['coverage']*100:.1f}%")
    print(f"Occupied: {results['statistics']['num_occupied']}")
    print(f"Avg Fitness: {results['statistics']['avg_fitness']:.2f}")
    print(f"Max Fitness: {results['statistics']['max_fitness']:.2f}")
    print(f"Total Evaluations: {results['evaluations']}")

    print("\nTop 5 Logos:")
    for i, logo in enumerate(results['best_logos'][:5], 1):
        print(f"  {i}. {logo['logo_id']}: fitness={logo['fitness']:.2f}, behavior={logo['behavior']}")

    # Save results
    output_path = experiment.save_results()

    # Finalize
    trace_path = experiment.finalize()

    print("\n" + "="*80)
    print(f"📁 Results: {output_path}")
    print(f"📊 Trace: {trace_path}")
    print("="*80)


if __name__ == "__main__":
    main()
