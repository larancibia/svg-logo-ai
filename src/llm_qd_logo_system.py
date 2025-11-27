"""
LLM-Guided Quality-Diversity Logo System
=========================================
Revolutionary integration of LLM intelligence with QD systematic exploration.

This system combines:
1. LLM semantic understanding (natural language → design concepts)
2. MAP-Elites systematic exploration (comprehensive coverage)
3. Intelligent mutation (LLM-guided toward under-explored regions)
4. Multi-dimensional behavior space (5D: complexity, style, symmetry, color, aesthetic)

Key Innovation: LLM acts as intelligent search operator, not just generator.
The system uses curiosity-driven selection to explore under-represented niches.
"""

import os
import random
import copy
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import json

from map_elites_archive import MAPElitesArchive, ArchiveEntry
from behavior_characterization import BehaviorCharacterizer
from llm_guided_mutation import LLMGuidedMutator
from logo_validator import LogoValidator
from experiment_tracker import ExperimentTracker


class LLMGuidedQDLogoSystem:
    """
    Main LLM-Guided QD Logo Generation System

    Revolutionary Algorithm:
    1. Parse natural language query into design requirements
    2. Generate diverse initial population with LLM
    3. For each iteration:
       a. Curiosity-driven selection (prioritize under-explored regions)
       b. LLM-guided mutation toward target behavior
       c. Evaluate quality (fitness) and behavior
       d. Add to archive if better in its niche
    4. Return comprehensive archive of diverse, high-quality logos
    """

    def __init__(self,
                 grid_dimensions: Tuple[int, ...] = (10, 10, 10, 10, 10),
                 experiment_name: str = "llm_qd_experiment",
                 model_name: str = "gemini-2.5-flash"):
        """
        Initialize LLM-Guided QD system

        Args:
            grid_dimensions: Dimensions of MAP-Elites grid (default: 10x10x10x10x10 for 5D space)
            experiment_name: Name for tracking experiment
            model_name: LLM model to use
        """
        # Core components
        self.archive = MAPElitesArchive(dimensions=grid_dimensions)
        self.characterizer = BehaviorCharacterizer(num_bins=grid_dimensions[0])
        self.mutator = LLMGuidedMutator(model_name=model_name)
        self.validator = LogoValidator()
        self.tracker = ExperimentTracker(experiment_name=experiment_name)

        # System parameters
        self.grid_dimensions = grid_dimensions
        self.model_name = model_name
        self.iteration = 0

        # Tracking
        self.api_calls = 0
        self.total_cost_estimate = 0.0
        self.history = []

        self.tracker.log_step(
            step_type="initialization",
            description="LLM-QD System initialized",
            metadata={
                "grid_dimensions": str(grid_dimensions),
                "model_name": model_name,
                "total_grid_cells": int(np.prod(grid_dimensions))
            }
        )

    def parse_query(self, query: str) -> Dict:
        """
        Parse natural language query into design genome

        Args:
            query: Natural language description (e.g., "minimalist tech logo with circular motifs")

        Returns:
            Genome dictionary with design parameters
        """
        # Simple keyword extraction (could be enhanced with LLM parsing)
        query_lower = query.lower()

        # Extract style keywords
        style_keywords = []
        style_words = [
            "minimalist", "modern", "elegant", "professional", "bold",
            "geometric", "organic", "abstract", "playful", "sophisticated",
            "clean", "sleek", "innovative", "timeless", "refined", "vibrant"
        ]
        for word in style_words:
            if word in query_lower:
                style_keywords.append(word)

        if not style_keywords:
            style_keywords = ["modern", "professional"]

        # Extract industry hints
        industry = "technology"
        if any(w in query_lower for w in ["health", "medical", "wellness"]):
            industry = "healthcare"
        elif any(w in query_lower for w in ["finance", "fintech", "bank"]):
            industry = "finance"
        elif any(w in query_lower for w in ["creative", "design", "art"]):
            industry = "creative"
        elif any(w in query_lower for w in ["energy", "power", "electric"]):
            industry = "energy"

        # Extract color hints
        color_palette = []
        if "blue" in query_lower:
            color_palette = ["#2563eb", "#3b82f6"]
        elif "green" in query_lower:
            color_palette = ["#10b981", "#34d399"]
        elif "purple" in query_lower:
            color_palette = ["#8b5cf6", "#a78bfa"]
        elif "orange" in query_lower or "warm" in query_lower:
            color_palette = ["#f59e0b", "#fbbf24"]
        else:
            # Default based on industry
            color_schemes = {
                "technology": ["#2563eb", "#3b82f6"],
                "healthcare": ["#10b981", "#34d399"],
                "finance": ["#1e40af", "#3730a3"],
                "creative": ["#8b5cf6", "#a78bfa"],
                "energy": ["#f59e0b", "#fbbf24"]
            }
            color_palette = color_schemes.get(industry, ["#2563eb"])

        # Extract complexity hints
        complexity_target = 25  # Default
        if "simple" in query_lower or "minimal" in query_lower:
            complexity_target = 15
        elif "complex" in query_lower or "detailed" in query_lower:
            complexity_target = 40

        # Create genome
        genome = {
            "company": "Client",  # Generic
            "industry": industry,
            "style_keywords": style_keywords,
            "color_palette": color_palette,
            "design_principles": ["balance", "simplicity"],
            "complexity_target": complexity_target,
            "golden_ratio_weight": 0.8,
            "color_harmony_type": "complementary",
            "original_query": query
        }

        self.tracker.log_step(
            step_type="query_parsing",
            description=f"Parsed query: {query}",
            data=genome
        )

        return genome

    def initialize_population(self, base_genome: Dict, n: int = 20) -> int:
        """
        Generate diverse initial population

        Args:
            base_genome: Base genome from query parsing
            n: Number of initial individuals

        Returns:
            Number of individuals successfully added to archive
        """
        self.tracker.log_step(
            step_type="initialization",
            description=f"Generating {n} initial diverse individuals"
        )

        added_count = 0

        for i in range(n):
            try:
                # Create variation of base genome
                genome = self._vary_genome(base_genome)

                # Generate SVG
                svg_code = self.mutator.generate_from_genome(genome)
                self.api_calls += 1

                # Evaluate
                fitness, breakdown = self._evaluate_fitness(svg_code)
                behavior_data = self.characterizer.characterize(svg_code)

                # Add to archive
                logo_id = f"init_{i}_{datetime.now().strftime('%H%M%S%f')}"
                success = self.archive.add(
                    logo_id=logo_id,
                    svg_code=svg_code,
                    genome=genome,
                    fitness=fitness,
                    aesthetic_breakdown=breakdown,
                    behavior=behavior_data['bins'],
                    raw_behavior=behavior_data['raw_scores'],
                    generation=0,
                    parent_ids=[]
                )

                if success:
                    added_count += 1
                    self.tracker.log_result(
                        result_type="initial_individual",
                        metrics={
                            "fitness": fitness,
                            "complexity": behavior_data['raw_scores']['complexity'],
                            "behavior_bin": str(behavior_data['bins'])
                        }
                    )

                # Rate limiting: 6 seconds between calls (10 calls/min max)
                time.sleep(6)

            except Exception as e:
                self.tracker.log_step(
                    step_type="error",
                    description=f"Failed to generate initial individual {i}: {e}"
                )
                continue

        self.tracker.log_step(
            step_type="initialization_complete",
            description=f"Added {added_count}/{n} individuals to archive",
            metadata={"coverage": self.archive.get_coverage()}
        )

        return added_count

    def search(self, user_query: str, iterations: int = 100) -> MAPElitesArchive:
        """
        Main LLM-Guided QD search algorithm

        Args:
            user_query: Natural language query
            iterations: Number of search iterations

        Returns:
            Filled MAP-Elites archive
        """
        self.tracker.log_decision(
            decision=f"Starting LLM-QD search for {iterations} iterations",
            rationale="LLM-guided exploration systematically fills behavior space with high-quality diverse solutions",
            alternatives=["Random search", "Pure evolutionary", "Pure LLM generation"]
        )

        # Parse query
        base_genome = self.parse_query(user_query)

        # Initialize population
        print(f"\n{'='*80}")
        print(f"LLM-GUIDED QD SEARCH")
        print(f"{'='*80}")
        print(f"Query: {user_query}")
        print(f"Iterations: {iterations}")
        print(f"Grid dimensions: {self.grid_dimensions}")
        print(f"{'='*80}\n")

        self.initialize_population(base_genome, n=20)

        # Main search loop
        for i in range(iterations):
            self.iteration = i + 1

            try:
                # 1. Curiosity-driven selection
                parent_behavior, parent_entry = self._select_parent_curiosity()

                if parent_entry is None:
                    print(f"⚠ Iteration {self.iteration}: No parent available, skipping")
                    continue

                # 2. Select target behavior (under-explored region)
                target_behavior = self._select_target_behavior(parent_behavior)

                # 3. LLM-guided mutation toward target
                child_svg = self.mutator.mutate_toward_target(
                    source_svg=parent_entry.svg_code,
                    current_behavior=parent_behavior,
                    target_behavior=target_behavior,
                    genome=parent_entry.genome
                )
                self.api_calls += 1

                # 4. Evaluate child
                fitness, breakdown = self._evaluate_fitness(child_svg)
                behavior_data = self.characterizer.characterize(child_svg)

                # 5. Add to archive
                logo_id = f"gen{self.iteration}_{datetime.now().strftime('%H%M%S%f')}"
                success = self.archive.add(
                    logo_id=logo_id,
                    svg_code=child_svg,
                    genome=parent_entry.genome,
                    fitness=fitness,
                    aesthetic_breakdown=breakdown,
                    behavior=behavior_data['bins'],
                    raw_behavior=behavior_data['raw_scores'],
                    generation=self.iteration,
                    parent_ids=[parent_entry.logo_id]
                )

                # 6. Log progress
                stats = self.archive.get_statistics()
                self.history.append({
                    "iteration": self.iteration,
                    "coverage": stats['coverage'],
                    "num_occupied": stats['num_occupied'],
                    "avg_fitness": stats['avg_fitness'],
                    "max_fitness": stats['max_fitness'],
                    "added": success
                })

                if self.iteration % 10 == 0:
                    print(f"Iteration {self.iteration}/{iterations}: "
                          f"Coverage={stats['coverage']*100:.1f}%, "
                          f"Occupied={stats['num_occupied']}, "
                          f"Avg Fitness={stats['avg_fitness']:.1f}, "
                          f"Max Fitness={stats['max_fitness']:.1f}")

                    self.tracker.log_result(
                        result_type="checkpoint",
                        metrics={
                            "iteration": self.iteration,
                            "coverage": stats['coverage'],
                            "num_occupied": stats['num_occupied'],
                            "avg_fitness": stats['avg_fitness'],
                            "max_fitness": stats['max_fitness']
                        }
                    )

                # Rate limiting: 6 seconds between calls (10 calls/min max)
                time.sleep(6)

            except Exception as e:
                self.tracker.log_step(
                    step_type="error",
                    description=f"Iteration {self.iteration} failed: {e}"
                )
                continue

        # Final statistics
        final_stats = self.archive.get_statistics()
        print(f"\n{'='*80}")
        print(f"SEARCH COMPLETE")
        print(f"{'='*80}")
        print(f"Coverage: {final_stats['coverage']*100:.1f}%")
        print(f"Occupied cells: {final_stats['num_occupied']}/{np.prod(self.grid_dimensions)}")
        print(f"Average fitness: {final_stats['avg_fitness']:.2f}")
        print(f"Max fitness: {final_stats['max_fitness']:.2f}")
        print(f"API calls: {self.api_calls}")
        print(f"{'='*80}\n")

        self.tracker.log_result(
            result_type="final_results",
            metrics=final_stats,
            description="LLM-QD search completed"
        )

        return self.archive

    def _select_parent_curiosity(self) -> Tuple[Tuple[int, ...], Optional[ArchiveEntry]]:
        """
        Curiosity-driven parent selection
        Prioritizes parents in regions with many empty neighbors
        """
        if not self.archive.archive:
            return None, None

        # Get all occupied cells
        occupied_cells = list(self.archive.archive.keys())

        # Calculate curiosity score for each (number of empty neighbors)
        curiosity_scores = []
        for behavior in occupied_cells:
            empty_neighbors = len(self.archive.get_empty_neighbors(behavior, distance=1))
            curiosity_scores.append(empty_neighbors)

        # Weighted random selection (higher curiosity = higher probability)
        total_curiosity = sum(curiosity_scores) + len(occupied_cells)  # Add small baseline
        probabilities = [(score + 1) / total_curiosity for score in curiosity_scores]

        selected_idx = np.random.choice(len(occupied_cells), p=probabilities)
        selected_behavior = occupied_cells[selected_idx]

        return selected_behavior, self.archive.get(selected_behavior)

    def _select_target_behavior(self, current_behavior: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Select target behavior for mutation
        Prioritizes empty neighbors
        """
        # Get empty neighbors
        empty_neighbors = self.archive.get_empty_neighbors(current_behavior, distance=2)

        if empty_neighbors:
            # Select random empty neighbor
            return random.choice(empty_neighbors)
        else:
            # If no empty neighbors, select random nearby cell
            target = list(current_behavior)
            for i in range(len(target)):
                delta = random.randint(-2, 2)
                target[i] = max(0, min(self.grid_dimensions[i] - 1, target[i] + delta))
            return tuple(target)

    def _evaluate_fitness(self, svg_code: str) -> Tuple[float, Dict]:
        """Evaluate fitness using aesthetic metrics"""
        try:
            results = self.validator.validate_all(svg_code)
            fitness = results['final_score']

            breakdown = {
                'total': fitness,
                'legacy_score': results['legacy_score'],
                'aesthetic': results['level5_aesthetic']['score'],
                'golden_ratio': results['level5_aesthetic']['golden_ratio'],
                'color_harmony': results['level5_aesthetic']['color_harmony'],
                'visual_interest': results['level5_aesthetic']['visual_interest'],
                'professional': results['level4_professional']['score']
            }

            return fitness, breakdown

        except Exception as e:
            self.tracker.log_step(
                step_type="warning",
                description=f"Fitness evaluation failed: {e}"
            )
            return 50.0, {}

    def _vary_genome(self, base_genome: Dict) -> Dict:
        """Create variation of base genome for diversity"""
        genome = copy.deepcopy(base_genome)

        # Vary complexity
        genome['complexity_target'] = base_genome['complexity_target'] + random.randint(-10, 10)
        genome['complexity_target'] = max(15, min(50, genome['complexity_target']))

        # Vary style keywords (add/remove one)
        if random.random() < 0.5 and len(genome['style_keywords']) > 1:
            genome['style_keywords'].pop(random.randint(0, len(genome['style_keywords'])-1))

        # Vary golden ratio weight
        genome['golden_ratio_weight'] = base_genome['golden_ratio_weight'] + random.uniform(-0.2, 0.2)
        genome['golden_ratio_weight'] = max(0.3, min(1.0, genome['golden_ratio_weight']))

        return genome

    def generate_diverse_logos(self, query: str, n: int = 50) -> List[Dict]:
        """
        User-facing API: Generate N diverse high-quality logos

        Args:
            query: Natural language description
            n: Number of logos to return

        Returns:
            List of logo dictionaries with SVG, fitness, behavior, etc.
        """
        # Run search
        iterations = n * 5  # Oversample to ensure good coverage
        self.search(query, iterations=iterations)

        # Get best N diverse logos
        best_logos = self.archive.get_best_logos(n=n)

        results = []
        for entry in best_logos:
            results.append({
                'svg_code': entry.svg_code,
                'fitness': entry.fitness,
                'behavior': entry.behavior,
                'raw_behavior': entry.raw_behavior,
                'aesthetic_breakdown': entry.aesthetic_breakdown,
                'logo_id': entry.logo_id
            })

        return results

    def save_results(self, output_dir: str = None):
        """Save complete experiment results"""
        if output_dir is None:
            output_dir = f"/home/luis/svg-logo-ai/experiments/llm_qd_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save archive
        self.archive.save_to_disk(str(output_path))

        # Save history
        with open(output_path / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)

        # Save configuration
        config = {
            "grid_dimensions": self.grid_dimensions,
            "model_name": self.model_name,
            "total_iterations": self.iteration,
            "api_calls": self.api_calls,
            "final_statistics": self.archive.get_statistics()
        }

        with open(output_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        # Export tracker trace
        self.tracker.finalize()

        print(f"\nResults saved to: {output_path}")
        return str(output_path)


def demo():
    """Quick demo of LLM-QD system"""
    print("="*80)
    print("LLM-GUIDED QD LOGO SYSTEM DEMO")
    print("="*80)

    # Initialize system
    system = LLMGuidedQDLogoSystem(
        grid_dimensions=(5, 5, 5, 5),  # Small grid for demo
        experiment_name="demo_llm_qd"
    )

    # Run search
    query = "minimalist tech logo with circular motifs conveying innovation"
    system.search(query, iterations=10)

    # Save results
    output_path = system.save_results()

    print(f"\nDemo complete! Results saved to {output_path}")


if __name__ == "__main__":
    demo()
