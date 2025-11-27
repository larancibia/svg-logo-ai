#!/usr/bin/env python3
"""
EVOLUTIONARY LOGO DESIGN SYSTEM
================================
Sistema de diseño evolutivo de logos SVG con métricas científicas

Paper: "Evolutionary SVG Logo Optimization with Aesthetic Fitness Functions"

Architecture:
- Population: 20 individuals per generation
- Fitness: Aesthetic score (0-100) from logo_validator v2.0
- Selection: Tournament selection (k=3)
- Crossover: Prompt mixing + parameter blending
- Mutation: Intelligent mutations guided by design principles
- Elitism: Top 20% preserved

Scientific Tracking:
- Generation-by-generation metrics
- Statistical significance tests
- Convergence analysis
- Diversity metrics
"""

import random
import json
import copy
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass, asdict

from logo_validator import LogoValidator
from logo_metadata import LogoMetadata


@dataclass
class Individual:
    """Individual in the evolutionary population"""
    genome: Dict  # Genetic representation
    phenotype: str  # SVG code
    fitness: float = 0.0
    aesthetic_breakdown: Dict = None
    generation: int = 0
    parent_ids: List[str] = None

    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []
        # Generate unique ID
        self.id = f"gen{self.generation}_{datetime.now().strftime('%H%M%S%f')}"


class EvolutionaryLogoSystem:
    """
    Evolutionary system for logo design optimization
    """

    def __init__(self,
                 population_size: int = 20,
                 elite_size: int = 4,
                 mutation_rate: float = 0.3,
                 tournament_size: int = 3):
        """
        Initialize evolutionary system

        Args:
            population_size: Number of individuals per generation
            elite_size: Number of best individuals to preserve
            mutation_rate: Probability of mutation (0-1)
            tournament_size: Size of tournament for selection
        """
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

        self.validator = LogoValidator()
        self.metadata = LogoMetadata("../output/evolutionary_metadata.json")

        self.population: List[Individual] = []
        self.generation = 0
        self.history = []  # Track all generations

        # Design vocabulary for mutations
        self.style_words = [
            "minimalist", "modern", "elegant", "professional", "bold",
            "geometric", "organic", "abstract", "symbolic", "sophisticated",
            "clean", "sleek", "innovative", "timeless", "refined"
        ]

        self.color_schemes = {
            "tech": ["#2563eb", "#3b82f6", "#60a5fa"],  # Blues
            "health": ["#10b981", "#34d399", "#6ee7b7"],  # Greens
            "finance": ["#1e40af", "#3730a3", "#4338ca"],  # Deep blues
            "creative": ["#8b5cf6", "#a78bfa", "#c4b5fd"],  # Purples
            "energy": ["#f59e0b", "#fbbf24", "#fcd34d"],  # Oranges/yellows
        }

        self.design_principles = [
            "golden_ratio",
            "rule_of_thirds",
            "symmetry",
            "asymmetry_balance",
            "negative_space",
            "gestalt_closure",
            "figure_ground"
        ]

    def create_genome(self,
                      company: str,
                      industry: str,
                      style_preference: str = None) -> Dict:
        """
        Create a genome (genetic representation) for a logo

        Genome structure:
        {
            'company': str,
            'industry': str,
            'style_keywords': List[str],
            'color_palette': List[str],
            'design_principles': List[str],
            'complexity_target': int,
            'golden_ratio_weight': float,
            'color_harmony_type': str
        }
        """
        # Random style selection
        styles = random.sample(self.style_words, k=random.randint(2, 4))
        if style_preference:
            styles.insert(0, style_preference)

        # Color palette based on industry
        industry_colors = self.color_schemes.get(
            industry.lower(),
            random.choice(list(self.color_schemes.values()))
        )
        colors = random.sample(industry_colors, k=random.randint(1, 2))

        # Design principles
        principles = random.sample(self.design_principles, k=random.randint(1, 3))

        genome = {
            'company': company,
            'industry': industry,
            'style_keywords': styles,
            'color_palette': colors,
            'design_principles': principles,
            'complexity_target': random.randint(20, 40),  # Optimal range
            'golden_ratio_weight': random.uniform(0.5, 1.0),
            'color_harmony_type': random.choice(['monochrome', 'complementary', 'analogous'])
        }

        return genome

    def genome_to_prompt(self, genome: Dict) -> str:
        """Convert genome to LLM prompt for logo generation"""

        prompt = f"""Generate a professional SVG logo for {genome['company']}, a {genome['industry']} company.

DESIGN REQUIREMENTS:
Style: {', '.join(genome['style_keywords'])}
Target complexity: {genome['complexity_target']} elements (optimal range: 20-40)

AESTHETIC PRINCIPLES:
"""

        if 'golden_ratio' in genome['design_principles']:
            prompt += f"- Apply Golden Ratio (φ=1.618) with weight {genome['golden_ratio_weight']:.2f}\n"

        if 'symmetry' in genome['design_principles']:
            prompt += "- Use symmetrical composition\n"
        elif 'asymmetry_balance' in genome['design_principles']:
            prompt += "- Use asymmetrical balance\n"

        if 'negative_space' in genome['design_principles']:
            prompt += "- Leverage negative space creatively\n"

        if 'gestalt_closure' in genome['design_principles']:
            prompt += "- Apply Gestalt principle of closure\n"

        prompt += f"\nCOLOR SCHEME:\n"
        prompt += f"Type: {genome['color_harmony_type']}\n"
        prompt += f"Palette: {', '.join(genome['color_palette'])}\n"
        prompt += "Maximum 3 colors for professional appearance\n"

        prompt += """
OUTPUT FORMAT:
Provide clean, valid SVG code with:
- viewBox for scalability
- Simple, memorable shapes
- Professional color palette
- Optimized for 16x16 to 1024x1024 rendering

Return ONLY the SVG code, nothing else.
"""

        return prompt

    def evaluate_fitness(self, svg_code: str) -> Tuple[float, Dict]:
        """
        Evaluate fitness using aesthetic metrics v2.0

        Returns:
            fitness: Score 0-100
            breakdown: Detailed metrics
        """
        results = self.validator.validate_all(svg_code)

        fitness = results['final_score']

        breakdown = {
            'total': fitness,
            'legacy_score': results['legacy_score'],
            'aesthetic': results['level5_aesthetic']['score'],
            'golden_ratio': results['level5_aesthetic']['golden_ratio'],
            'color_harmony': results['level5_aesthetic']['color_harmony'],
            'visual_interest': results['level5_aesthetic']['visual_interest'],
            'professional': results['level4_professional']['score'],
            'technical': (results['level1_xml']['score'] * 0.4 +
                         results['level2_svg']['score'] * 0.3 +
                         results['level3_quality']['score'] * 0.3)
        }

        return fitness, breakdown

    def tournament_selection(self, population: List[Individual], k: int = None) -> Individual:
        """
        Tournament selection: Pick k random individuals, return best
        """
        if k is None:
            k = self.tournament_size

        tournament = random.sample(population, k)
        return max(tournament, key=lambda ind: ind.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Dict:
        """
        Genetic crossover: Combine genomes of two parents

        Strategy:
        - Mix style keywords
        - Blend numeric parameters
        - Randomly choose categorical parameters
        """
        child_genome = {}

        # Company and industry from parent1 (same for all in this context)
        child_genome['company'] = parent1.genome['company']
        child_genome['industry'] = parent1.genome['industry']

        # Mix style keywords (take from both parents)
        all_styles = parent1.genome['style_keywords'] + parent2.genome['style_keywords']
        child_genome['style_keywords'] = random.sample(
            list(set(all_styles)),
            k=min(4, len(set(all_styles)))
        )

        # Mix color palettes
        all_colors = parent1.genome['color_palette'] + parent2.genome['color_palette']
        child_genome['color_palette'] = random.sample(
            list(set(all_colors)),
            k=min(2, len(set(all_colors)))
        )

        # Mix design principles
        all_principles = parent1.genome['design_principles'] + parent2.genome['design_principles']
        child_genome['design_principles'] = random.sample(
            list(set(all_principles)),
            k=min(3, len(set(all_principles)))
        )

        # Blend numeric parameters (average)
        child_genome['complexity_target'] = int(
            (parent1.genome['complexity_target'] + parent2.genome['complexity_target']) / 2
        )
        child_genome['golden_ratio_weight'] = (
            parent1.genome['golden_ratio_weight'] + parent2.genome['golden_ratio_weight']
        ) / 2

        # Randomly choose categorical
        child_genome['color_harmony_type'] = random.choice([
            parent1.genome['color_harmony_type'],
            parent2.genome['color_harmony_type']
        ])

        return child_genome

    def mutate(self, genome: Dict) -> Dict:
        """
        Genetic mutation: Random modifications to genome

        Mutation types:
        1. Add/remove style keyword
        2. Change color
        3. Modify design principle
        4. Adjust numeric parameters
        """
        mutated = copy.deepcopy(genome)

        # Mutation 1: Style keywords (30% chance)
        if random.random() < 0.3:
            if random.random() < 0.5 and len(mutated['style_keywords']) > 2:
                # Remove a keyword
                mutated['style_keywords'].pop(random.randint(0, len(mutated['style_keywords'])-1))
            else:
                # Add a new keyword
                new_style = random.choice([s for s in self.style_words
                                         if s not in mutated['style_keywords']])
                mutated['style_keywords'].append(new_style)

        # Mutation 2: Color (20% chance)
        if random.random() < 0.2:
            available_colors = self.color_schemes.get(
                mutated['industry'].lower(),
                random.choice(list(self.color_schemes.values()))
            )
            mutated['color_palette'] = random.sample(available_colors, k=random.randint(1, 2))

        # Mutation 3: Design principles (25% chance)
        if random.random() < 0.25:
            if random.random() < 0.5 and len(mutated['design_principles']) > 1:
                mutated['design_principles'].pop(0)
            else:
                new_principle = random.choice([p for p in self.design_principles
                                              if p not in mutated['design_principles']])
                mutated['design_principles'].append(new_principle)

        # Mutation 4: Numeric parameters (40% chance)
        if random.random() < 0.4:
            # Complexity target ±5
            mutated['complexity_target'] = max(15, min(45,
                mutated['complexity_target'] + random.randint(-5, 5)))

        if random.random() < 0.3:
            # Golden ratio weight ±0.2
            mutated['golden_ratio_weight'] = max(0.3, min(1.0,
                mutated['golden_ratio_weight'] + random.uniform(-0.2, 0.2)))

        # Mutation 5: Color harmony type (15% chance)
        if random.random() < 0.15:
            mutated['color_harmony_type'] = random.choice(
                ['monochrome', 'complementary', 'analogous']
            )

        return mutated

    def evolve_generation(self, generator_func) -> Dict:
        """
        Evolve one generation

        Args:
            generator_func: Function that takes prompt and returns SVG code

        Returns:
            Generation statistics
        """
        new_population = []

        # Elitism: Keep best individuals
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        elites = sorted_pop[:self.elite_size]

        for elite in elites:
            elite_copy = Individual(
                genome=copy.deepcopy(elite.genome),
                phenotype=elite.phenotype,
                fitness=elite.fitness,
                aesthetic_breakdown=elite.aesthetic_breakdown,
                generation=self.generation + 1,
                parent_ids=[elite.id]
            )
            new_population.append(elite_copy)

        # Generate rest through crossover + mutation
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.tournament_selection(self.population)
            parent2 = self.tournament_selection(self.population)

            # Crossover
            child_genome = self.crossover(parent1, parent2)

            # Mutation
            if random.random() < self.mutation_rate:
                child_genome = self.mutate(child_genome)

            # Generate phenotype (SVG)
            prompt = self.genome_to_prompt(child_genome)
            svg_code = generator_func(prompt)

            # Evaluate fitness
            fitness, breakdown = self.evaluate_fitness(svg_code)

            # Create individual
            child = Individual(
                genome=child_genome,
                phenotype=svg_code,
                fitness=fitness,
                aesthetic_breakdown=breakdown,
                generation=self.generation + 1,
                parent_ids=[parent1.id, parent2.id]
            )

            new_population.append(child)

        # Update population
        self.population = new_population
        self.generation += 1

        # Calculate statistics
        fitnesses = [ind.fitness for ind in self.population]
        stats = {
            'generation': self.generation,
            'mean_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses),
            'best_individual_id': max(self.population, key=lambda x: x.fitness).id
        }

        self.history.append(stats)

        return stats

    def save_experiment(self, output_dir: str = "../experiments"):
        """Save complete experiment data for scientific analysis"""
        output_path = Path(output_dir) / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config = {
            'population_size': self.population_size,
            'elite_size': self.elite_size,
            'mutation_rate': self.mutation_rate,
            'tournament_size': self.tournament_size,
            'total_generations': self.generation
        }

        with open(output_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        # Save history (statistics per generation)
        # Convert numpy types to Python types for JSON serialization
        history_serializable = []
        for gen_stats in self.history:
            gen_dict = {}
            for key, value in gen_stats.items():
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    gen_dict[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    gen_dict[key] = float(value)
                else:
                    gen_dict[key] = value
            history_serializable.append(gen_dict)

        with open(output_path / "history.json", 'w') as f:
            json.dump(history_serializable, f, indent=2)

        # Save final population
        population_data = []
        for ind in self.population:
            # Convert numpy types in aesthetic_breakdown
            aesthetic_breakdown = {}
            if ind.aesthetic_breakdown:
                for key, value in ind.aesthetic_breakdown.items():
                    if isinstance(value, (np.integer, np.int64, np.int32)):
                        aesthetic_breakdown[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64, np.float32)):
                        aesthetic_breakdown[key] = float(value)
                    else:
                        aesthetic_breakdown[key] = value

            data = {
                'id': ind.id,
                'genome': ind.genome,
                'fitness': float(ind.fitness) if isinstance(ind.fitness, (np.floating, np.float64, np.float32)) else ind.fitness,
                'aesthetic_breakdown': aesthetic_breakdown,
                'generation': int(ind.generation) if isinstance(ind.generation, (np.integer, np.int64, np.int32)) else ind.generation,
                'parent_ids': ind.parent_ids
            }
            population_data.append(data)

            # Save SVG files
            svg_path = output_path / f"{ind.id}.svg"
            with open(svg_path, 'w') as f:
                f.write(ind.phenotype)

        with open(output_path / "final_population.json", 'w') as f:
            json.dump(population_data, f, indent=2)

        print(f"\n✅ Experiment saved to: {output_path}")

        return output_path


def demo():
    """Demo of evolutionary system"""

    # Mock generator for testing
    def mock_generator(prompt: str) -> str:
        """Returns a simple SVG for testing"""
        return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="60" fill="none" stroke="#2563eb" stroke-width="8"/>
  <path d="M70 100 Q100 70 130 100" fill="none" stroke="#2563eb" stroke-width="6"/>
</svg>"""

    # Create system
    evo = EvolutionaryLogoSystem(population_size=10, elite_size=2)

    # Initialize population
    print("Creating initial population...")
    for i in range(10):
        genome = evo.create_genome("TechCorp", "technology", "modern")
        prompt = evo.genome_to_prompt(genome)
        svg = mock_generator(prompt)
        fitness, breakdown = evo.evaluate_fitness(svg)

        ind = Individual(
            genome=genome,
            phenotype=svg,
            fitness=fitness,
            aesthetic_breakdown=breakdown,
            generation=0
        )
        evo.population.append(ind)

    # Evolve for 3 generations
    print("\nEvolving...")
    for gen in range(3):
        stats = evo.evolve_generation(mock_generator)
        print(f"Generation {stats['generation']}: "
              f"Mean={stats['mean_fitness']:.1f}, "
              f"Max={stats['max_fitness']:.1f}")

    # Save experiment
    evo.save_experiment()


if __name__ == "__main__":
    demo()
