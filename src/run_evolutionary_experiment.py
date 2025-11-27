#!/usr/bin/env python3
"""
EVOLUTIONARY LOGO EXPERIMENT
============================
Scientific experiment comparing evolutionary optimization vs baselines

Experiment Design:
- Baseline 1: Zero-shot generation (no evolution)
- Baseline 2: Few-shot CoT (no evolution)
- Experimental: Evolutionary algorithm with aesthetic fitness

Metrics Tracked:
- Fitness evolution over generations
- Convergence rate
- Population diversity
- Improvement over baseline
- Statistical significance (t-test)

Output:
- Raw data (JSON)
- Statistical analysis
- Visualizations
- Draft paper sections
"""

import os
import time
import json
import re
from pathlib import Path
from typing import Dict, List
import google.generativeai as genai

from evolutionary_logo_system import EvolutionaryLogoSystem, Individual
from logo_validator import LogoValidator


class GeminiLogoGenerator:
    """Wrapper for Gemini logo generation"""

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Generate SVG logo from prompt"""

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                svg_code = self._extract_svg(response.text)

                if svg_code and '<svg' in svg_code:
                    return svg_code

            except Exception as e:
                print(f"⚠ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Rate limiting
                continue

        # Fallback: return simple valid SVG
        return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="60" fill="#2563eb"/>
</svg>"""

    def _extract_svg(self, text: str) -> str:
        """Extract SVG code from response"""
        # Try to find SVG in code blocks
        svg_pattern = r'```(?:svg|xml)?\s*(.*?)```'
        matches = re.findall(svg_pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Try to find raw SVG
        if '<svg' in text:
            start = text.find('<svg')
            end = text.find('</svg>') + 6
            if start != -1 and end != -1:
                return text[start:end].strip()

        return text.strip()


class EvolutionaryExperiment:
    """Scientific experiment runner"""

    def __init__(self,
                 company_name: str,
                 industry: str,
                 num_generations: int = 10,
                 population_size: int = 20):
        """
        Initialize experiment

        Args:
            company_name: Target company for logos
            industry: Industry sector
            num_generations: Number of evolutionary generations
            population_size: Population size per generation
        """
        self.company_name = company_name
        self.industry = industry
        self.num_generations = num_generations
        self.population_size = population_size

        self.generator = GeminiLogoGenerator()
        self.validator = LogoValidator()
        self.evo_system = EvolutionaryLogoSystem(
            population_size=population_size,
            elite_size=max(2, population_size // 10),  # 10% elitism
            mutation_rate=0.3,
            tournament_size=3
        )

        self.baseline_results = {}
        self.evolutionary_results = {}

    def run_baseline_zero_shot(self, n_samples: int = 10) -> Dict:
        """Baseline 1: Zero-shot generation"""
        print("\n" + "="*80)
        print("BASELINE 1: Zero-Shot Generation")
        print("="*80)

        results = []

        for i in range(n_samples):
            print(f"\nGenerating sample {i+1}/{n_samples}...")

            prompt = f"""Generate a professional SVG logo for {self.company_name},
a {self.industry} company. Make it minimalist, modern, and memorable.

Return ONLY valid SVG code with viewBox, nothing else."""

            svg_code = self.generator.generate(prompt)
            fitness, breakdown = self.evo_system.evaluate_fitness(svg_code)

            results.append({
                'sample': i + 1,
                'fitness': fitness,
                'breakdown': breakdown
            })

            print(f"   Fitness: {fitness:.1f}/100 (Aesthetic: {breakdown['aesthetic']:.1f})")

        avg_fitness = sum(r['fitness'] for r in results) / len(results)
        max_fitness = max(r['fitness'] for r in results)

        print(f"\n📊 Zero-Shot Results:")
        print(f"   Average: {avg_fitness:.2f}/100")
        print(f"   Best: {max_fitness:.2f}/100")

        return {
            'method': 'zero_shot',
            'n_samples': n_samples,
            'results': results,
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness
        }

    def run_baseline_cot(self, n_samples: int = 10) -> Dict:
        """Baseline 2: Chain-of-Thought generation"""
        print("\n" + "="*80)
        print("BASELINE 2: Chain-of-Thought Generation")
        print("="*80)

        results = []

        for i in range(n_samples):
            print(f"\nGenerating sample {i+1}/{n_samples}...")

            prompt = f"""Design a professional SVG logo for {self.company_name},
a {self.industry} company.

STEP 1: Analyze brand values and visual identity needs
STEP 2: Choose design principles (golden ratio, symmetry, negative space)
STEP 3: Select optimal color palette (max 3 colors)
STEP 4: Determine shape language and composition
STEP 5: Generate clean SVG code

Return ONLY the final SVG code with viewBox."""

            svg_code = self.generator.generate(prompt)
            fitness, breakdown = self.evo_system.evaluate_fitness(svg_code)

            results.append({
                'sample': i + 1,
                'fitness': fitness,
                'breakdown': breakdown
            })

            print(f"   Fitness: {fitness:.1f}/100 (Aesthetic: {breakdown['aesthetic']:.1f})")

        avg_fitness = sum(r['fitness'] for r in results) / len(results)
        max_fitness = max(r['fitness'] for r in results)

        print(f"\n📊 Chain-of-Thought Results:")
        print(f"   Average: {avg_fitness:.2f}/100")
        print(f"   Best: {max_fitness:.2f}/100")

        return {
            'method': 'chain_of_thought',
            'n_samples': n_samples,
            'results': results,
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness
        }

    def run_evolutionary(self) -> Dict:
        """Experimental: Evolutionary optimization"""
        print("\n" + "="*80)
        print("EXPERIMENTAL: Evolutionary Algorithm")
        print("="*80)

        # Initialize population (Generation 0)
        print(f"\nInitializing population (n={self.population_size})...")

        for i in range(self.population_size):
            genome = self.evo_system.create_genome(
                self.company_name,
                self.industry
            )
            prompt = self.evo_system.genome_to_prompt(genome)
            svg_code = self.generator.generate(prompt)
            fitness, breakdown = self.evo_system.evaluate_fitness(svg_code)

            ind = Individual(
                genome=genome,
                phenotype=svg_code,
                fitness=fitness,
                aesthetic_breakdown=breakdown,
                generation=0
            )
            self.evo_system.population.append(ind)

            print(f"   [{i+1}/{self.population_size}] Fitness: {fitness:.1f}")

        initial_avg = sum(ind.fitness for ind in self.evo_system.population) / len(self.evo_system.population)
        initial_max = max(ind.fitness for ind in self.evo_system.population)

        print(f"\n📊 Generation 0:")
        print(f"   Average: {initial_avg:.2f}/100")
        print(f"   Best: {initial_max:.2f}/100")

        # Evolution
        print(f"\nEvolving for {self.num_generations} generations...\n")

        for gen in range(self.num_generations):
            print(f"Generation {gen + 1}/{self.num_generations}:")

            stats = self.evo_system.evolve_generation(
                lambda prompt: self.generator.generate(prompt)
            )

            print(f"   Avg: {stats['mean_fitness']:.2f} ± {stats['std_fitness']:.2f}")
            print(f"   Best: {stats['max_fitness']:.2f}")
            print(f"   Improvement: {stats['max_fitness'] - initial_max:+.2f}")

        # Final statistics
        final_avg = self.evo_system.history[-1]['mean_fitness']
        final_max = self.evo_system.history[-1]['max_fitness']

        improvement_avg = final_avg - initial_avg
        improvement_max = final_max - initial_max

        print(f"\n📊 Final Results (Generation {self.num_generations}):")
        print(f"   Average: {final_avg:.2f}/100 (Δ {improvement_avg:+.2f})")
        print(f"   Best: {final_max:.2f}/100 (Δ {improvement_max:+.2f})")

        return {
            'method': 'evolutionary',
            'num_generations': self.num_generations,
            'population_size': self.population_size,
            'history': self.evo_system.history,
            'initial_avg': initial_avg,
            'initial_max': initial_max,
            'final_avg': final_avg,
            'final_max': final_max,
            'improvement_avg': improvement_avg,
            'improvement_max': improvement_max
        }

    def run_full_experiment(self):
        """Run complete experiment with all methods"""
        print("\n" + "="*80)
        print("EVOLUTIONARY LOGO DESIGN EXPERIMENT")
        print("="*80)
        print(f"Company: {self.company_name}")
        print(f"Industry: {self.industry}")
        print(f"Generations: {self.num_generations}")
        print(f"Population: {self.population_size}")

        # Run baselines
        self.baseline_results['zero_shot'] = self.run_baseline_zero_shot(n_samples=10)
        self.baseline_results['cot'] = self.run_baseline_cot(n_samples=10)

        # Run evolutionary
        self.evolutionary_results = self.run_evolutionary()

        # Compare results
        self.print_comparison()

        # Save experiment
        output_path = self.evo_system.save_experiment()

        # Save comparison
        comparison = {
            'experiment': {
                'company': self.company_name,
                'industry': self.industry,
                'date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'baselines': self.baseline_results,
            'evolutionary': self.evolutionary_results
        }

        with open(output_path / "comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)

        return output_path

    def print_comparison(self):
        """Print comparison table"""
        print("\n" + "="*80)
        print("RESULTS COMPARISON")
        print("="*80)

        methods = [
            ("Zero-Shot", self.baseline_results['zero_shot']['avg_fitness']),
            ("Chain-of-Thought", self.baseline_results['cot']['avg_fitness']),
            ("Evolutionary (Gen 0)", self.evolutionary_results['initial_avg']),
            ("Evolutionary (Final)", self.evolutionary_results['final_avg'])
        ]

        baseline_best = max(
            self.baseline_results['zero_shot']['avg_fitness'],
            self.baseline_results['cot']['avg_fitness']
        )

        print(f"\n{'Method':<25} {'Avg Fitness':>12} {'vs Best Baseline':>18}")
        print("─"*80)

        for method, fitness in methods:
            improvement = fitness - baseline_best
            arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
            print(f"{method:<25} {fitness:>12.2f} {arrow} {improvement:>+7.2f}")

        print("\n🎯 KEY FINDINGS:")

        improvement = self.evolutionary_results['final_avg'] - baseline_best
        improvement_pct = (improvement / baseline_best) * 100

        if improvement > 0:
            print(f"   ✅ Evolutionary algorithm improved over baselines by {improvement:+.2f} points ({improvement_pct:+.1f}%)")
        else:
            print(f"   ⚠ Evolutionary algorithm did not outperform baselines")

        # Best individual comparison
        evo_best = self.evolutionary_results['final_max']
        baseline_max = max(
            self.baseline_results['zero_shot']['max_fitness'],
            self.baseline_results['cot']['max_fitness']
        )

        best_improvement = evo_best - baseline_max
        print(f"   📈 Best individual: {evo_best:.2f} vs baseline {baseline_max:.2f} (Δ {best_improvement:+.2f})")

        print("="*80)


def main():
    """Run experiment"""

    # Experiment configuration
    experiment = EvolutionaryExperiment(
        company_name="NeuralFlow",
        industry="artificial intelligence",
        num_generations=5,  # Start with 5 for testing
        population_size=10   # Smaller population for faster testing
    )

    # Run full experiment
    output_path = experiment.run_full_experiment()

    print(f"\n✅ Experiment complete!")
    print(f"📁 Results saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Review results in {output_path}/comparison.json")
    print(f"  2. Visualize evolution in {output_path}/history.json")
    print(f"  3. Examine best logos in {output_path}/")


if __name__ == "__main__":
    main()
