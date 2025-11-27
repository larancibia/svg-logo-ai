#!/usr/bin/env python3
"""
RAG-ENHANCED EVOLUTIONARY LOGO EXPERIMENT
==========================================
Combines Retrieval-Augmented Generation with evolutionary algorithms

System Architecture:
1. Knowledge Base: ChromaDB storing successful logo genomes + SVGs
2. RAG Retrieval: Find similar successful examples when generating new logos
3. Few-Shot Enhancement: Provide retrieved examples to LLM
4. Evolutionary Process: Standard genetic algorithm with RAG-enhanced generation
5. Full Tracking: Log everything in ChromaDB using ExperimentTracker

Expected Improvement: +10-15% over baseline (target: ~95-100/100)
"""

import os
import time
import json
import re
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import google.generativeai as genai
import chromadb
from chromadb.config import Settings

from evolutionary_logo_system import EvolutionaryLogoSystem, Individual
from logo_validator import LogoValidator
from experiment_tracker import ExperimentTracker


class ChromaDBKnowledgeBase:
    """ChromaDB-based knowledge base for successful logos"""

    def __init__(self, db_path: str = "/home/luis/svg-logo-ai/chroma_db/logos"):
        """Initialize ChromaDB knowledge base"""
        Path(db_path).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create collection for logo knowledge
        try:
            self.collection = self.client.get_collection("successful_logos")
        except:
            self.collection = self.client.create_collection(
                name="successful_logos",
                metadata={"description": "High-fitness logos from evolutionary experiments"}
            )

    def add_logo(self, logo_id: str, genome: Dict, svg_code: str,
                 fitness: float, aesthetic_breakdown: Dict, generation: int):
        """Add a successful logo to the knowledge base"""

        # Create searchable document from genome
        document = self._genome_to_text(genome)

        # Store with full metadata
        metadata = {
            "fitness": fitness,
            "generation": generation,
            "style": ",".join(genome.get('style_keywords', [])),
            "colors": ",".join(genome.get('color_palette', [])),
            "principles": ",".join(genome.get('design_principles', [])),
            "complexity": genome.get('complexity_target', 0),
            "aesthetic_score": aesthetic_breakdown.get('aesthetic', 0),
            "golden_ratio": aesthetic_breakdown.get('golden_ratio', 0),
            "color_harmony": aesthetic_breakdown.get('color_harmony', 0),
        }

        # Store SVG and genome as JSON in metadata
        metadata['svg_code'] = svg_code
        metadata['genome_json'] = json.dumps(genome)

        self.collection.add(
            ids=[logo_id],
            documents=[document],
            metadatas=[metadata]
        )

        print(f"Added logo {logo_id} to knowledge base (fitness: {fitness:.1f})")

    def retrieve_similar(self, genome: Dict, n_results: int = 3) -> List[Dict]:
        """Retrieve similar successful logos for few-shot learning"""

        query_text = self._genome_to_text(genome)

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        if not results['ids'][0]:
            return []

        # Parse results
        similar_logos = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            similar_logos.append({
                'id': results['ids'][0][i],
                'genome': json.loads(metadata['genome_json']),
                'svg_code': metadata['svg_code'],
                'fitness': metadata['fitness'],
                'distance': results['distances'][0][i] if 'distances' in results else 0
            })

        return similar_logos

    def _genome_to_text(self, genome: Dict) -> str:
        """Convert genome to searchable text"""
        parts = [
            f"Company: {genome.get('company', 'unknown')}",
            f"Industry: {genome.get('industry', 'unknown')}",
            f"Style: {' '.join(genome.get('style_keywords', []))}",
            f"Colors: {' '.join(genome.get('color_palette', []))}",
            f"Design Principles: {' '.join(genome.get('design_principles', []))}",
            f"Complexity: {genome.get('complexity_target', 0)}",
            f"Color Harmony: {genome.get('color_harmony_type', 'unknown')}"
        ]
        return " | ".join(parts)

    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        count = self.collection.count()

        if count == 0:
            return {"count": 0, "avg_fitness": 0, "max_fitness": 0}

        all_logos = self.collection.get()
        fitnesses = [m['fitness'] for m in all_logos['metadatas']]

        return {
            "count": count,
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "max_fitness": max(fitnesses),
            "min_fitness": min(fitnesses)
        }


class RAGEnhancedGenerator:
    """Gemini generator enhanced with RAG few-shot examples"""

    def __init__(self, knowledge_base: ChromaDBKnowledgeBase, tracker: ExperimentTracker):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.kb = knowledge_base
        self.tracker = tracker

        self.rag_enabled = True
        self.n_examples = 3  # Number of few-shot examples

    def generate(self, prompt: str, genome: Dict = None, max_retries: int = 3) -> str:
        """Generate SVG with RAG enhancement"""

        # Retrieve similar examples if RAG enabled and genome provided
        examples = []
        if self.rag_enabled and genome is not None:
            examples = self.kb.retrieve_similar(genome, n_results=self.n_examples)

            if examples:
                fitness_values = [e['fitness'] for e in examples]
                self.tracker.log_step(
                    step_type="rag_retrieval",
                    description=f"Retrieved {len(examples)} similar logos for few-shot learning",
                    metadata={
                        "n_examples": len(examples),
                        "avg_fitness": sum(fitness_values) / len(fitness_values),
                        "max_fitness": max(fitness_values),
                        "min_fitness": min(fitness_values)
                    }
                )

        # Enhance prompt with few-shot examples
        enhanced_prompt = self._build_enhanced_prompt(prompt, examples)

        # Generate with retries
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(enhanced_prompt)
                svg_code = self._extract_svg(response.text)

                if svg_code and '<svg' in svg_code:
                    return svg_code

            except Exception as e:
                print(f"⚠ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue

        # Fallback
        return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="60" fill="#2563eb"/>
</svg>"""

    def _build_enhanced_prompt(self, base_prompt: str, examples: List[Dict]) -> str:
        """Build prompt with few-shot examples"""

        if not examples:
            return base_prompt

        # Add few-shot examples section
        enhanced = "You are an expert logo designer. Here are examples of successful, high-quality logos:\n\n"

        for i, example in enumerate(examples, 1):
            genome = example['genome']
            enhanced += f"EXAMPLE {i} (Fitness: {example['fitness']:.1f}/100):\n"
            enhanced += f"Style: {', '.join(genome.get('style_keywords', []))}\n"
            enhanced += f"Colors: {', '.join(genome.get('color_palette', []))}\n"
            enhanced += f"Principles: {', '.join(genome.get('design_principles', []))}\n"
            enhanced += f"SVG:\n{example['svg_code'][:500]}...\n\n"

        enhanced += "="*60 + "\n\n"
        enhanced += "Now, using these examples as inspiration (but creating something NEW), "
        enhanced += base_prompt

        return enhanced

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


class RAGEvolutionaryExperiment:
    """RAG-enhanced evolutionary experiment"""

    def __init__(self,
                 company_name: str,
                 industry: str,
                 num_generations: int = 5,
                 population_size: int = 10,
                 knowledge_base_path: str = "/home/luis/svg-logo-ai/chroma_db/logos"):
        """Initialize RAG-enhanced experiment"""

        self.company_name = company_name
        self.industry = industry
        self.num_generations = num_generations
        self.population_size = population_size

        # Initialize components
        self.tracker = ExperimentTracker(
            experiment_name="rag_evolutionary",
            base_dir="/home/luis/svg-logo-ai"
        )

        self.kb = ChromaDBKnowledgeBase(knowledge_base_path)
        self.generator = RAGEnhancedGenerator(self.kb, self.tracker)
        self.validator = LogoValidator()

        self.evo_system = EvolutionaryLogoSystem(
            population_size=population_size,
            elite_size=max(2, population_size // 10),
            mutation_rate=0.3,
            tournament_size=3
        )

        # Log initialization
        kb_stats = self.kb.get_stats()
        self.tracker.log_decision(
            decision="Use RAG-enhanced evolutionary approach",
            rationale="Combine evolutionary optimization with few-shot learning from successful examples",
            alternatives=[
                "Pure evolutionary without RAG",
                "Pure RAG without evolution",
                "Rule-based generation"
            ],
            metadata={
                "kb_count": kb_stats['count'],
                "kb_avg_fitness": kb_stats['avg_fitness'],
                "kb_max_fitness": kb_stats['max_fitness'],
                "population_size": population_size,
                "num_generations": num_generations
            }
        )

    def initialize_kb_from_experiment(self, experiment_path: str):
        """Load successful logos from previous experiment into knowledge base"""

        print(f"\nInitializing knowledge base from: {experiment_path}")

        # Load final population
        pop_file = Path(experiment_path) / "final_population.json"
        if not pop_file.exists():
            print(f"⚠ No final population found at {pop_file}")
            return

        with open(pop_file, 'r') as f:
            population = json.load(f)

        # Add successful logos (fitness >= 87)
        added = 0
        for individual in population:
            if individual['fitness'] >= 87:
                # Load SVG file
                svg_file = Path(experiment_path) / f"{individual['id']}.svg"
                if svg_file.exists():
                    with open(svg_file, 'r') as f:
                        svg_code = f.read()

                    self.kb.add_logo(
                        logo_id=individual['id'],
                        genome=individual['genome'],
                        svg_code=svg_code,
                        fitness=individual['fitness'],
                        aesthetic_breakdown=individual['aesthetic_breakdown'],
                        generation=individual['generation']
                    )
                    added += 1

        stats = self.kb.get_stats()
        print(f"\n✅ Added {added} successful logos to knowledge base")
        print(f"📊 KB Stats: {stats['count']} logos, avg fitness: {stats['avg_fitness']:.1f}")

        self.tracker.log_step(
            step_type="kb_initialization",
            description=f"Initialized knowledge base with {added} successful logos",
            metadata={
                "logos_added": added,
                "kb_count": stats['count'],
                "kb_avg_fitness": stats['avg_fitness'],
                "kb_max_fitness": stats['max_fitness']
            }
        )

    def run_rag_evolutionary(self) -> Dict:
        """Run RAG-enhanced evolutionary optimization"""

        print("\n" + "="*80)
        print("RAG-ENHANCED EVOLUTIONARY ALGORITHM")
        print("="*80)

        # Initialize population (Generation 0)
        print(f"\nInitializing population (n={self.population_size})...")

        for i in range(self.population_size):
            genome = self.evo_system.create_genome(
                self.company_name,
                self.industry
            )
            prompt = self.evo_system.genome_to_prompt(genome)

            # Generate with RAG enhancement
            svg_code = self.generator.generate(prompt, genome=genome)
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

            self.tracker.log_step(
                step_type="individual_generation",
                description=f"Generated individual {i+1} with RAG",
                metadata={
                    "individual_id": ind.id,
                    "fitness": fitness,
                    "generation": 0
                }
            )

        initial_avg = sum(ind.fitness for ind in self.evo_system.population) / len(self.evo_system.population)
        initial_max = max(ind.fitness for ind in self.evo_system.population)

        print(f"\n📊 Generation 0:")
        print(f"   Average: {initial_avg:.2f}/100")
        print(f"   Best: {initial_max:.2f}/100")

        self.tracker.log_result(
            result_type="generation_stats",
            metrics={
                "generation": 0,
                "avg_fitness": initial_avg,
                "max_fitness": initial_max,
                "min_fitness": min(ind.fitness for ind in self.evo_system.population)
            },
            description="Initial population statistics"
        )

        # Evolution with RAG
        print(f"\nEvolving for {self.num_generations} generations with RAG...\n")

        for gen in range(self.num_generations):
            print(f"Generation {gen + 1}/{self.num_generations}:")

            # Custom evolution with RAG-enhanced generation
            stats = self._evolve_generation_with_rag(gen + 1)

            print(f"   Avg: {stats['mean_fitness']:.2f} ± {stats['std_fitness']:.2f}")
            print(f"   Best: {stats['max_fitness']:.2f}")
            print(f"   Improvement: {stats['max_fitness'] - initial_max:+.2f}")

            self.tracker.log_result(
                result_type="generation_stats",
                metrics={
                    "generation": gen + 1,
                    "avg_fitness": stats['mean_fitness'],
                    "max_fitness": stats['max_fitness'],
                    "min_fitness": stats['min_fitness'],
                    "std_fitness": stats['std_fitness']
                },
                description=f"Generation {gen + 1} statistics with RAG"
            )

        # Final statistics
        final_avg = self.evo_system.history[-1]['mean_fitness']
        final_max = self.evo_system.history[-1]['max_fitness']

        improvement_avg = final_avg - initial_avg
        improvement_max = final_max - initial_max

        print(f"\n📊 Final Results (Generation {self.num_generations}):")
        print(f"   Average: {final_avg:.2f}/100 (Δ {improvement_avg:+.2f})")
        print(f"   Best: {final_max:.2f}/100 (Δ {improvement_max:+.2f})")

        self.tracker.log_result(
            result_type="final_results",
            metrics={
                "initial_avg": initial_avg,
                "initial_max": initial_max,
                "final_avg": final_avg,
                "final_max": final_max,
                "improvement_avg": improvement_avg,
                "improvement_max": improvement_max,
                "improvement_pct": (improvement_avg / initial_avg) * 100
            },
            description="RAG-enhanced evolutionary experiment final results"
        )

        return {
            'method': 'rag_evolutionary',
            'num_generations': self.num_generations,
            'population_size': self.population_size,
            'history': self.evo_system.history,
            'initial_avg': initial_avg,
            'initial_max': initial_max,
            'final_avg': final_avg,
            'final_max': final_max,
            'improvement_avg': improvement_avg,
            'improvement_max': improvement_max,
            'kb_stats': self.kb.get_stats()
        }

    def _evolve_generation_with_rag(self, generation: int) -> Dict:
        """Evolve one generation with RAG-enhanced generation"""

        # Sort by fitness
        self.evo_system.population.sort(key=lambda x: x.fitness, reverse=True)

        # Elite preservation
        new_population = self.evo_system.population[:self.evo_system.elite_size]

        # Generate offspring with RAG
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.evo_system.tournament_selection(self.evo_system.population)
            parent2 = self.evo_system.tournament_selection(self.evo_system.population)

            # Crossover
            child_genome = self.evo_system.crossover(parent1, parent2)

            # Mutation
            if random.random() < self.evo_system.mutation_rate:
                child_genome = self.evo_system.mutate(child_genome)

            # Generate phenotype with RAG
            prompt = self.evo_system.genome_to_prompt(child_genome)
            svg_code = self.generator.generate(prompt, genome=child_genome)
            fitness, breakdown = self.evo_system.evaluate_fitness(svg_code)

            child = Individual(
                genome=child_genome,
                phenotype=svg_code,
                fitness=fitness,
                aesthetic_breakdown=breakdown,
                generation=generation,
                parent_ids=[parent1.id, parent2.id]
            )

            new_population.append(child)

        self.evo_system.population = new_population

        # Statistics
        fitnesses = [ind.fitness for ind in self.evo_system.population]
        stats = {
            'generation': generation,
            'mean_fitness': sum(fitnesses) / len(fitnesses),
            'max_fitness': max(fitnesses),
            'min_fitness': min(fitnesses),
            'std_fitness': float(np.std(fitnesses))
        }

        self.evo_system.history.append(stats)
        return stats

    def save_results(self, output_dir: Optional[str] = None):
        """Save experiment results"""

        if output_dir is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_dir = f"/home/luis/svg-logo-ai/experiments/rag_experiment_{timestamp}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save final population
        final_pop = []
        for ind in self.evo_system.population:
            final_pop.append({
                'id': ind.id,
                'genome': ind.genome,
                'fitness': ind.fitness,
                'aesthetic_breakdown': ind.aesthetic_breakdown,
                'generation': ind.generation,
                'parent_ids': ind.parent_ids
            })

            # Save SVG
            svg_file = output_path / f"{ind.id}.svg"
            with open(svg_file, 'w') as f:
                f.write(ind.phenotype)

        with open(output_path / "final_population.json", 'w') as f:
            json.dump(final_pop, f, indent=2)

        # Save history
        with open(output_path / "history.json", 'w') as f:
            json.dump(self.evo_system.history, f, indent=2)

        # Save KB stats
        with open(output_path / "kb_stats.json", 'w') as f:
            json.dump(self.kb.get_stats(), f, indent=2)

        print(f"\n✅ Results saved to: {output_path}")

        self.tracker.log_step(
            step_type="save_results",
            description=f"Saved experiment results to {output_path}",
            metadata={"output_dir": str(output_path)}
        )

        return output_path

    def finalize(self):
        """Finalize experiment and export tracking data"""
        trace_path = self.tracker.finalize()
        print(f"\n📊 Experiment trace: {trace_path}")
        return trace_path


def main():
    """Run RAG-enhanced evolutionary experiment"""

    print("="*80)
    print("RAG-ENHANCED EVOLUTIONARY LOGO EXPERIMENT")
    print("="*80)

    # Initialize experiment
    experiment = RAGEvolutionaryExperiment(
        company_name="NeuralFlow",
        industry="artificial intelligence",
        num_generations=5,  # Full scale run
        population_size=20  # Full population size
    )

    # Initialize knowledge base from previous successful experiment
    experiment.initialize_kb_from_experiment(
        "/home/luis/svg-logo-ai/experiments/experiment_20251127_053108"
    )

    # Run experiment
    results = experiment.run_rag_evolutionary()

    # Save results
    output_path = experiment.save_results()

    # Finalize tracking
    trace_path = experiment.finalize()

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Method: RAG-Enhanced Evolutionary")
    print(f"Generations: {results['num_generations']}")
    print(f"Population: {results['population_size']}")
    print(f"\nResults:")
    print(f"  Initial: Avg={results['initial_avg']:.2f}, Max={results['initial_max']:.2f}")
    print(f"  Final:   Avg={results['final_avg']:.2f}, Max={results['final_max']:.2f}")
    print(f"  Improvement: {results['improvement_avg']:+.2f} ({(results['improvement_avg']/results['initial_avg']*100):+.1f}%)")
    print(f"\nKnowledge Base: {results['kb_stats']['count']} logos, avg={results['kb_stats']['avg_fitness']:.1f}")
    print(f"\n📁 Results: {output_path}")
    print(f"📊 Trace: {trace_path}")
    print("="*80)


if __name__ == "__main__":
    main()
