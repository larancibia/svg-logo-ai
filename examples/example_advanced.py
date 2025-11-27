#!/usr/bin/env python3
"""
Advanced LLM-QD Logo System Example

Demonstrates all features:
- Custom configuration
- Search strategies
- RAG enhancement
- Natural language queries
- Advanced visualization

Prerequisites:
    - pip install -r requirements.txt
    - export GOOGLE_API_KEY="your-key"
    - Knowledge base initialized (python src/initialize_rag_kb.py)

Usage:
    python examples/example_advanced.py
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_qd_logo_system import LLMQDLogoSystem
from qd_search_strategies import CuriosityDrivenSearch, NoveltySearch
from rag_evolutionary_system import RAGEvolutionarySystem
from qd_visualization import QDVisualizer


def run_basic_qd():
    """Example 1: Basic Quality-Diversity with default settings."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Quality-Diversity")
    print("=" * 70)

    system = LLMQDLogoSystem(
        company_name="InnovateCo",
        archive_dimensions=(10, 10, 10, 10),
        num_iterations=20,
        batch_size=10
    )

    results = system.run_experiment()

    print(f"\nGenerated {len(results['archive'])} logos")
    print(f"Coverage: {results['coverage']:.1%}")
    print(f"Best fitness: {results['best_fitness']:.1f}")

    return results


def run_curiosity_driven():
    """Example 2: Curiosity-driven exploration strategy."""
    print("\n" + "=" * 70)
    print("Example 2: Curiosity-Driven Exploration")
    print("=" * 70)
    print("Focuses on under-explored regions of behavioral space")

    system = LLMQDLogoSystem(
        company_name="ExploreTech",
        archive_dimensions=(10, 10, 10, 10),
        num_iterations=20,
        search_strategy=CuriosityDrivenSearch()
    )

    results = system.run_experiment()

    print(f"\nCoverage: {results['coverage']:.1%}")
    print("This should show higher coverage than random search")

    return results


def run_novelty_search():
    """Example 3: Novelty search for maximum diversity."""
    print("\n" + "=" * 70)
    print("Example 3: Novelty Search")
    print("=" * 70)
    print("Maximizes behavioral diversity")

    system = LLMQDLogoSystem(
        company_name="DiverseCo",
        archive_dimensions=(10, 10, 10, 10),
        num_iterations=20,
        search_strategy=NoveltySearch()
    )

    results = system.run_experiment()

    print(f"\nBehavioral diversity achieved: {results['diversity_score']:.2f}")

    return results


def run_rag_enhanced():
    """Example 4: RAG-enhanced evolution for higher quality."""
    print("\n" + "=" * 70)
    print("Example 4: RAG-Enhanced Evolution")
    print("=" * 70)
    print("Uses retrieval-augmented generation for better logos")

    system = RAGEvolutionarySystem(
        company_name="QualityCo",
        population_size=10,
        generations=5,
        use_rag=True,
        top_k_examples=3
    )

    results = system.run_experiment()

    print(f"\nBest fitness: {results['best_fitness']:.1f}")
    print("RAG typically achieves 2-3 points higher fitness")

    return results


def run_with_custom_config():
    """Example 5: Custom configuration for specific use case."""
    print("\n" + "=" * 70)
    print("Example 5: Custom Configuration")
    print("=" * 70)

    # Custom behavioral space targeting specific design styles
    system = LLMQDLogoSystem(
        company_name="CustomCo",
        archive_dimensions=(8, 8, 8, 8),  # Slightly smaller
        num_iterations=15,
        batch_size=8,

        # Custom behavioral ranges
        complexity_range=(15, 40),         # Medium complexity only
        style_range=(0.0, 0.5),            # Geometric bias
        symmetry_range=(0.5, 1.0),         # Prefer symmetric
        color_range=(0.3, 0.7),            # Moderate color richness

        # Quality constraints
        min_fitness=75,                    # Only keep good logos

        # Mutation settings
        mutation_rate=0.3,                 # Higher mutation
        semantic_mutation_prob=0.7,        # Prefer semantic mutations
    )

    results = system.run_experiment()

    print(f"\nLogos match custom constraints:")
    print(f"- Geometric style bias")
    print(f"- High symmetry")
    print(f"- Moderate complexity")
    print(f"Coverage: {results['coverage']:.1%}")

    return results


def visualize_results(results, output_dir):
    """Example 6: Advanced visualization."""
    print("\n" + "=" * 70)
    print("Example 6: Advanced Visualization")
    print("=" * 70)

    visualizer = QDVisualizer(results['archive'], output_dir)

    # Generate all visualizations
    print("\nGenerating visualizations...")

    # 2D heatmaps for all dimension pairs
    visualizer.plot_2d_heatmap('complexity', 'style')
    visualizer.plot_2d_heatmap('symmetry', 'color_richness')

    # Fitness distribution
    visualizer.plot_fitness_distribution()

    # Coverage over time
    visualizer.plot_coverage_evolution(results['coverage_history'])

    # 3D scatter plot
    visualizer.plot_3d_scatter('complexity', 'style', 'symmetry')

    print(f"Visualizations saved to: {output_dir}/visualizations/")


def compare_strategies():
    """Example 7: Compare different search strategies."""
    print("\n" + "=" * 70)
    print("Example 7: Strategy Comparison")
    print("=" * 70)

    strategies = {
        'Random': None,
        'Curiosity': CuriosityDrivenSearch(),
        'Novelty': NoveltySearch(),
    }

    results = {}

    for name, strategy in strategies.items():
        print(f"\nRunning {name} search...")

        system = LLMQDLogoSystem(
            company_name=f"Compare{name}",
            archive_dimensions=(8, 8, 8, 8),
            num_iterations=15,
            search_strategy=strategy
        )

        results[name] = system.run_experiment()

    # Compare results
    print("\n" + "-" * 70)
    print("Comparison Results:")
    print("-" * 70)
    print(f"{'Strategy':<15} {'Coverage':<12} {'Best Fitness':<15} {'QD-Score':<12}")
    print("-" * 70)

    for name, result in results.items():
        print(f"{name:<15} {result['coverage']:>10.1%} "
              f"{result['best_fitness']:>13.1f} "
              f"{result['qd_score']:>10.1f}")

    return results


def natural_language_queries():
    """Example 8: Generate logos using natural language."""
    print("\n" + "=" * 70)
    print("Example 8: Natural Language Queries")
    print("=" * 70)

    from nl_query_parser import NLQueryParser

    parser = NLQueryParser()

    queries = [
        "Create a geometric, highly symmetric logo with minimal colors",
        "Generate an organic, flowing design with rich color palette",
        "Make a simple, iconic logo with moderate symmetry",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")

        # Parse query to behavioral constraints
        constraints = parser.parse(query)
        print(f"Parsed constraints: {constraints}")

        # Generate logo matching constraints
        system = LLMQDLogoSystem(
            company_name=f"NLQuery{i}",
            archive_dimensions=(8, 8, 8, 8),
            num_iterations=5,
            **constraints  # Apply parsed constraints
        )

        results = system.run_experiment()
        best_logo = results['top_logos'][0]

        print(f"Generated logo with behavior: {best_logo['behavior']}")
        print(f"Fitness: {best_logo['fitness']:.1f}")


def main():
    """Run all examples."""
    print("=" * 70)
    print("LLM-QD Logo System - Advanced Examples")
    print("=" * 70)
    print("\nThis script demonstrates all major features.")
    print("Each example takes 2-5 minutes to run.")
    print()

    # Create output directory
    output_dir = Path("examples_output")
    output_dir.mkdir(exist_ok=True)

    try:
        # Run examples
        print("\nRunning examples...")

        # Basic examples
        results1 = run_basic_qd()
        results2 = run_curiosity_driven()
        results3 = run_novelty_search()
        results4 = run_rag_enhanced()
        results5 = run_with_custom_config()

        # Advanced examples
        visualize_results(results1, output_dir / "basic_qd")
        comparison_results = compare_strategies()
        natural_language_queries()

        print("\n" + "=" * 70)
        print("All Examples Completed!")
        print("=" * 70)
        print(f"\nResults saved to: {output_dir.absolute()}")
        print("\nKey Takeaways:")
        print("- Basic QD: Good balance of coverage and quality")
        print("- Curiosity: Best for exploration")
        print("- Novelty: Maximum diversity")
        print("- RAG: Highest quality logos")
        print("- Custom: Targeted design constraints")
        print("- Natural Language: User-friendly interface")

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nMake sure:")
        print("1. GOOGLE_API_KEY is set")
        print("2. Dependencies are installed")
        print("3. Knowledge base is initialized (for RAG)")
        raise


if __name__ == "__main__":
    main()
