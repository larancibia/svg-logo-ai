#!/usr/bin/env python3
"""
Basic LLM-QD Logo System Example

The simplest possible usage - generate a few logos with Quality-Diversity.

Prerequisites:
    - pip install -r requirements.txt
    - export GOOGLE_API_KEY="your-key"

Usage:
    python examples/example_basic.py
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_qd_logo_system import LLMQDLogoSystem


def main():
    """Run a basic LLM-QD experiment."""

    print("=" * 60)
    print("LLM-QD Logo System - Basic Example")
    print("=" * 60)
    print()

    # Create system with minimal configuration
    print("Initializing LLM-QD system...")
    system = LLMQDLogoSystem(
        company_name="TechCorp",
        archive_dimensions=(5, 5, 5, 5),  # Small archive for quick demo
        num_iterations=10,                 # Few iterations
        batch_size=5                       # Small batches
    )

    # Run experiment
    print("\nRunning Quality-Diversity experiment...")
    print("This will take about 2-3 minutes...")
    print()

    results = system.run_experiment()

    # Display results
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print()
    print(f"Logos Generated: {len(results['archive'])}")
    print(f"Coverage: {results['coverage']:.1%}")
    print(f"Best Fitness: {results['best_fitness']:.1f}/100")
    print(f"Average Fitness: {results['avg_fitness']:.1f}/100")
    print(f"QD-Score: {results['qd_score']:.1f}")
    print()

    # Show top 3 logos
    print("Top 3 Logos:")
    print("-" * 60)
    for i, logo in enumerate(results['top_logos'][:3], 1):
        print(f"{i}. Fitness: {logo['fitness']:.1f}")
        print(f"   Behavior: {logo['behavior']}")
        print(f"   File: {logo['file_path']}")
        print()

    print(f"\nAll logos saved to: {results['output_dir']}")
    print(f"Visualizations saved to: {results['output_dir']}/heatmaps/")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
