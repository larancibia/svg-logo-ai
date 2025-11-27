#!/usr/bin/env python3
"""
Custom Natural Language Query Example

Demonstrates the natural language interface for logo generation.

Prerequisites:
    - pip install -r requirements.txt
    - export GOOGLE_API_KEY="your-key"

Usage:
    python examples/example_custom_query.py
    python examples/example_custom_query.py --interactive
"""

import sys
import os
import argparse

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nl_query_parser import NLQueryParser
from llm_qd_logo_system import LLMQDLogoSystem


def demo_predefined_queries():
    """Run demo with predefined queries."""

    print("=" * 70)
    print("Natural Language Query Demo - Predefined Queries")
    print("=" * 70)
    print()

    parser = NLQueryParser()

    # Example queries covering different styles
    queries = [
        {
            "query": "Create a geometric, symmetric tech company logo with blue colors",
            "company": "TechFlow"
        },
        {
            "query": "Generate an organic, flowing logo with warm colors for a coffee shop",
            "company": "CoffeeHub"
        },
        {
            "query": "Design a minimal, iconic logo with high contrast for a finance app",
            "company": "FinVest"
        },
        {
            "query": "Make a complex, detailed logo with rich colors for a creative agency",
            "company": "CreativeStudio"
        },
        {
            "query": "Create a simple, clean logo with cool colors for a health app",
            "company": "HealthPlus"
        },
    ]

    for i, example in enumerate(queries, 1):
        print(f"\n{'=' * 70}")
        print(f"Example {i}/{len(queries)}")
        print(f"{'=' * 70}")
        print(f"\nCompany: {example['company']}")
        print(f"Query: \"{example['query']}\"")
        print()

        # Parse query
        print("Parsing query...")
        constraints = parser.parse(example['query'])

        print("\nParsed Constraints:")
        for key, value in constraints.items():
            print(f"  {key}: {value}")

        # Generate logo
        print("\nGenerating logo...")
        system = LLMQDLogoSystem(
            company_name=example['company'],
            archive_dimensions=(5, 5, 5, 5),
            num_iterations=5,
            **constraints
        )

        results = system.run_experiment()
        best = results['top_logos'][0]

        print("\nResults:")
        print(f"  Fitness: {best['fitness']:.1f}/100")
        print(f"  Complexity: {best['behavior']['complexity']:.2f}")
        print(f"  Style: {best['behavior']['style']:.2f}")
        print(f"  Symmetry: {best['behavior']['symmetry']:.2f}")
        print(f"  Color Richness: {best['behavior']['color_richness']:.2f}")
        print(f"  Saved to: {best['file_path']}")


def interactive_mode():
    """Interactive query mode."""

    print("=" * 70)
    print("Natural Language Query - Interactive Mode")
    print("=" * 70)
    print()
    print("Enter natural language queries to generate logos.")
    print("Type 'quit' or 'exit' to stop.")
    print()
    print("Example queries:")
    print("  - Create a geometric logo with blue colors")
    print("  - Generate an organic, flowing design")
    print("  - Make a minimal, symmetric logo")
    print()

    parser = NLQueryParser()

    while True:
        # Get query
        print("-" * 70)
        query = input("\nEnter your query: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if not query:
            print("Please enter a query.")
            continue

        # Get company name
        company = input("Company name (default: 'CustomCo'): ").strip()
        if not company:
            company = "CustomCo"

        print()

        try:
            # Parse query
            print("Parsing query...")
            constraints = parser.parse(query)

            print("\nParsed Constraints:")
            for key, value in constraints.items():
                print(f"  {key}: {value}")

            # Confirm
            confirm = input("\nGenerate logo with these constraints? (Y/n): ").strip()
            if confirm.lower() in ['n', 'no']:
                print("Skipping generation.")
                continue

            # Generate
            print("\nGenerating logo (this takes 2-3 minutes)...")
            system = LLMQDLogoSystem(
                company_name=company,
                archive_dimensions=(5, 5, 5, 5),
                num_iterations=5,
                **constraints
            )

            results = system.run_experiment()
            best = results['top_logos'][0]

            print("\n" + "=" * 70)
            print("Logo Generated!")
            print("=" * 70)
            print(f"\nFitness: {best['fitness']:.1f}/100")
            print(f"File: {best['file_path']}")
            print(f"\nBehavioral Characteristics:")
            for dim, value in best['behavior'].items():
                print(f"  {dim}: {value:.2f}")

        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again with a different query.")


def demo_style_variations():
    """Generate variations on a style using NL queries."""

    print("=" * 70)
    print("Style Variations Demo")
    print("=" * 70)
    print("\nGenerating variations of a geometric tech logo...")
    print()

    parser = NLQueryParser()

    base_query = "geometric tech logo"

    variations = [
        "very simple " + base_query,
        "moderately complex " + base_query,
        "highly detailed " + base_query,
    ]

    results = []

    for i, query in enumerate(variations, 1):
        print(f"\nVariation {i}: \"{query}\"")

        constraints = parser.parse(query)
        system = LLMQDLogoSystem(
            company_name=f"TechVar{i}",
            archive_dimensions=(5, 5, 5, 5),
            num_iterations=3,
            **constraints
        )

        result = system.run_experiment()
        results.append(result['top_logos'][0])

        print(f"  Complexity: {result['top_logos'][0]['behavior']['complexity']:.2f}")

    # Compare
    print("\n" + "=" * 70)
    print("Comparison:")
    print("-" * 70)
    print(f"{'Query':<30} {'Complexity':<12} {'Fitness':<10}")
    print("-" * 70)

    for query, result in zip(variations, results):
        print(f"{query:<30} {result['behavior']['complexity']:>10.2f} "
              f"{result['fitness']:>8.1f}")


def demo_color_preferences():
    """Generate logos with different color preferences."""

    print("=" * 70)
    print("Color Preference Demo")
    print("=" * 70)
    print("\nGenerating logos with different color schemes...")
    print()

    parser = NLQueryParser()

    color_queries = [
        "logo with minimal colors, almost monochrome",
        "logo with moderate color variety",
        "logo with rich, diverse color palette",
    ]

    for i, query in enumerate(color_queries, 1):
        print(f"\nQuery {i}: \"{query}\"")

        constraints = parser.parse(query)
        system = LLMQDLogoSystem(
            company_name=f"ColorTest{i}",
            archive_dimensions=(5, 5, 5, 5),
            num_iterations=3,
            **constraints
        )

        result = system.run_experiment()
        best = result['top_logos'][0]

        print(f"  Color Richness: {best['behavior']['color_richness']:.2f}")
        print(f"  Fitness: {best['fitness']:.1f}")


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Natural Language Query Examples for LLM-QD Logo System"
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help="Run in interactive mode"
    )
    parser.add_argument(
        '--demo',
        choices=['predefined', 'variations', 'colors', 'all'],
        default='all',
        help="Which demo to run (default: all)"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    else:
        if args.demo in ['predefined', 'all']:
            demo_predefined_queries()

        if args.demo in ['variations', 'all']:
            demo_style_variations()

        if args.demo in ['colors', 'all']:
            demo_color_preferences()

        print("\n" + "=" * 70)
        print("All demos completed!")
        print("=" * 70)
        print("\nTry interactive mode with: --interactive")


if __name__ == "__main__":
    main()
