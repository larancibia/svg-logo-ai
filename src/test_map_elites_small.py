#!/usr/bin/env python3
"""
Small-Scale MAP-Elites Test
============================
Test the MAP-Elites system with a smaller grid for faster validation.

Grid: 5x5x5x5 = 625 cells
Iterations: 100
Expected coverage: 10-20%
"""

import sys
from pathlib import Path

from map_elites_experiment import LLMMELogo


def main():
    """Run small-scale test"""
    print("="*80)
    print("MAP-ELITES SMALL-SCALE TEST")
    print("Grid: 5x5x5x5 (625 cells)")
    print("="*80)

    # Create test experiment with smaller grid
    experiment = LLMMELogo(
        company_name="TestCorp",
        industry="technology",
        grid_dimensions=(5, 5, 5, 5),  # Small 4D grid
        use_llm=False,  # Mock mode (no API calls)
        experiment_name="map_elites_test_small"
    )

    # Initialize with fewer random logos
    print("\n[Step 1/3] Initializing archive...")
    experiment.initialize_archive(n_random=50)

    # Run fewer iterations
    print("\n[Step 2/3] Running iterations...")
    experiment.run_iterations(n_iterations=100)

    # Get results
    print("\n[Step 3/3] Collecting results...")
    results = experiment.get_results()

    # Print summary
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Grid: {results['grid_dimensions']} ({results['total_cells']} total cells)")
    print(f"Coverage: {results['statistics']['coverage']*100:.1f}% ({results['statistics']['num_occupied']}/{results['total_cells']} cells)")
    print(f"Avg Fitness: {results['statistics']['avg_fitness']:.2f}")
    print(f"Max Fitness: {results['statistics']['max_fitness']:.2f}")
    print(f"Min Fitness: {results['statistics']['min_fitness']:.2f}")
    print(f"Total Evaluations: {results['evaluations']}")
    print(f"Successful Additions: {results['successful_additions']}")
    print(f"Failed Additions: {results['failed_additions']}")

    # Print behavior distribution of top logos
    print("\n" + "-"*80)
    print("Top 10 Logos (Behavior Diversity Check):")
    print("-"*80)
    print("Rank | Logo ID           | Fitness | Complexity | Style | Symmetry | Color")
    print("-"*80)
    for i, logo in enumerate(results['best_logos'][:10], 1):
        b = logo['behavior']
        print(f"{i:4d} | {logo['logo_id']:17s} | {logo['fitness']:7.2f} | "
              f"{b[0]:10d} | {b[1]:5d} | {b[2]:8d} | {b[3]:5d}")

    # Validation checks
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)

    checks_passed = 0
    checks_total = 0

    # Check 1: Coverage > 5%
    checks_total += 1
    if results['statistics']['coverage'] > 0.05:
        print("✓ Coverage > 5%")
        checks_passed += 1
    else:
        print(f"✗ Coverage too low: {results['statistics']['coverage']*100:.1f}%")

    # Check 2: Average fitness reasonable (> 60)
    checks_total += 1
    if results['statistics']['avg_fitness'] > 60:
        print("✓ Average fitness > 60")
        checks_passed += 1
    else:
        print(f"✗ Average fitness too low: {results['statistics']['avg_fitness']:.2f}")

    # Check 3: Behavioral diversity (different bins in top 10)
    checks_total += 1
    unique_behaviors = len(set(tuple(logo['behavior']) for logo in results['best_logos'][:10]))
    if unique_behaviors >= 5:
        print(f"✓ Behavioral diversity: {unique_behaviors}/10 unique behaviors in top 10")
        checks_passed += 1
    else:
        print(f"✗ Low behavioral diversity: only {unique_behaviors}/10 unique")

    # Check 4: Some logos in different complexity bins
    checks_total += 1
    complexity_bins = set(logo['behavior'][0] for logo in results['best_logos'][:10])
    if len(complexity_bins) >= 2:
        print(f"✓ Complexity diversity: {len(complexity_bins)} different bins")
        checks_passed += 1
    else:
        print(f"✗ No complexity diversity")

    print("\n" + "="*80)
    print(f"VALIDATION RESULT: {checks_passed}/{checks_total} checks passed")
    print("="*80)

    # Save results
    output_path = experiment.save_results()
    trace_path = experiment.finalize()

    print(f"\n📁 Results: {output_path}")
    print(f"📊 Trace: {trace_path}")

    # Return success/failure code
    return 0 if checks_passed == checks_total else 1


if __name__ == "__main__":
    sys.exit(main())
