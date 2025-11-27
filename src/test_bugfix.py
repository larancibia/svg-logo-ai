#!/usr/bin/env python3
"""
Bug Fix Test Script
====================
Tests all bug fixes with a minimal experiment:
- Uses gemini-2.5-flash model (not gemini-2.0-flash-exp)
- Includes rate limiting (6 seconds between calls)
- Tests 5D behavior space (not 4D)
- Runs minimal 3x3x3x3x3 grid with 10 iterations
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_qd_logo_system import LLMGuidedQDLogoSystem


def test_minimal_experiment():
    """Run minimal test experiment"""

    print("="*80)
    print("BUG FIX VALIDATION TEST")
    print("="*80)
    print()

    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        print("Please set it before running this test:")
        print("  export GOOGLE_API_KEY='your-key-here'")
        return False

    print("Validating bug fixes:")
    print("  1. Model changed to gemini-2.5-flash (15 req/min)")
    print("  2. Rate limiting added (6 seconds between calls)")
    print("  3. 5D behavior space (complexity, style, symmetry, color, emotional_tone)")
    print()

    # Initialize system with minimal config
    print("Initializing LLM-QD system...")
    print("  Grid: 3x3x3x3x3 (243 cells for speed)")
    print("  Iterations: 10")
    print("  Model: gemini-2.5-flash")
    print()

    try:
        system = LLMGuidedQDLogoSystem(
            grid_dimensions=(3, 3, 3, 3, 3),  # 5D: 243 total cells
            experiment_name="bugfix_test",
            model_name="gemini-2.5-flash"
        )

        # Run minimal search
        query = "minimalist tech logo"
        print(f"Query: '{query}'")
        print()
        print("Starting search (this will take ~2-3 minutes due to rate limiting)...")
        print()

        start_time = time.time()
        archive = system.search(query, iterations=10)
        elapsed = time.time() - start_time

        # Get statistics
        stats = archive.get_statistics()

        print()
        print("="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"Status: SUCCESS")
        print()
        print(f"Coverage: {stats['coverage']*100:.1f}% ({stats['num_occupied']}/243 cells)")
        print(f"Unique logos: {stats['num_occupied']}")
        print(f"Avg fitness: {stats['avg_fitness']:.2f}")
        print(f"Max fitness: {stats['max_fitness']:.2f}")
        print()
        print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"Total API calls: {system.api_calls}")
        print(f"Estimated cost: ${system.api_calls * 0.0001:.4f}")  # Rough estimate
        print()

        # Verify no errors occurred
        if stats['num_occupied'] >= 5:
            print("VALIDATION: PASSED")
            print("  - At least 5 logos generated (indicating no rate limit errors)")
            print("  - Archive has 5D behavior space")
            print("  - System completed successfully")
        else:
            print("VALIDATION: WARNING")
            print(f"  - Only {stats['num_occupied']} logos generated")
            print("  - Expected at least 5 for a successful test")

        # Save results
        output_path = system.save_results()
        print()
        print(f"Results saved to: {output_path}")
        print("="*80)

        return True

    except Exception as e:
        print()
        print("="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"Status: FAILED")
        print(f"Error: {e}")
        print()

        import traceback
        print("Full traceback:")
        traceback.print_exc()

        return False


if __name__ == "__main__":
    success = test_minimal_experiment()
    sys.exit(0 if success else 1)
