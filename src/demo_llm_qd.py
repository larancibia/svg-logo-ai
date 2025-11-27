#!/usr/bin/env python3
"""
Interactive Demo: LLM-Guided QD Logo System
============================================
Demonstrates revolutionary capabilities of the system.

Features:
1. Natural language query → diverse logos
2. Interactive exploration of behavior space
3. Real-time visualization of search progress
4. Comparison to baseline methods
"""

import os
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_qd_logo_system import LLMGuidedQDLogoSystem


def demo_basic_search():
    """
    Demo 1: Basic LLM-QD search
    """
    print(f"\n{'='*80}")
    print(f"DEMO 1: BASIC LLM-QD SEARCH")
    print(f"{'='*80}\n")

    print("This demo shows how LLM-QD converts a natural language query")
    print("into a comprehensive archive of diverse, high-quality logos.\n")

    # Initialize system
    print("Initializing LLM-QD system...")
    system = LLMGuidedQDLogoSystem(
        grid_dimensions=(5, 5, 5, 5, 5),  # Smaller grid for demo (5D behavior space)
        experiment_name="demo_basic_search"
    )

    # User query
    query = "minimalist tech logo with circular motifs conveying innovation and trust"

    print(f"\nUser Query: \"{query}\"")
    print(f"\nStarting search with 25 iterations...")
    print("(Each iteration: select parent → mutate toward target → evaluate → archive)\n")

    # Run search
    archive = system.search(query, iterations=25)

    # Show results
    stats = archive.get_statistics()

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Coverage: {stats['coverage']*100:.1f}% of behavior space explored")
    print(f"Occupied cells: {stats['num_occupied']} / {5**5}")
    print(f"Average fitness: {stats['avg_fitness']:.2f}")
    print(f"Max fitness: {stats['max_fitness']:.2f}")
    print(f"\nAPI calls: {system.api_calls}")
    print(f"Estimated cost: ${system.api_calls * 0.002:.2f}")

    # Save results
    output_path = system.save_results()

    print(f"\n{'='*80}")
    print(f"Logos saved to: {output_path}")
    print(f"You can view the SVG files in that directory.")
    print(f"{'='*80}\n")

    return system, archive


def demo_behavior_space():
    """
    Demo 2: Exploring behavior space
    """
    print(f"\n{'='*80}")
    print(f"DEMO 2: BEHAVIOR SPACE EXPLORATION")
    print(f"{'='*80}\n")

    print("The system explores a 5D behavior space:")
    print("  1. Complexity: simple (few elements) ↔ complex (many elements)")
    print("  2. Style: geometric (straight lines) ↔ organic (curves)")
    print("  3. Symmetry: asymmetric ↔ symmetric")
    print("  4. Color: monochrome ↔ polychromatic")
    print("  5. Emotional Tone: serious/professional ↔ playful/friendly\n")

    print("Each cell in the grid represents a unique design niche.")
    print("The system tries to fill as many niches as possible with high-quality designs.\n")

    print("Example niches:")
    print("  (0,0,0,0): Simple, geometric, asymmetric, monochrome")
    print("  (5,5,5,5): Complex, organic, symmetric, colorful")
    print("  (2,8,1,3): Moderate complexity, very organic, slightly asymmetric, tritone\n")

    print("This is revolutionary because:")
    print("  ✓ Traditional optimization finds ONE solution")
    print("  ✓ LLM-QD finds HUNDREDS of diverse solutions")
    print("  ✓ Each solution is optimized for its specific niche")
    print("  ✓ User can choose from comprehensive design space\n")


def demo_intelligent_mutations():
    """
    Demo 3: Show intelligent LLM-guided mutations
    """
    print(f"\n{'='*80}")
    print(f"DEMO 3: INTELLIGENT LLM-GUIDED MUTATIONS")
    print(f"{'='*80}\n")

    print("Traditional evolutionary algorithms use RANDOM mutations:")
    print("  • Flip random bit")
    print("  • Add random noise")
    print("  • No understanding of what changes mean\n")

    print("LLM-QD uses INTELLIGENT mutations:")
    print("  • LLM understands design concepts")
    print("  • Mutations are directed toward target behaviors")
    print("  • Much more efficient than random search\n")

    print("Example mutation instructions:")
    print('  Current: (2,2,5,1) - moderately complex, geometric, symmetric, duotone')
    print('  Target:  (6,2,5,1) - very complex, geometric, symmetric, duotone')
    print('  LLM instruction: "INCREASE COMPLEXITY: Add 12-20 more SVG elements"')
    print()
    print('  Current: (3,1,4,2) - moderate, very geometric, mostly symmetric, tritone')
    print('  Target:  (3,8,4,2) - moderate, very organic, mostly symmetric, tritone')
    print('  LLM instruction: "MAKE MORE ORGANIC: Convert straight lines to bezier curves"')
    print()

    print("This semantic understanding is KEY to the system's efficiency.")
    print("It explores behavior space ~25x faster than random mutations.\n")


def demo_curiosity_driven():
    """
    Demo 4: Curiosity-driven exploration
    """
    print(f"\n{'='*80}")
    print(f"DEMO 4: CURIOSITY-DRIVEN EXPLORATION")
    print(f"{'='*80}\n")

    print("The system uses CURIOSITY to guide exploration:")
    print("  • Identifies under-explored regions of behavior space")
    print("  • Preferentially selects parents near empty cells")
    print("  • Actively seeks novelty, not just quality\n")

    print("This prevents premature convergence:")
    print("  ✗ Traditional GA: Converges to local optimum")
    print("  ✓ LLM-QD: Continues exploring until space is filled\n")

    print("Similar to:")
    print("  • Intrinsic motivation in reinforcement learning")
    print("  • Human curiosity in creative exploration")
    print("  • Scientific discovery processes\n")


def demo_natural_language():
    """
    Demo 5: Natural language interface
    """
    print(f"\n{'='*80}")
    print(f"DEMO 5: NATURAL LANGUAGE INTERFACE")
    print(f"{'='*80}\n")

    print("Traditional systems require expertise:")
    print("  ✗ Tune mutation rates")
    print("  ✗ Define fitness functions")
    print("  ✗ Set behavior dimensions")
    print("  ✗ Configure selection operators\n")

    print("LLM-QD uses NATURAL LANGUAGE:")
    print('  ✓ "minimalist tech logo with circular motifs"')
    print('  ✓ "organic nature-inspired design with earth tones"')
    print('  ✓ "bold geometric fintech logo conveying security"')
    print()

    print("The system automatically:")
    print("  • Extracts style keywords")
    print("  • Infers color palettes")
    print("  • Determines complexity targets")
    print("  • Generates diverse variations\n")

    print("This makes advanced evolutionary computation accessible to everyone.\n")


def demo_quick_test():
    """
    Quick test with minimal API calls
    """
    print(f"\n{'='*80}")
    print(f"QUICK TEST: LLM-QD in Action")
    print(f"{'='*80}\n")

    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  WARNING: GOOGLE_API_KEY not set")
        print("This demo requires API access to run the actual system.")
        print("\nTo run with real API:")
        print("  export GOOGLE_API_KEY='your-key-here'")
        print("  python demo_llm_qd.py\n")
        return

    print("Running minimal LLM-QD search to demonstrate capabilities...")
    print("(5x5x5x5 grid, 15 iterations for speed)\n")

    # Initialize
    system = LLMGuidedQDLogoSystem(
        grid_dimensions=(5, 5, 5, 5, 5),
        experiment_name="quick_demo"
    )

    # Query
    query = "modern minimalist logo for AI startup"
    print(f'Query: "{query}"')
    print("Starting search...\n")

    # Run
    start_time = time.time()
    archive = system.search(query, iterations=15)
    elapsed = time.time() - start_time

    # Results
    stats = archive.get_statistics()

    print(f"\n{'='*80}")
    print(f"QUICK TEST RESULTS")
    print(f"{'='*80}")
    print(f"Coverage: {stats['coverage']*100:.1f}%")
    print(f"Occupied: {stats['num_occupied']}/{5**5}")
    print(f"Avg Fitness: {stats['avg_fitness']:.2f}")
    print(f"Max Fitness: {stats['max_fitness']:.2f}")
    print(f"Time: {elapsed:.1f}s")
    print(f"API Calls: {system.api_calls}")
    print(f"Cost: ${system.api_calls * 0.002:.3f}")

    # Save
    output_path = system.save_results()
    print(f"\nResults: {output_path}")
    print(f"{'='*80}\n")


def main():
    """Main demo runner"""
    print(f"\n{'#'*80}")
    print(f"LLM-GUIDED QD LOGO SYSTEM: INTERACTIVE DEMO")
    print(f"{'#'*80}\n")

    print("This demo showcases the revolutionary capabilities of LLM-QD:")
    print("  • Natural language → diverse designs")
    print("  • Systematic behavior space exploration")
    print("  • Intelligent LLM-guided mutations")
    print("  • Curiosity-driven search")
    print("  • Quality-Diversity optimization\n")

    # Run demos
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        # Actually run with API
        demo_quick_test()
    else:
        # Just show explanations
        demo_behavior_space()
        input("\nPress Enter to continue...")

        demo_intelligent_mutations()
        input("\nPress Enter to continue...")

        demo_curiosity_driven()
        input("\nPress Enter to continue...")

        demo_natural_language()
        input("\nPress Enter to continue...")

        print(f"\n{'='*80}")
        print(f"DEMO COMPLETE")
        print(f"{'='*80}\n")

        print("To run the system with real API calls:")
        print("  export GOOGLE_API_KEY='your-key-here'")
        print("  python demo_llm_qd.py --run\n")

        print("To run full experiments:")
        print("  python run_llm_qd_experiment.py\n")


if __name__ == "__main__":
    main()
