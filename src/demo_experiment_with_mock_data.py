#!/usr/bin/env python3
"""
DEMO: Simulated Evolutionary Experiment
=======================================
Genera resultados realistas sin necesitar API real
Útil para demostración y desarrollo del paper
"""

import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)


def generate_mock_experiment():
    """Generate realistic experimental data"""

    print("="*80)
    print("DEMO: EVOLUTIONARY LOGO EXPERIMENT (Simulated Data)")
    print("="*80)
    print("\nNote: Using simulated data for demonstration")
    print("Real experiment requires valid Google Gemini API key\n")

    # Configuration
    config = {
        "population_size": 20,
        "elite_size": 4,
        "mutation_rate": 0.3,
        "tournament_size": 3,
        "total_generations": 10
    }

    # Baseline 1: Zero-Shot (realistic scores based on current system)
    print("="*80)
    print("BASELINE 1: Zero-Shot Generation (Simulated)")
    print("="*80)

    zero_shot_scores = [
        80.5, 82.3, 84.1, 81.7, 83.2,
        85.0, 79.8, 86.3, 82.5, 81.9
    ]

    zero_shot_results = []
    for i, score in enumerate(zero_shot_scores):
        aesthetic = score * 0.85 + random.uniform(-3, 3)  # Realistic correlation
        result = {
            'sample': i + 1,
            'fitness': score,
            'breakdown': {
                'aesthetic': aesthetic,
                'golden_ratio': aesthetic * 0.75 + random.uniform(-5, 5),
                'color_harmony': aesthetic * 1.05 + random.uniform(-3, 3),
                'visual_interest': aesthetic * 0.90 + random.uniform(-4, 4),
                'professional': score * 1.1 - random.uniform(0, 5),
                'technical': 95.0  # Usually high
            }
        }
        zero_shot_results.append(result)
        print(f"Sample {i+1}/10: Fitness={score:.1f}/100 (Aesthetic: {aesthetic:.1f})")

    zero_shot_avg = np.mean(zero_shot_scores)
    zero_shot_max = np.max(zero_shot_scores)

    print(f"\n📊 Zero-Shot Results:")
    print(f"   Average: {zero_shot_avg:.2f}/100")
    print(f"   Best: {zero_shot_max:.2f}/100")

    # Baseline 2: Chain-of-Thought (slightly better than zero-shot)
    print("\n" + "="*80)
    print("BASELINE 2: Chain-of-Thought (Simulated)")
    print("="*80)

    cot_scores = [
        82.1, 84.5, 85.8, 83.9, 86.2,
        84.0, 81.7, 87.9, 85.3, 84.6
    ]

    cot_results = []
    for i, score in enumerate(cot_scores):
        aesthetic = score * 0.88 + random.uniform(-2, 2)  # Slightly better correlation
        result = {
            'sample': i + 1,
            'fitness': score,
            'breakdown': {
                'aesthetic': aesthetic,
                'golden_ratio': aesthetic * 0.80 + random.uniform(-4, 4),
                'color_harmony': aesthetic * 1.00 + random.uniform(-2, 2),
                'visual_interest': aesthetic * 0.92 + random.uniform(-3, 3),
                'professional': score * 1.08 - random.uniform(0, 3),
                'technical': 96.0
            }
        }
        cot_results.append(result)
        print(f"Sample {i+1}/10: Fitness={score:.1f}/100 (Aesthetic: {aesthetic:.1f})")

    cot_avg = np.mean(cot_scores)
    cot_max = np.max(cot_scores)

    print(f"\n📊 Chain-of-Thought Results:")
    print(f"   Average: {cot_avg:.2f}/100")
    print(f"   Best: {cot_max:.2f}/100")

    # Evolutionary: Realistic progression over 10 generations
    print("\n" + "="*80)
    print("EXPERIMENTAL: Evolutionary Algorithm (Simulated)")
    print("="*80)

    # Generation 0 (similar to baselines)
    gen0_scores = [
        81.2, 83.5, 82.8, 80.9, 84.2,
        85.8, 79.5, 86.1, 83.7, 82.4,
        84.5, 81.8, 83.2, 80.3, 85.0,
        82.9, 84.7, 81.5, 83.9, 82.1
    ]

    initial_avg = np.mean(gen0_scores)
    initial_max = np.max(gen0_scores)
    initial_std = np.std(gen0_scores)

    print(f"\nInitializing population (n={config['population_size']})...")
    for i, score in enumerate(gen0_scores):
        print(f"   [{i+1}/{len(gen0_scores)}] Fitness: {score:.1f}")

    print(f"\n📊 Generation 0:")
    print(f"   Average: {initial_avg:.2f}/100")
    print(f"   Best: {initial_max:.2f}/100")

    # Evolution: Realistic improvement with diminishing returns
    print(f"\nEvolving for {config['total_generations']} generations...\n")

    history = []
    history.append({
        'generation': 0,
        'mean_fitness': initial_avg,
        'std_fitness': initial_std,
        'max_fitness': initial_max,
        'min_fitness': np.min(gen0_scores),
        'best_individual_id': f"gen0_mock_{int(initial_max*100)}"
    })

    # Simulate realistic evolution
    current_mean = initial_avg
    current_max = initial_max
    current_std = initial_std

    for gen in range(1, config['total_generations'] + 1):
        # Realistic improvement: fast initially, then plateau
        if gen <= 3:
            mean_improvement = random.uniform(1.2, 2.0)  # Fast improvement
            max_improvement = random.uniform(0.8, 1.5)
        elif gen <= 6:
            mean_improvement = random.uniform(0.5, 1.0)  # Moderate
            max_improvement = random.uniform(0.3, 0.8)
        else:
            mean_improvement = random.uniform(0.1, 0.4)  # Plateau
            max_improvement = random.uniform(0.1, 0.3)

        current_mean += mean_improvement
        current_max += max_improvement
        current_std *= 0.92  # Diversity decreases (convergence)

        # Add some noise
        current_mean += random.uniform(-0.2, 0.2)
        current_max += random.uniform(-0.1, 0.1)

        history.append({
            'generation': gen,
            'mean_fitness': current_mean,
            'std_fitness': current_std,
            'max_fitness': current_max,
            'min_fitness': current_mean - current_std * 1.5,
            'best_individual_id': f"gen{gen}_mock_{int(current_max*100)}"
        })

        print(f"Generation {gen}/{config['total_generations']}:")
        print(f"   Avg: {current_mean:.2f} ± {current_std:.2f}")
        print(f"   Best: {current_max:.2f}")
        print(f"   Improvement: {current_max - initial_max:+.2f}")

    final_avg = current_mean
    final_max = current_max

    improvement_avg = final_avg - initial_avg
    improvement_max = final_max - initial_max

    print(f"\n📊 Final Results (Generation {config['total_generations']}):")
    print(f"   Average: {final_avg:.2f}/100 (Δ {improvement_avg:+.2f})")
    print(f"   Best: {final_max:.2f}/100 (Δ {improvement_max:+.2f})")

    # Prepare data structures
    baselines = {
        'zero_shot': {
            'method': 'zero_shot',
            'n_samples': 10,
            'avg_fitness': float(zero_shot_avg),
            'max_fitness': float(zero_shot_max),
            'results': zero_shot_results
        },
        'cot': {
            'method': 'chain_of_thought',
            'n_samples': 10,
            'avg_fitness': float(cot_avg),
            'max_fitness': float(cot_max),
            'results': cot_results
        }
    }

    evolutionary = {
        'method': 'evolutionary',
        'num_generations': config['total_generations'],
        'population_size': config['population_size'],
        'initial_avg': float(initial_avg),
        'initial_max': float(initial_max),
        'final_avg': float(final_avg),
        'final_max': float(final_max),
        'improvement_avg': float(improvement_avg),
        'improvement_max': float(improvement_max),
        'history': history
    }

    # Comparison
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)

    baseline_best = max(zero_shot_avg, cot_avg)

    methods = [
        ("Zero-Shot", zero_shot_avg),
        ("Chain-of-Thought", cot_avg),
        ("Evolutionary (Gen 0)", initial_avg),
        ("Evolutionary (Final)", final_avg)
    ]

    print(f"\n{'Method':<25} {'Avg Fitness':>12} {'vs Best Baseline':>18}")
    print("─"*80)

    for method, fitness in methods:
        improvement = fitness - baseline_best
        arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
        print(f"{method:<25} {fitness:>12.2f} {arrow} {improvement:>+7.2f}")

    print("\n🎯 KEY FINDINGS:")

    improvement = final_avg - baseline_best
    improvement_pct = (improvement / baseline_best) * 100

    print(f"   ✅ Evolutionary algorithm improved over baselines by {improvement:+.2f} points ({improvement_pct:+.1f}%)")

    # Best individual comparison
    evo_best = final_max
    baseline_max = max(zero_shot_max, cot_max)
    best_improvement = evo_best - baseline_max

    print(f"   📈 Best individual: {evo_best:.2f} vs baseline {baseline_max:.2f} (Δ {best_improvement:+.2f})")
    print("="*80)

    # Save experiment
    output_path = Path("../experiments") / f"demo_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Save history
    with open(output_path / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # Save comparison
    comparison = {
        'experiment': {
            'company': 'NeuralFlow (DEMO)',
            'industry': 'artificial intelligence',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'note': 'Simulated data for demonstration purposes'
        },
        'baselines': baselines,
        'evolutionary': evolutionary
    }

    with open(output_path / "comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n✅ Demo experiment saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Review results in {output_path}/comparison.json")
    print(f"  2. Run analysis: cd src && python3 analyze_experiment.py")
    print(f"  3. Generate visualizations and statistics")
    print(f"\nTo run with REAL data:")
    print(f"  1. Get valid Google Gemini API key from: https://makersuite.google.com/app/apikey")
    print(f"  2. export GOOGLE_API_KEY='your-key-here'")
    print(f"  3. python3 run_evolutionary_experiment.py")

    return output_path


if __name__ == "__main__":
    demo_path = generate_mock_experiment()
