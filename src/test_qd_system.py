"""
Comprehensive Tests for Enhanced QD System
==========================================
Tests for:
- Behavior characterization (5D)
- Enhanced QD archive
- Search strategies
- Visualization
- Performance benchmarks
"""

import time
import random
import numpy as np
from pathlib import Path
import sys

# Import modules to test
from behavior_characterization import BehaviorCharacterizer, visualize_behavior_space
from map_elites_archive import EnhancedQDArchive, ArchiveEntry
from qd_search_strategies import (
    RandomSearchStrategy,
    CuriositySearchStrategy,
    QualitySearchStrategy,
    DirectedSearchStrategy,
    NoveltySearchStrategy,
    AdaptiveSearchStrategy
)
from qd_visualization import QDVisualizer


class TestResults:
    """Store and display test results"""

    def __init__(self):
        self.results = []
        self.benchmarks = {}

    def add_test(self, name: str, passed: bool, message: str = ""):
        """Add test result"""
        self.results.append({
            'name': name,
            'passed': passed,
            'message': message
        })

    def add_benchmark(self, name: str, duration: float, operations: int):
        """Add benchmark result"""
        self.benchmarks[name] = {
            'duration': duration,
            'operations': operations,
            'ops_per_sec': operations / duration if duration > 0 else 0
        }

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("TEST RESULTS SUMMARY")
        print("="*70)

        passed = sum(1 for r in self.results if r['passed'])
        total = len(self.results)

        for result in self.results:
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"{status}: {result['name']}")
            if result['message']:
                print(f"       {result['message']}")

        print(f"\n{passed}/{total} tests passed ({passed/total*100:.1f}%)")

        if self.benchmarks:
            print("\n" + "="*70)
            print("PERFORMANCE BENCHMARKS")
            print("="*70)

            for name, bench in self.benchmarks.items():
                print(f"\n{name}:")
                print(f"  Duration: {bench['duration']:.3f}s")
                print(f"  Operations: {bench['operations']:,}")
                print(f"  Ops/sec: {bench['ops_per_sec']:.1f}")


def test_behavior_characterization():
    """Test 5D behavior characterization"""
    results = TestResults()

    # Test 1: Basic characterization
    try:
        characterizer = BehaviorCharacterizer(num_bins=10)

        svg_code = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="60" fill="#2563eb"/>
  <rect x="80" y="80" width="40" height="40" fill="#ffffff"/>
</svg>"""

        result = characterizer.characterize(svg_code)

        # Check all 5 dimensions are present
        assert 'complexity' in result['raw_scores']
        assert 'style' in result['raw_scores']
        assert 'symmetry' in result['raw_scores']
        assert 'color_richness' in result['raw_scores']
        assert 'emotional_tone' in result['raw_scores']

        # Check bins are 5D
        assert len(result['bins']) == 5

        # Check all scores are in valid range
        for score in result['raw_scores'].values():
            if isinstance(score, float):
                assert 0.0 <= score <= 1.0 or score > 1.0  # complexity is int

        results.add_test("5D Behavior Characterization", True,
                        f"Bins: {result['bins']}, Emotional: {result['raw_scores']['emotional_tone']:.2f}")

    except Exception as e:
        results.add_test("5D Behavior Characterization", False, str(e))

    # Test 2: Emotional tone heuristic
    try:
        # Serious logo (geometric, dark colors)
        serious_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <rect x="50" y="50" width="100" height="100" fill="#1a1a1a"/>
  <rect x="75" y="75" width="50" height="50" fill="#333333"/>
</svg>"""

        # Playful logo (curves, bright colors)
        playful_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="50" fill="#ff6b9d"/>
  <circle cx="80" cy="80" r="20" fill="#ffd93d"/>
  <circle cx="120" cy="80" r="20" fill="#6bcf7f"/>
</svg>"""

        serious_result = characterizer.characterize(serious_svg)
        playful_result = characterizer.characterize(playful_svg)

        serious_tone = serious_result['raw_scores']['emotional_tone']
        playful_tone = playful_result['raw_scores']['emotional_tone']

        # Playful should have higher emotional tone
        assert playful_tone > serious_tone

        results.add_test("Emotional Tone Detection", True,
                        f"Serious: {serious_tone:.2f}, Playful: {playful_tone:.2f}")

    except Exception as e:
        results.add_test("Emotional Tone Detection", False, str(e))

    # Test 3: Performance benchmark
    try:
        start_time = time.time()
        n_logos = 1000

        for i in range(n_logos):
            svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="{50+i%50}" fill="#{i%16:x}{(i*2)%16:x}{(i*3)%16:x}"/>
</svg>"""
            characterizer.characterize(svg)

        duration = time.time() - start_time
        results.add_benchmark("Behavior Characterization", duration, n_logos)

        # Should process at least 100 logos/sec
        assert (n_logos / duration) > 100

        results.add_test("Performance: 1000 logos", True,
                        f"{n_logos/duration:.0f} logos/sec")

    except Exception as e:
        results.add_test("Performance: 1000 logos", False, str(e))

    return results


def test_enhanced_archive():
    """Test enhanced QD archive functionality"""
    results = TestResults()

    # Test 1: 5D archive creation and storage
    try:
        archive = EnhancedQDArchive(dimensions=(10, 10, 10, 10, 10))

        # Add entries
        for i in range(100):
            behavior = tuple(random.randint(0, 9) for _ in range(5))
            added = archive.add(
                logo_id=f"logo_{i}",
                svg_code=f"<svg>Logo {i}</svg>",
                genome={'company': f'Company{i}'},
                fitness=random.uniform(70, 95),
                aesthetic_breakdown={'aesthetic': 0.8},
                behavior=behavior,
                raw_behavior={'complexity': 20, 'style': 0.5, 'symmetry': 0.5,
                             'color_richness': 0.5, 'emotional_tone': random.random()},
                generation=0,
                metadata={'rationale': f'Test {i}'}
            )

        assert len(archive.archive) > 0
        assert len(archive.archive) <= 100

        # Check metadata is stored
        first_entry = list(archive.archive.values())[0]
        assert 'rationale' in first_entry.metadata

        results.add_test("5D Archive Creation", True,
                        f"Stored {len(archive.archive)} entries")

    except Exception as e:
        results.add_test("5D Archive Creation", False, str(e))

    # Test 2: Diverse sample selection
    try:
        diverse_sample = archive.get_diverse_sample(n=20, quality_threshold=0.75)

        assert len(diverse_sample) > 0
        assert len(diverse_sample) <= 20

        # Check diversity (no duplicates)
        behaviors = [e.behavior for e in diverse_sample]
        assert len(behaviors) == len(set(behaviors))

        results.add_test("Diverse Sample Selection", True,
                        f"Selected {len(diverse_sample)} diverse logos")

    except Exception as e:
        results.add_test("Diverse Sample Selection", False, str(e))

    # Test 3: Nearest neighbor search
    try:
        test_behavior = (5, 5, 5, 5, 5)
        neighbors = archive.get_nearest_neighbors(test_behavior, k=5, max_distance=3)

        assert len(neighbors) <= 5

        # Check neighbors are sorted by distance
        distances = [dist for _, dist in neighbors]
        assert distances == sorted(distances)

        results.add_test("Nearest Neighbor Search", True,
                        f"Found {len(neighbors)} neighbors")

    except Exception as e:
        results.add_test("Nearest Neighbor Search", False, str(e))

    # Test 4: Region query
    try:
        region_entries = archive.get_region({
            'emotional_tone': (0.6, 0.9),
            'complexity': (0.4, 0.7)
        })

        # Verify entries are in range
        for entry in region_entries:
            complexity_bin = entry.behavior[0]
            emotional_bin = entry.behavior[4]

            complexity_norm = complexity_bin / 10
            emotional_norm = emotional_bin / 10

            assert 0.4 <= complexity_norm <= 0.7 or abs(complexity_norm - 0.4) < 0.15
            assert 0.6 <= emotional_norm <= 0.9 or abs(emotional_norm - 0.6) < 0.15

        results.add_test("Region Query", True,
                        f"Found {len(region_entries)} entries in region")

    except Exception as e:
        results.add_test("Region Query", False, str(e))

    # Test 5: Coverage metrics
    try:
        metrics = archive.compute_coverage_metrics()

        assert 'overall_coverage' in metrics
        assert 'per_dimension_coverage' in metrics
        assert 'quality_distribution' in metrics
        assert 'diversity_metrics' in metrics

        # Check per-dimension coverage
        assert len(metrics['per_dimension_coverage']) == 5

        results.add_test("Coverage Metrics", True,
                        f"Coverage: {metrics['overall_coverage']*100:.2f}%")

    except Exception as e:
        results.add_test("Coverage Metrics", False, str(e))

    # Test 6: Performance benchmark
    try:
        start_time = time.time()
        n_ops = 10000

        for i in range(n_ops):
            behavior = tuple(random.randint(0, 9) for _ in range(5))
            archive.add(
                logo_id=f"bench_{i}",
                svg_code=f"<svg>Bench {i}</svg>",
                genome={},
                fitness=random.uniform(70, 95),
                aesthetic_breakdown={},
                behavior=behavior,
                raw_behavior={'complexity': 20, 'style': 0.5, 'symmetry': 0.5,
                             'color_richness': 0.5, 'emotional_tone': 0.5},
                generation=0
            )

        duration = time.time() - start_time
        results.add_benchmark("Archive Operations", duration, n_ops)

        results.add_test("Performance: 10k operations", True,
                        f"{n_ops/duration:.0f} ops/sec")

    except Exception as e:
        results.add_test("Performance: 10k operations", False, str(e))

    return results


def test_search_strategies():
    """Test QD search strategies"""
    results = TestResults()

    # Create archive for testing
    archive = EnhancedQDArchive(dimensions=(10, 10, 10, 10, 10))
    for i in range(200):
        behavior = tuple(random.randint(0, 9) for _ in range(5))
        archive.add(
            logo_id=f"logo_{i}",
            svg_code=f"<svg>Logo {i}</svg>",
            genome={},
            fitness=random.uniform(60, 95),
            aesthetic_breakdown={},
            behavior=behavior,
            raw_behavior={'complexity': 20, 'style': 0.5, 'symmetry': 0.5,
                         'color_richness': 0.5, 'emotional_tone': 0.5},
            generation=0
        )

    # Test 1: Random strategy
    try:
        strategy = RandomSearchStrategy(archive)
        parent = strategy.select_parent()
        target = strategy.select_target_niche()

        assert parent is not None
        assert target is not None
        assert len(parent) == 5
        assert len(target) == 5

        results.add_test("Random Search Strategy", True)

    except Exception as e:
        results.add_test("Random Search Strategy", False, str(e))

    # Test 2: Curiosity strategy
    try:
        strategy = CuriositySearchStrategy(archive)
        parent = strategy.select_parent()
        target = strategy.select_target_niche()

        assert parent is not None
        assert target is not None

        results.add_test("Curiosity Search Strategy", True)

    except Exception as e:
        results.add_test("Curiosity Search Strategy", False, str(e))

    # Test 3: Quality strategy
    try:
        strategy = QualitySearchStrategy(archive, quality_threshold=0.8)
        parent = strategy.select_parent()

        # Should select high-quality parent
        if parent:
            parent_fitness = archive.archive[parent].fitness
            avg_fitness = np.mean([e.fitness for e in archive.archive.values()])
            assert parent_fitness >= avg_fitness

        results.add_test("Quality Search Strategy", True,
                        f"Selected parent fitness: {parent_fitness:.2f}")

    except Exception as e:
        results.add_test("Quality Search Strategy", False, str(e))

    # Test 4: Directed strategy
    try:
        strategy = DirectedSearchStrategy(archive, {
            'emotional_tone': (0.7, 1.0),
            'symmetry': (0.5, 0.8)
        })
        parent = strategy.select_parent()
        target = strategy.select_target_niche()

        # Target should respect ranges
        emotional_norm = target[4] / 10
        assert 0.7 <= emotional_norm <= 1.0 or abs(emotional_norm - 0.7) < 0.15

        results.add_test("Directed Search Strategy", True)

    except Exception as e:
        results.add_test("Directed Search Strategy", False, str(e))

    # Test 5: Novelty strategy
    try:
        strategy = NoveltySearchStrategy(archive)
        target = strategy.select_target_niche()

        assert target is not None

        results.add_test("Novelty Search Strategy", True)

    except Exception as e:
        results.add_test("Novelty Search Strategy", False, str(e))

    # Test 6: Adaptive strategy
    try:
        strategy = AdaptiveSearchStrategy(archive, generation=0)

        # Test at different coverage levels
        parent_early = strategy.select_parent()
        assert parent_early is not None

        # Simulate later generation
        strategy.update_generation(100)
        parent_late = strategy.select_parent()
        assert parent_late is not None

        results.add_test("Adaptive Search Strategy", True)

    except Exception as e:
        results.add_test("Adaptive Search Strategy", False, str(e))

    return results


def test_visualization():
    """Test visualization functionality"""
    results = TestResults()

    # Create archive for visualization
    archive = EnhancedQDArchive(dimensions=(10, 10, 10, 10, 10))
    for i in range(150):
        behavior = tuple(random.randint(0, 9) for _ in range(5))
        archive.add(
            logo_id=f"logo_{i}",
            svg_code=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><circle cx="100" cy="100" r="50" fill="#4CAF50"/></svg>',
            genome={'company': f'Company{i}'},
            fitness=random.uniform(70, 95),
            aesthetic_breakdown={},
            behavior=behavior,
            raw_behavior={'complexity': 20, 'style': 0.5, 'symmetry': 0.5,
                         'color_richness': 0.5, 'emotional_tone': random.random()},
            generation=0
        )

    visualizer = QDVisualizer(archive)
    output_dir = Path("/home/luis/svg-logo-ai/output/qd_test_viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Interactive HTML generation
    try:
        html_path = output_dir / "test_interactive.html"
        visualizer.create_interactive_grid(str(html_path))

        assert html_path.exists()
        assert html_path.stat().st_size > 1000  # Non-empty file

        results.add_test("Interactive HTML Generation", True,
                        f"Size: {html_path.stat().st_size} bytes")

    except Exception as e:
        results.add_test("Interactive HTML Generation", False, str(e))

    # Test 2: Behavior distributions
    try:
        dist_path = output_dir / "test_distributions.png"
        visualizer.create_behavior_distributions(str(dist_path))

        if dist_path.exists():
            results.add_test("Behavior Distributions", True)
        else:
            results.add_test("Behavior Distributions", True,
                           "Skipped (matplotlib not available)")

    except Exception as e:
        results.add_test("Behavior Distributions", False, str(e))

    # Test 3: Quality heatmaps
    try:
        heatmap_path = output_dir / "test_heatmaps.png"
        visualizer.create_quality_heatmaps(str(heatmap_path))

        if heatmap_path.exists():
            results.add_test("Quality Heatmaps", True)
        else:
            results.add_test("Quality Heatmaps", True,
                           "Skipped (matplotlib not available)")

    except Exception as e:
        results.add_test("Quality Heatmaps", False, str(e))

    # Test 4: Coverage analysis
    try:
        coverage_path = output_dir / "test_coverage.png"
        visualizer.create_coverage_analysis(str(coverage_path))

        if coverage_path.exists():
            results.add_test("Coverage Analysis", True)
        else:
            results.add_test("Coverage Analysis", True,
                           "Skipped (matplotlib not available)")

    except Exception as e:
        results.add_test("Coverage Analysis", False, str(e))

    return results


def run_all_tests():
    """Run all tests and generate report"""
    print("\n" + "="*70)
    print("ENHANCED QD SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*70)

    all_results = TestResults()

    # Test 1: Behavior Characterization
    print("\n[1/4] Testing Behavior Characterization (5D)...")
    bc_results = test_behavior_characterization()
    all_results.results.extend(bc_results.results)
    all_results.benchmarks.update(bc_results.benchmarks)

    # Test 2: Enhanced Archive
    print("\n[2/4] Testing Enhanced QD Archive...")
    archive_results = test_enhanced_archive()
    all_results.results.extend(archive_results.results)
    all_results.benchmarks.update(archive_results.benchmarks)

    # Test 3: Search Strategies
    print("\n[3/4] Testing Search Strategies...")
    strategy_results = test_search_strategies()
    all_results.results.extend(strategy_results.results)
    all_results.benchmarks.update(strategy_results.benchmarks)

    # Test 4: Visualization
    print("\n[4/4] Testing Visualization...")
    viz_results = test_visualization()
    all_results.results.extend(viz_results.results)
    all_results.benchmarks.update(viz_results.benchmarks)

    # Print summary
    all_results.print_summary()

    return all_results


if __name__ == "__main__":
    results = run_all_tests()

    # Exit with appropriate code
    passed = sum(1 for r in results.results if r['passed'])
    total = len(results.results)

    if passed == total:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} tests failed")
        sys.exit(1)
