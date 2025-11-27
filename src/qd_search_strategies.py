"""
QD Search Strategies for MAP-Elites Logo Generation
====================================================
Different search strategies for Quality-Diversity optimization.

Strategies:
1. RandomSearchStrategy: Uniform random selection from archive
2. CuriositySearchStrategy: Prioritize under-explored regions
3. QualitySearchStrategy: Focus on high-quality regions for local refinement
4. DirectedSearchStrategy: Target specific behavioral regions based on user query
5. NoveltySearchStrategy: Prioritize regions far from current solutions
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Dict
from abc import ABC, abstractmethod
from collections import Counter


class QDSearchStrategy(ABC):
    """Base class for QD search strategies"""

    def __init__(self, archive):
        """
        Initialize search strategy

        Args:
            archive: EnhancedQDArchive instance
        """
        self.archive = archive

    @abstractmethod
    def select_parent(self) -> Optional[Tuple]:
        """
        Select a parent from the archive for mutation/crossover

        Returns:
            Behavior coordinates of selected parent, or None if archive is empty
        """
        pass

    @abstractmethod
    def select_target_niche(self) -> Optional[Tuple]:
        """
        Select a target behavioral niche to explore

        Returns:
            Behavior coordinates of target niche, or None if unable to select
        """
        pass


class RandomSearchStrategy(QDSearchStrategy):
    """Uniform random selection from archive"""

    def select_parent(self) -> Optional[Tuple]:
        """
        Select random occupied cell from archive

        Returns:
            Behavior coordinates of random parent
        """
        if not self.archive.archive:
            return None

        return random.choice(list(self.archive.archive.keys()))

    def select_target_niche(self) -> Optional[Tuple]:
        """
        Select random niche (occupied or empty)

        Returns:
            Random behavior coordinates
        """
        return tuple(
            random.randint(0, dim - 1)
            for dim in self.archive.dimensions
        )


class CuriositySearchStrategy(QDSearchStrategy):
    """
    Prioritize under-explored regions

    Favors:
    - Empty cells
    - Cells in sparse regions
    - Dimensions with lower coverage
    """

    def __init__(self, archive, exploration_weight: float = 0.7):
        """
        Initialize curiosity-driven search

        Args:
            archive: EnhancedQDArchive instance
            exploration_weight: Weight for exploration vs exploitation (0-1)
        """
        super().__init__(archive)
        self.exploration_weight = exploration_weight

    def select_parent(self) -> Optional[Tuple]:
        """
        Select parent biased toward sparse regions

        More likely to select from regions with fewer neighbors
        """
        if not self.archive.archive:
            return None

        # Calculate sparsity for each occupied cell
        candidates = list(self.archive.archive.keys())
        sparsities = []

        for behavior in candidates:
            # Count occupied neighbors
            neighbors = self.archive.get_neighbors(behavior, distance=2)
            occupied_neighbors = sum(1 for n in neighbors if n in self.archive.archive)
            sparsity = 1.0 / (1 + occupied_neighbors)  # Higher = more sparse
            sparsities.append(sparsity)

        # Sample based on sparsity
        if random.random() < self.exploration_weight:
            # Weighted by sparsity
            total_sparsity = sum(sparsities)
            if total_sparsity > 0:
                probabilities = [s / total_sparsity for s in sparsities]
                return random.choices(candidates, weights=probabilities, k=1)[0]

        # Fallback to random
        return random.choice(candidates)

    def select_target_niche(self) -> Optional[Tuple]:
        """
        Select target niche in under-explored region

        Prioritizes:
        1. Empty cells near occupied cells
        2. Dimensions with low coverage
        """
        # Try to find empty cell near occupied cells
        if self.archive.archive and random.random() < 0.8:
            # Pick random occupied cell
            occupied = random.choice(list(self.archive.archive.keys()))

            # Find empty neighbors
            for distance in range(1, 4):
                empty_neighbors = self.archive.get_empty_neighbors(occupied, distance)
                if empty_neighbors:
                    return random.choice(empty_neighbors)

        # Fallback: target dimension with lowest coverage
        coverage_metrics = self.archive.compute_coverage_metrics()
        per_dim_coverage = coverage_metrics.get('per_dimension_coverage', {})

        if per_dim_coverage:
            dimension_names = ['complexity', 'style', 'symmetry', 'color_richness', 'emotional_tone']

            # Find dimension with lowest coverage
            lowest_coverage_dim = min(
                range(len(self.archive.dimensions)),
                key=lambda i: per_dim_coverage.get(
                    dimension_names[i] if i < len(dimension_names) else f'dim_{i}',
                    0.5
                )
            )

            # Generate target favoring low-coverage dimension
            target = list(
                random.randint(0, dim - 1)
                for dim in self.archive.dimensions
            )

            # Bias the low-coverage dimension toward extremes
            if random.random() < 0.7:
                target[lowest_coverage_dim] = random.choice([0, self.archive.dimensions[lowest_coverage_dim] - 1])

            return tuple(target)

        # Ultimate fallback: random
        return tuple(
            random.randint(0, dim - 1)
            for dim in self.archive.dimensions
        )


class QualitySearchStrategy(QDSearchStrategy):
    """
    Focus on high-quality regions for local refinement

    Selects parents from high-fitness cells and targets nearby niches
    """

    def __init__(self, archive, quality_threshold: float = 0.75):
        """
        Initialize quality-focused search

        Args:
            archive: EnhancedQDArchive instance
            quality_threshold: Minimum fitness percentile (0-1)
        """
        super().__init__(archive)
        self.quality_threshold = quality_threshold

    def select_parent(self) -> Optional[Tuple]:
        """
        Select parent from high-quality cells

        Returns:
            Behavior of high-quality parent
        """
        if not self.archive.archive:
            return None

        # Get fitness threshold
        fitnesses = [entry.fitness for entry in self.archive.archive.values()]
        threshold = np.percentile(fitnesses, self.quality_threshold * 100)

        # Filter high-quality entries
        high_quality = [
            behavior for behavior, entry in self.archive.archive.items()
            if entry.fitness >= threshold
        ]

        if high_quality:
            return random.choice(high_quality)

        # Fallback: best entry
        best_entry = max(self.archive.archive.values(), key=lambda e: e.fitness)
        return best_entry.behavior

    def select_target_niche(self) -> Optional[Tuple]:
        """
        Select target near high-quality regions

        Explores locally around best solutions
        """
        # Select high-quality parent
        parent = self.select_parent()
        if parent is None:
            return None

        # Target nearby empty cell
        for distance in range(1, 4):
            empty_neighbors = self.archive.get_empty_neighbors(parent, distance)
            if empty_neighbors:
                return random.choice(empty_neighbors)

        # Fallback: near parent
        return tuple(
            max(0, min(dim - 1, parent[i] + random.randint(-2, 2)))
            for i, dim in enumerate(self.archive.dimensions)
        )


class DirectedSearchStrategy(QDSearchStrategy):
    """
    Target specific behavioral regions based on user query or requirements

    Useful for LLM-guided search where specific characteristics are desired
    """

    def __init__(self, archive, target_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize directed search

        Args:
            archive: EnhancedQDArchive instance
            target_ranges: Dict mapping dimension names to (min, max) target ranges
                          e.g., {'emotional_tone': (0.7, 1.0), 'symmetry': (0.6, 0.9)}
        """
        super().__init__(archive)
        self.target_ranges = target_ranges or {}

    def set_target_ranges(self, target_ranges: Dict[str, Tuple[float, float]]):
        """Update target ranges"""
        self.target_ranges = target_ranges

    def select_parent(self) -> Optional[Tuple]:
        """
        Select parent from target region or nearby

        Returns:
            Behavior of parent in or near target region
        """
        if not self.archive.archive:
            return None

        if not self.target_ranges:
            # No target specified, random selection
            return random.choice(list(self.archive.archive.keys()))

        # Get entries in target region
        region_entries = self.archive.get_region(self.target_ranges)

        if region_entries:
            # Select from region
            return random.choice([e.behavior for e in region_entries])

        # No entries in target, select closest
        dimension_names = ['complexity', 'style', 'symmetry', 'color_richness', 'emotional_tone']

        def distance_to_target(behavior):
            """Calculate distance from behavior to target range center"""
            dist = 0
            for dim_name, (min_val, max_val) in self.target_ranges.items():
                if dim_name in dimension_names:
                    dim_idx = dimension_names.index(dim_name)
                    if dim_idx < len(behavior):
                        normalized = behavior[dim_idx] / self.archive.dimensions[dim_idx]
                        target_center = (min_val + max_val) / 2
                        dist += abs(normalized - target_center)
            return dist

        # Find closest entry
        closest = min(
            self.archive.archive.keys(),
            key=distance_to_target
        )
        return closest

    def select_target_niche(self) -> Optional[Tuple]:
        """
        Select target niche within specified ranges

        Returns:
            Behavior coordinates in target region
        """
        if not self.target_ranges:
            # No target, random selection
            return tuple(
                random.randint(0, dim - 1)
                for dim in self.archive.dimensions
            )

        # Generate target within ranges
        dimension_names = ['complexity', 'style', 'symmetry', 'color_richness', 'emotional_tone']
        target = []

        for dim_idx, dim_size in enumerate(self.archive.dimensions):
            dim_name = dimension_names[dim_idx] if dim_idx < len(dimension_names) else f'dim_{dim_idx}'

            if dim_name in self.target_ranges:
                # Sample within target range
                min_val, max_val = self.target_ranges[dim_name]
                # Convert to bin indices
                min_bin = int(min_val * dim_size)
                max_bin = int(max_val * dim_size)
                bin_val = random.randint(
                    max(0, min_bin),
                    min(dim_size - 1, max_bin)
                )
                target.append(bin_val)
            else:
                # No constraint, random
                target.append(random.randint(0, dim_size - 1))

        return tuple(target)


class NoveltySearchStrategy(QDSearchStrategy):
    """
    Prioritize regions far from current solutions

    Maximizes behavioral novelty
    """

    def __init__(self, archive, novelty_weight: float = 0.8):
        """
        Initialize novelty search

        Args:
            archive: EnhancedQDArchive instance
            novelty_weight: Weight for novelty vs quality (0-1)
        """
        super().__init__(archive)
        self.novelty_weight = novelty_weight

    def select_parent(self) -> Optional[Tuple]:
        """
        Select parent randomly (novelty search doesn't favor specific parents)

        Returns:
            Random parent behavior
        """
        if not self.archive.archive:
            return None

        return random.choice(list(self.archive.archive.keys()))

    def select_target_niche(self) -> Optional[Tuple]:
        """
        Select target niche far from existing solutions

        Returns:
            Behavior coordinates maximizing distance to archive
        """
        if not self.archive.archive:
            # Empty archive, random selection
            return tuple(
                random.randint(0, dim - 1)
                for dim in self.archive.dimensions
            )

        # Sample multiple candidates and pick most novel
        num_candidates = min(50, int(np.prod(self.archive.dimensions) * 0.01))
        candidates = []

        for _ in range(num_candidates):
            candidate = tuple(
                random.randint(0, dim - 1)
                for dim in self.archive.dimensions
            )

            # Skip if occupied
            if candidate in self.archive.archive:
                continue

            # Calculate novelty (average distance to nearest neighbors)
            neighbors = self.archive.get_nearest_neighbors(candidate, k=5, max_distance=10)

            if neighbors:
                avg_distance = np.mean([dist for _, dist in neighbors])
            else:
                avg_distance = float('inf')  # Very novel (no neighbors)

            candidates.append((candidate, avg_distance))

        if candidates:
            # Select most novel (highest avg distance)
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        # Fallback: random
        return tuple(
            random.randint(0, dim - 1)
            for dim in self.archive.dimensions
        )


class AdaptiveSearchStrategy(QDSearchStrategy):
    """
    Adaptive strategy that switches between different strategies based on progress

    Starts with curiosity (exploration), transitions to quality (exploitation)
    """

    def __init__(self, archive, generation: int = 0):
        """
        Initialize adaptive search

        Args:
            archive: EnhancedQDArchive instance
            generation: Current generation number
        """
        super().__init__(archive)
        self.generation = generation
        self.curiosity_strategy = CuriositySearchStrategy(archive)
        self.quality_strategy = QualitySearchStrategy(archive)
        self.random_strategy = RandomSearchStrategy(archive)

    def update_generation(self, generation: int):
        """Update generation count for adaptation"""
        self.generation = generation

    def select_parent(self) -> Optional[Tuple]:
        """
        Select parent using adaptive strategy

        Early: Curiosity (exploration)
        Late: Quality (exploitation)
        """
        coverage = self.archive.get_coverage()

        # Low coverage: explore
        if coverage < 0.3:
            return self.curiosity_strategy.select_parent()
        # Medium coverage: balance
        elif coverage < 0.6:
            if random.random() < 0.5:
                return self.curiosity_strategy.select_parent()
            else:
                return self.quality_strategy.select_parent()
        # High coverage: exploit
        else:
            return self.quality_strategy.select_parent()

    def select_target_niche(self) -> Optional[Tuple]:
        """
        Select target niche using adaptive strategy
        """
        coverage = self.archive.get_coverage()

        # Adapt based on coverage
        if coverage < 0.3:
            return self.curiosity_strategy.select_target_niche()
        elif coverage < 0.6:
            if random.random() < 0.6:
                return self.curiosity_strategy.select_target_niche()
            else:
                return self.random_strategy.select_target_niche()
        else:
            if random.random() < 0.7:
                return self.quality_strategy.select_target_niche()
            else:
                return self.curiosity_strategy.select_target_niche()


def demo():
    """Demo QD search strategies"""
    from map_elites_archive import EnhancedQDArchive

    print("="*60)
    print("QD Search Strategies Demo")
    print("="*60)

    # Create archive and populate with some entries
    archive = EnhancedQDArchive(dimensions=(10, 10, 10, 10, 10))

    # Add dummy data
    for i in range(100):
        behavior = tuple(random.randint(0, 9) for _ in range(5))
        archive.add(
            logo_id=f"logo_{i}",
            svg_code=f"<svg>Logo {i}</svg>",
            genome={'test': i},
            fitness=random.uniform(60, 95),
            aesthetic_breakdown={},
            behavior=behavior,
            raw_behavior={'complexity': 20, 'style': 0.5, 'symmetry': 0.5,
                         'color_richness': 0.5, 'emotional_tone': 0.5},
            generation=0
        )

    print(f"Archive populated with {len(archive.archive)} entries")
    print(f"Coverage: {archive.get_coverage()*100:.2f}%\n")

    # Test each strategy
    strategies = {
        'Random': RandomSearchStrategy(archive),
        'Curiosity': CuriositySearchStrategy(archive),
        'Quality': QualitySearchStrategy(archive),
        'Directed': DirectedSearchStrategy(archive, {'emotional_tone': (0.7, 1.0)}),
        'Novelty': NoveltySearchStrategy(archive),
        'Adaptive': AdaptiveSearchStrategy(archive)
    }

    for name, strategy in strategies.items():
        print(f"\n{name} Strategy:")
        parent = strategy.select_parent()
        target = strategy.select_target_niche()
        print(f"  Selected parent: {parent}")
        print(f"  Target niche: {target}")

        if parent and parent in archive.archive:
            print(f"  Parent fitness: {archive.archive[parent].fitness:.2f}")


if __name__ == "__main__":
    demo()
