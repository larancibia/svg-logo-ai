"""
MAP-Elites Archive for Logo Generation (Enhanced for LLM-QD)
=============================================================
Maintains a multi-dimensional grid of diverse, high-quality logos.

Each cell in the grid represents a unique combination of behavioral characteristics.
Only the best logo (highest fitness) is kept per cell.

Enhanced Features:
- Support for 5D behavior space (100,000 cells)
- Rich metadata storage (design rationale, generation history)
- Advanced search strategies (diversity, quality, directed)
- Efficient spatial indexing for neighbor queries
- Export formats for interactive visualization
- Comprehensive coverage analytics
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Callable
from collections import defaultdict
import chromadb
from chromadb.config import Settings
from dataclasses import dataclass, asdict, field


@dataclass
class ArchiveEntry:
    """Single entry in the MAP-Elites archive (enhanced with metadata)"""
    logo_id: str
    svg_code: str
    genome: Dict
    fitness: float
    aesthetic_breakdown: Dict
    behavior: Tuple  # Now supports 5D: (complexity, style, symmetry, color, emotional)
    raw_behavior: Dict  # Raw behavior scores (all 5 dimensions)
    generation: int
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)  # Rich metadata (rationale, etc.)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'logo_id': self.logo_id,
            'genome': self.genome,
            'fitness': self.fitness,
            'aesthetic_breakdown': self.aesthetic_breakdown,
            'behavior': list(self.behavior),
            'raw_behavior': self.raw_behavior,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'metadata': self.metadata
        }


class EnhancedQDArchive:
    """
    Enhanced MAP-Elites archive for LLM-guided QD

    Structure: N-dimensional grid where each cell contains the best logo
    for that behavioral niche.

    Now supports:
    - 5D behavior space (100,000 cells)
    - Rich metadata storage
    - Advanced search strategies
    - Spatial indexing for fast neighbor queries
    """

    def __init__(self,
                 dimensions: Tuple[int, ...] = (10, 10, 10, 10, 10),
                 chroma_db_path: str = "/home/luis/svg-logo-ai/chroma_db/map_elites_5d"):
        """
        Initialize Enhanced QD Archive

        Args:
            dimensions: Tuple of grid dimensions (default: (10,10,10,10,10) = 100k cells)
            chroma_db_path: Path to ChromaDB storage
        """
        self.dimensions = dimensions
        self.archive = {}  # Key: behavior tuple, Value: ArchiveEntry
        self.chroma_db_path = Path(chroma_db_path)
        self.spatial_index = defaultdict(set)  # For fast region queries

        # Initialize ChromaDB for persistent storage
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Create collection for archive
        try:
            self.collection = self.client.get_collection("map_elites_archive")
        except:
            self.collection = self.client.create_collection(
                name="map_elites_archive",
                metadata={"description": "MAP-Elites archive of diverse logos"}
            )

    def add(self,
            logo_id: str,
            svg_code: str,
            genome: Dict,
            fitness: float,
            aesthetic_breakdown: Dict,
            behavior: Tuple,
            raw_behavior: Dict,
            generation: int,
            parent_ids: List[str] = None,
            metadata: Dict = None) -> bool:
        """
        Add logo to archive (only if it's better than existing entry in that cell)

        Args:
            logo_id: Unique identifier
            svg_code: SVG code
            genome: Genome dictionary
            fitness: Fitness score
            aesthetic_breakdown: Breakdown of aesthetic metrics
            behavior: Binned behavior coordinates (supports 5D)
            raw_behavior: Raw behavior scores (all dimensions)
            generation: Generation number
            parent_ids: Parent logo IDs
            metadata: Optional metadata dict (design rationale, query context, etc.)

        Returns:
            True if added (cell was empty or new logo is better), False otherwise
        """
        # Validate behavior dimensions
        if len(behavior) != len(self.dimensions):
            raise ValueError(f"Behavior dimensions {len(behavior)} don't match archive {len(self.dimensions)}")

        # Check if any bin is out of range
        for i, (bin_val, dim_size) in enumerate(zip(behavior, self.dimensions)):
            if not (0 <= bin_val < dim_size):
                print(f"Warning: Behavior bin {i} value {bin_val} out of range [0, {dim_size})")
                return False

        # Check if cell is empty or if new logo is better
        if behavior not in self.archive or fitness > self.archive[behavior].fitness:
            # Create entry
            entry = ArchiveEntry(
                logo_id=logo_id,
                svg_code=svg_code,
                genome=genome,
                fitness=fitness,
                aesthetic_breakdown=aesthetic_breakdown,
                behavior=behavior,
                raw_behavior=raw_behavior,
                generation=generation,
                parent_ids=parent_ids or [],
                metadata=metadata or {}
            )

            # Update archive
            self.archive[behavior] = entry

            # Update spatial index for fast region queries
            self._update_spatial_index(behavior)

            # Store in ChromaDB
            self._store_in_chromadb(entry)

            return True

        return False

    def add_with_metadata(self, logo: Dict, behavior: Tuple, fitness: float, metadata: Dict) -> bool:
        """
        Convenience method to add logo with rich metadata

        Args:
            logo: Dict with logo_id, svg_code, genome, aesthetic_breakdown
            behavior: Binned behavior coordinates
            fitness: Fitness score
            metadata: Rich metadata dict with:
                - design_rationale: LLM explanation
                - generation_history: List of mutations/operations
                - parent_info: Info about parents
                - user_query: Original query context
                - timestamp: When created

        Returns:
            True if added, False otherwise
        """
        return self.add(
            logo_id=logo['logo_id'],
            svg_code=logo['svg_code'],
            genome=logo['genome'],
            fitness=fitness,
            aesthetic_breakdown=logo.get('aesthetic_breakdown', {}),
            behavior=behavior,
            raw_behavior=logo.get('raw_behavior', {}),
            generation=logo.get('generation', 0),
            parent_ids=logo.get('parent_ids', []),
            metadata=metadata
        )

    def get(self, behavior: Tuple[int, int, int, int]) -> Optional[ArchiveEntry]:
        """
        Get logo at specific behavior coordinates

        Args:
            behavior: Behavior bin tuple

        Returns:
            ArchiveEntry if cell is occupied, None otherwise
        """
        return self.archive.get(behavior)

    def get_random_occupied(self) -> Optional[Tuple[Tuple[int, int, int, int], ArchiveEntry]]:
        """
        Get random occupied cell from archive

        Returns:
            (behavior, entry) tuple, or None if archive is empty
        """
        if not self.archive:
            return None

        behavior = random.choice(list(self.archive.keys()))
        return behavior, self.archive[behavior]

    def get_neighbors(self, behavior: Tuple[int, int, int, int],
                      distance: int = 1) -> List[Tuple[int, int, int, int]]:
        """
        Get neighboring cells (Manhattan distance)

        Args:
            behavior: Center behavior
            distance: Manhattan distance (default: 1 for immediate neighbors)

        Returns:
            List of neighboring behavior coordinates
        """
        neighbors = []

        # Generate all neighbors within Manhattan distance
        for d0 in range(-distance, distance + 1):
            for d1 in range(-distance, distance + 1):
                for d2 in range(-distance, distance + 1):
                    for d3 in range(-distance, distance + 1):
                        if abs(d0) + abs(d1) + abs(d2) + abs(d3) <= distance:
                            neighbor = (
                                behavior[0] + d0,
                                behavior[1] + d1,
                                behavior[2] + d2,
                                behavior[3] + d3
                            )

                            # Check bounds
                            if all(0 <= neighbor[i] < self.dimensions[i] for i in range(len(self.dimensions))):
                                if neighbor != behavior:  # Exclude self
                                    neighbors.append(neighbor)

        return neighbors

    def get_empty_neighbors(self, behavior: Tuple[int, int, int, int],
                           distance: int = 1) -> List[Tuple[int, int, int, int]]:
        """Get unoccupied neighboring cells"""
        neighbors = self.get_neighbors(behavior, distance)
        return [n for n in neighbors if n not in self.archive]

    def get_coverage(self) -> float:
        """
        Calculate archive coverage (percentage of cells filled)

        Returns:
            Float between 0.0 and 1.0
        """
        total_cells = 1
        for dim in self.dimensions:
            total_cells *= dim

        return len(self.archive) / total_cells

    def get_statistics(self) -> Dict:
        """
        Get archive statistics

        Returns:
            Dict with coverage, fitness stats, behavior distribution, etc.
        """
        if not self.archive:
            return {
                'coverage': 0.0,
                'num_occupied': 0,
                'total_cells': sum(self.dimensions),
                'avg_fitness': 0.0,
                'max_fitness': 0.0,
                'min_fitness': 0.0
            }

        fitnesses = [entry.fitness for entry in self.archive.values()]

        return {
            'coverage': self.get_coverage(),
            'num_occupied': len(self.archive),
            'total_cells': sum(self.dimensions),
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'max_fitness': max(fitnesses),
            'min_fitness': min(fitnesses),
            'generations_represented': len(set(e.generation for e in self.archive.values()))
        }

    def get_best_logos(self, n: int = 10) -> List[ArchiveEntry]:
        """
        Get top N logos by fitness

        Args:
            n: Number of logos to return

        Returns:
            List of ArchiveEntry sorted by fitness (descending)
        """
        sorted_entries = sorted(self.archive.values(), key=lambda e: e.fitness, reverse=True)
        return sorted_entries[:n]

    def get_diverse_sample(self, n: int = 100, quality_threshold: float = 0.7) -> List[ArchiveEntry]:
        """
        Get N diverse high-quality logos using greedy selection

        Maximizes diversity in behavior space while maintaining quality threshold

        Args:
            n: Number of logos to sample
            quality_threshold: Minimum fitness (0-1 normalized)

        Returns:
            List of diverse, high-quality ArchiveEntry objects
        """
        # Filter by quality
        quality_entries = [e for e in self.archive.values() if e.fitness >= quality_threshold * 100]

        if len(quality_entries) <= n:
            return quality_entries

        # Greedy diversity selection
        selected = []
        remaining = quality_entries.copy()

        # Start with highest quality
        best = max(remaining, key=lambda e: e.fitness)
        selected.append(best)
        remaining.remove(best)

        # Iteratively select most diverse
        while len(selected) < n and remaining:
            max_min_dist = -1
            best_candidate = None

            for candidate in remaining:
                # Calculate minimum distance to selected set
                min_dist = min(self._behavior_distance(candidate.behavior, s.behavior)
                             for s in selected)

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break

        return selected

    def get_nearest_neighbors(self, behavior: Tuple, k: int = 5,
                              max_distance: int = 3) -> List[Tuple[ArchiveEntry, float]]:
        """
        Find k nearest neighbors in behavior space

        Args:
            behavior: Target behavior coordinates
            k: Number of neighbors to return
            max_distance: Maximum Manhattan distance to search

        Returns:
            List of (entry, distance) tuples sorted by distance
        """
        neighbors = []

        # Collect candidates within max_distance
        for distance in range(1, max_distance + 1):
            candidate_coords = self._get_neighbors_at_distance(behavior, distance)

            for coord in candidate_coords:
                if coord in self.archive:
                    dist = self._behavior_distance(behavior, coord)
                    neighbors.append((self.archive[coord], dist))

            # Early exit if we have enough
            if len(neighbors) >= k * 2:
                break

        # Sort by distance and return top k
        neighbors.sort(key=lambda x: x[1])
        return neighbors[:k]

    def get_region(self, behavior_ranges: Dict[str, Tuple[float, float]]) -> List[ArchiveEntry]:
        """
        Get all logos in a specific behavioral region

        Args:
            behavior_ranges: Dict mapping dimension names to (min, max) ranges
                           e.g., {'complexity': (0.5, 0.7), 'style': (0.3, 0.5)}
                           Dimensions: complexity, style, symmetry, color_richness, emotional_tone

        Returns:
            List of ArchiveEntry in the specified region
        """
        dimension_names = ['complexity', 'style', 'symmetry', 'color_richness', 'emotional_tone']
        results = []

        for entry in self.archive.values():
            # Check if entry is in all specified ranges
            in_region = True

            for dim_name, (min_val, max_val) in behavior_ranges.items():
                if dim_name not in dimension_names:
                    continue

                dim_idx = dimension_names.index(dim_name)
                if dim_idx < len(entry.behavior):
                    # Convert bin to normalized value
                    bin_val = entry.behavior[dim_idx]
                    normalized = bin_val / self.dimensions[dim_idx]

                    if not (min_val <= normalized <= max_val):
                        in_region = False
                        break

            if in_region:
                results.append(entry)

        return results

    def compute_coverage_metrics(self) -> Dict:
        """
        Detailed coverage analysis

        Returns:
            Dict with:
            - overall_coverage: Percentage of cells filled
            - per_dimension_coverage: Coverage per dimension
            - quality_distribution: Histogram of fitness scores
            - diversity_metrics: Various diversity measures
            - sparsity_map: Which regions are under-explored
        """
        if not self.archive:
            return {
                'overall_coverage': 0.0,
                'per_dimension_coverage': {},
                'quality_distribution': {},
                'diversity_metrics': {}
            }

        total_cells = np.prod(self.dimensions)
        occupied = len(self.archive)

        # Per-dimension coverage
        per_dim_coverage = {}
        dimension_names = ['complexity', 'style', 'symmetry', 'color_richness', 'emotional_tone']

        for dim_idx, dim_name in enumerate(dimension_names[:len(self.dimensions)]):
            unique_bins = set(entry.behavior[dim_idx] for entry in self.archive.values())
            coverage = len(unique_bins) / self.dimensions[dim_idx]
            per_dim_coverage[dim_name] = coverage

        # Quality distribution
        fitnesses = [e.fitness for e in self.archive.values()]
        quality_dist = {
            'min': min(fitnesses),
            'max': max(fitnesses),
            'mean': np.mean(fitnesses),
            'std': np.std(fitnesses),
            'median': np.median(fitnesses),
            'q25': np.percentile(fitnesses, 25),
            'q75': np.percentile(fitnesses, 75)
        }

        # Diversity metrics
        behaviors = [e.behavior for e in self.archive.values()]
        avg_distance = self._compute_average_distance(behaviors)

        diversity_metrics = {
            'average_behavior_distance': avg_distance,
            'unique_behaviors': len(set(behaviors)),
            'generation_diversity': len(set(e.generation for e in self.archive.values()))
        }

        return {
            'overall_coverage': occupied / total_cells,
            'occupied_cells': occupied,
            'total_cells': int(total_cells),
            'per_dimension_coverage': per_dim_coverage,
            'quality_distribution': quality_dist,
            'diversity_metrics': diversity_metrics
        }

    def export_for_visualization(self, output_path: str):
        """
        Export archive data for interactive visualization

        Creates JSON file with:
        - All archive entries
        - Coverage metrics
        - Metadata for interactive exploration

        Args:
            output_path: Path to save JSON file
        """
        export_data = {
            'dimensions': self.dimensions,
            'coverage_metrics': self.compute_coverage_metrics(),
            'entries': [entry.to_dict() for entry in self.archive.values()]
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Archive exported for visualization: {output_path}")
        print(f"  {len(self.archive)} entries")
        print(f"  Coverage: {export_data['coverage_metrics']['overall_coverage']*100:.2f}%")

    def _update_spatial_index(self, behavior: Tuple):
        """Update spatial index for fast region queries"""
        # Index by each dimension for faster filtering
        for dim_idx, bin_val in enumerate(behavior):
            key = (dim_idx, bin_val)
            self.spatial_index[key].add(behavior)

    def _behavior_distance(self, b1: Tuple, b2: Tuple) -> float:
        """Calculate Manhattan distance between two behaviors"""
        return sum(abs(a - b) for a, b in zip(b1, b2))

    def _get_neighbors_at_distance(self, behavior: Tuple, distance: int) -> List[Tuple]:
        """Get all cells at exactly Manhattan distance d from behavior"""
        neighbors = []

        def generate_offsets(remaining_dist, remaining_dims, current_offset):
            if remaining_dims == 0:
                if remaining_dist == 0:
                    neighbors.append(tuple(current_offset))
                return

            for offset in range(-remaining_dist, remaining_dist + 1):
                new_remaining = remaining_dist - abs(offset)
                generate_offsets(new_remaining, remaining_dims - 1, current_offset + [offset])

        generate_offsets(distance, len(self.dimensions), [])

        # Apply offsets and filter valid coordinates
        valid_neighbors = []
        for offset in neighbors:
            coord = tuple(behavior[i] + offset[i] for i in range(len(behavior)))
            if all(0 <= coord[i] < self.dimensions[i] for i in range(len(coord))):
                valid_neighbors.append(coord)

        return valid_neighbors

    def _compute_average_distance(self, behaviors: List[Tuple]) -> float:
        """Compute average pairwise distance between behaviors"""
        if len(behaviors) < 2:
            return 0.0

        total_dist = 0
        count = 0

        # Sample for efficiency (don't compute all pairs for large archives)
        sample_size = min(1000, len(behaviors))
        sampled = random.sample(behaviors, sample_size)

        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                total_dist += self._behavior_distance(sampled[i], sampled[j])
                count += 1

        return total_dist / count if count > 0 else 0.0

    def _store_in_chromadb(self, entry: ArchiveEntry):
        """Store entry in ChromaDB for retrieval and analysis (supports 5D)"""
        # Create searchable document
        genome = entry.genome

        # Build behavior string dynamically for any number of dimensions
        behavior_parts = []
        dim_names = ['Complexity', 'Style', 'Symmetry', 'Color', 'Emotional']
        for i, val in enumerate(entry.behavior):
            dim_name = dim_names[i] if i < len(dim_names) else f'Dim{i}'
            behavior_parts.append(f"{dim_name}={val}")

        behavior_str = ' '.join(behavior_parts)

        document = f"""
        Company: {genome.get('company', 'unknown')}
        Industry: {genome.get('industry', 'unknown')}
        Style: {' '.join(genome.get('style_keywords', []))}
        Colors: {' '.join(genome.get('color_palette', []))}
        Principles: {' '.join(genome.get('design_principles', []))}
        Behavior: {behavior_str}
        Metadata: {entry.metadata.get('design_rationale', '')}
        """

        # Metadata (dynamic for any number of dimensions)
        metadata = {
            'logo_id': entry.logo_id,
            'fitness': entry.fitness,
            'generation': entry.generation,
            'svg_code': entry.svg_code[:1000],  # Truncate for ChromaDB limit
            'genome_json': json.dumps(entry.genome)[:1000],
            'aesthetic_json': json.dumps(entry.aesthetic_breakdown)[:1000],
            'metadata_json': json.dumps(entry.metadata)[:1000]
        }

        # Add behavior dimensions
        for i, val in enumerate(entry.behavior):
            metadata[f'behavior_{i}'] = int(val)

        # Add raw behaviors
        for key, val in entry.raw_behavior.items():
            metadata[f'{key}_raw'] = float(val) if isinstance(val, (int, float)) else 0.0

        # Upsert (update or insert)
        try:
            self.collection.upsert(
                ids=[entry.logo_id],
                documents=[document],
                metadatas=[metadata]
            )
        except Exception as e:
            print(f"Warning: Failed to store in ChromaDB: {e}")

    def save_to_disk(self, output_dir: str):
        """
        Save archive to disk (JSON + SVG files)

        Args:
            output_dir: Directory to save archive
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        archive_data = []
        for behavior, entry in self.archive.items():
            archive_data.append(entry.to_dict())

            # Save SVG file
            svg_filename = f"{entry.logo_id}.svg"
            with open(output_path / svg_filename, 'w') as f:
                f.write(entry.svg_code)

        # Save archive index
        with open(output_path / "archive.json", 'w') as f:
            json.dump({
                'dimensions': self.dimensions,
                'num_occupied': len(self.archive),
                'coverage': self.get_coverage(),
                'statistics': self.get_statistics(),
                'entries': archive_data
            }, f, indent=2)

        print(f"Archive saved to {output_path}")
        print(f"  {len(self.archive)} logos across {len(self.dimensions)}D grid")
        print(f"  Coverage: {self.get_coverage()*100:.1f}%")

    def load_from_disk(self, archive_file: str):
        """
        Load archive from JSON file

        Args:
            archive_file: Path to archive.json
        """
        with open(archive_file, 'r') as f:
            data = json.load(f)

        archive_dir = Path(archive_file).parent

        # Load entries
        for entry_data in data['entries']:
            # Load SVG
            svg_file = archive_dir / f"{entry_data['logo_id']}.svg"
            if svg_file.exists():
                with open(svg_file, 'r') as f:
                    svg_code = f.read()

                # Recreate entry (with metadata support)
                behavior = tuple(entry_data['behavior'])
                entry = ArchiveEntry(
                    logo_id=entry_data['logo_id'],
                    svg_code=svg_code,
                    genome=entry_data['genome'],
                    fitness=entry_data['fitness'],
                    aesthetic_breakdown=entry_data['aesthetic_breakdown'],
                    behavior=behavior,
                    raw_behavior=entry_data['raw_behavior'],
                    generation=entry_data['generation'],
                    parent_ids=entry_data.get('parent_ids', []),
                    metadata=entry_data.get('metadata', {})
                )

                self.archive[behavior] = entry
                self._update_spatial_index(behavior)

        print(f"Loaded {len(self.archive)} logos from {archive_file}")


# Backward compatibility alias
MAPElitesArchive = EnhancedQDArchive


def demo():
    """Demo Enhanced QD Archive (5D)"""
    print("="*60)
    print("Enhanced QD Archive Demo (5D)")
    print("="*60)

    # Create small 5D archive for testing
    archive = EnhancedQDArchive(dimensions=(5, 5, 5, 5, 5))

    # Add some dummy entries
    for i in range(50):
        behavior = (
            random.randint(0, 4),
            random.randint(0, 4),
            random.randint(0, 4),
            random.randint(0, 4),
            random.randint(0, 4)
        )

        added = archive.add(
            logo_id=f"logo_{i}",
            svg_code=f"<svg>Logo {i}</svg>",
            genome={'test': i, 'company': f'Company{i}'},
            fitness=random.uniform(70, 95),
            aesthetic_breakdown={'aesthetic': random.uniform(70, 95)},
            behavior=behavior,
            raw_behavior={
                'complexity': 20,
                'style': 0.5,
                'symmetry': 0.5,
                'color_richness': 0.5,
                'emotional_tone': random.uniform(0, 1)
            },
            generation=0,
            metadata={'design_rationale': f'Test logo {i}'}
        )

        if added:
            print(f"Added logo_{i} to {behavior}")

    # Print statistics
    stats = archive.get_statistics()
    print("\n" + "="*60)
    print("Archive Statistics")
    print("="*60)
    print(f"Coverage: {stats['coverage']*100:.1f}%")
    print(f"Occupied cells: {stats['num_occupied']}")
    print(f"Avg fitness: {stats['avg_fitness']:.2f}")
    print(f"Max fitness: {stats['max_fitness']:.2f}")

    # Test neighbor finding
    print("\n" + "="*60)
    print("Neighbor Finding Test")
    print("="*60)
    test_behavior = (2, 2, 2, 2, 2)
    neighbors = archive.get_neighbors(test_behavior, distance=1)
    empty_neighbors = archive.get_empty_neighbors(test_behavior, distance=1)
    print(f"Center: {test_behavior}")
    print(f"Total neighbors (distance=1): {len(neighbors)}")
    print(f"Empty neighbors: {len(empty_neighbors)}")

    # Get best logos
    print("\n" + "="*60)
    print("Top 5 Logos")
    print("="*60)
    best = archive.get_best_logos(5)
    for i, entry in enumerate(best, 1):
        print(f"{i}. {entry.logo_id}: fitness={entry.fitness:.2f}, behavior={entry.behavior}")

    # Test diverse sample
    print("\n" + "="*60)
    print("Diverse Sample Test")
    print("="*60)
    diverse = archive.get_diverse_sample(n=10, quality_threshold=0.75)
    print(f"Got {len(diverse)} diverse high-quality logos")

    # Test coverage metrics
    print("\n" + "="*60)
    print("Coverage Metrics")
    print("="*60)
    metrics = archive.compute_coverage_metrics()
    print(f"Overall coverage: {metrics['overall_coverage']*100:.2f}%")
    print(f"Occupied: {metrics['occupied_cells']}/{metrics['total_cells']}")
    print("Per-dimension coverage:")
    for dim, cov in metrics['per_dimension_coverage'].items():
        print(f"  {dim}: {cov*100:.1f}%")
    print(f"Quality mean: {metrics['quality_distribution']['mean']:.2f}")
    print(f"Diversity score: {metrics['diversity_metrics']['average_behavior_distance']:.2f}")


if __name__ == "__main__":
    demo()
