"""
MAP-Elites Archive for Logo Generation
=======================================
Maintains a multi-dimensional grid of diverse, high-quality logos.

Each cell in the grid represents a unique combination of behavioral characteristics.
Only the best logo (highest fitness) is kept per cell.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import chromadb
from chromadb.config import Settings
from dataclasses import dataclass, asdict


@dataclass
class ArchiveEntry:
    """Single entry in the MAP-Elites archive"""
    logo_id: str
    svg_code: str
    genome: Dict
    fitness: float
    aesthetic_breakdown: Dict
    behavior: Tuple[int, int, int, int]  # (complexity_bin, style_bin, symmetry_bin, color_bin)
    raw_behavior: Dict  # Raw behavior scores
    generation: int
    parent_ids: List[str] = None

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
            'parent_ids': self.parent_ids or []
        }


class MAPElitesArchive:
    """
    MAP-Elites archive maintaining diverse logo population

    Structure: N-dimensional grid where each cell contains the best logo
    for that behavioral niche.
    """

    def __init__(self,
                 dimensions: Tuple[int, ...] = (10, 10, 10, 10),
                 chroma_db_path: str = "/home/luis/svg-logo-ai/chroma_db/map_elites"):
        """
        Initialize MAP-Elites archive

        Args:
            dimensions: Tuple of grid dimensions (e.g., (10, 10, 10, 10) for 4D grid)
            chroma_db_path: Path to ChromaDB storage
        """
        self.dimensions = dimensions
        self.archive = {}  # Key: behavior tuple, Value: ArchiveEntry
        self.chroma_db_path = Path(chroma_db_path)

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
            behavior: Tuple[int, int, int, int],
            raw_behavior: Dict,
            generation: int,
            parent_ids: List[str] = None) -> bool:
        """
        Add logo to archive (only if it's better than existing entry in that cell)

        Args:
            logo_id: Unique identifier
            svg_code: SVG code
            genome: Genome dictionary
            fitness: Fitness score
            aesthetic_breakdown: Breakdown of aesthetic metrics
            behavior: Binned behavior coordinates (complexity, style, symmetry, color)
            raw_behavior: Raw behavior scores
            generation: Generation number
            parent_ids: Parent logo IDs

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
                parent_ids=parent_ids
            )

            # Update archive
            self.archive[behavior] = entry

            # Store in ChromaDB
            self._store_in_chromadb(entry)

            return True

        return False

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

    def _store_in_chromadb(self, entry: ArchiveEntry):
        """Store entry in ChromaDB for retrieval and analysis"""
        # Create searchable document
        genome = entry.genome
        document = f"""
        Company: {genome.get('company', 'unknown')}
        Industry: {genome.get('industry', 'unknown')}
        Style: {' '.join(genome.get('style_keywords', []))}
        Colors: {' '.join(genome.get('color_palette', []))}
        Principles: {' '.join(genome.get('design_principles', []))}
        Behavior: Complexity={entry.behavior[0]} Style={entry.behavior[1]} Symmetry={entry.behavior[2]} Color={entry.behavior[3]}
        """

        # Metadata
        metadata = {
            'logo_id': entry.logo_id,
            'fitness': entry.fitness,
            'generation': entry.generation,
            'behavior_0': entry.behavior[0],
            'behavior_1': entry.behavior[1],
            'behavior_2': entry.behavior[2],
            'behavior_3': entry.behavior[3],
            'complexity_raw': entry.raw_behavior['complexity'],
            'style_raw': entry.raw_behavior['style'],
            'symmetry_raw': entry.raw_behavior['symmetry'],
            'color_richness_raw': entry.raw_behavior['color_richness'],
            'svg_code': entry.svg_code,
            'genome_json': json.dumps(entry.genome),
            'aesthetic_json': json.dumps(entry.aesthetic_breakdown)
        }

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

                # Recreate entry
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
                    parent_ids=entry_data.get('parent_ids', [])
                )

                self.archive[behavior] = entry

        print(f"Loaded {len(self.archive)} logos from {archive_file}")


def demo():
    """Demo MAP-Elites archive"""
    print("="*60)
    print("MAP-Elites Archive Demo")
    print("="*60)

    # Create small 3D archive for testing
    archive = MAPElitesArchive(dimensions=(5, 5, 5, 5))

    # Add some dummy entries
    for i in range(20):
        behavior = (
            random.randint(0, 4),
            random.randint(0, 4),
            random.randint(0, 4),
            random.randint(0, 4)
        )

        added = archive.add(
            logo_id=f"logo_{i}",
            svg_code=f"<svg>Logo {i}</svg>",
            genome={'test': i},
            fitness=random.uniform(70, 95),
            aesthetic_breakdown={'aesthetic': random.uniform(70, 95)},
            behavior=behavior,
            raw_behavior={'complexity': 20, 'style': 0.5, 'symmetry': 0.5, 'color_richness': 0.5},
            generation=0
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
    test_behavior = (2, 2, 2, 2)
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


if __name__ == "__main__":
    demo()
