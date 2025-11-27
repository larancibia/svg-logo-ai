"""
Initialize RAG Knowledge Base with successful logos from previous experiment
"""

import json
from pathlib import Path
import sys

from rag_evolutionary_system import RAGEvolutionarySystem
from experiment_tracker import ExperimentTracker


def load_previous_experiment_logos():
    """Load successful logos from experiment_20251127_053108"""

    exp_dir = Path("/home/luis/svg-logo-ai/experiments/experiment_20251127_053108")

    if not exp_dir.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return []

    # Load final population
    pop_file = exp_dir / "final_population.json"
    if not pop_file.exists():
        print(f"❌ Population file not found: {pop_file}")
        return []

    with open(pop_file, 'r') as f:
        population = json.load(f)

    # Load SVG files
    logos = []
    for individual in population:
        logo_id = individual['id']
        svg_file = exp_dir / f"{logo_id}.svg"

        if svg_file.exists():
            with open(svg_file, 'r') as f:
                svg_code = f.read()

            logos.append({
                "logo_id": logo_id,
                "svg_code": svg_code,
                "genome": individual['genome'],
                "fitness": individual['fitness'],
                "aesthetic_breakdown": individual['aesthetic_breakdown'],
                "generation": individual['generation']
            })
            print(f"✅ Loaded {logo_id} (fitness: {individual['fitness']})")
        else:
            print(f"⚠️  SVG not found for {logo_id}")

    return logos


def initialize_knowledge_base():
    """Initialize RAG knowledge base with successful logos"""

    print("\n" + "="*80)
    print("INITIALIZING RAG KNOWLEDGE BASE")
    print("="*80 + "\n")

    # Create tracker
    tracker = ExperimentTracker("KB_Initialization")

    tracker.log_step(
        step_type="initialization",
        description="Loading logos from previous experiment (experiment_20251127_053108)"
    )

    # Load logos
    logos = load_previous_experiment_logos()

    if not logos:
        print("❌ No logos found to initialize knowledge base")
        tracker.log_step(
            step_type="error",
            description="Failed to load logos from previous experiment"
        )
        return

    tracker.log_step(
        step_type="data_loaded",
        description=f"Loaded {len(logos)} logos from previous experiment",
        metadata={"num_logos": len(logos)}
    )

    # Create RAG system (this will setup ChromaDB)
    print(f"\n📊 Initializing RAG system with {len(logos)} logos...\n")

    rag_system = RAGEvolutionarySystem(
        use_rag=True,
        tracker=tracker
    )

    # Add each logo to knowledge base
    for logo in logos:
        rag_system.add_logo_to_knowledge_base(
            logo_id=logo['logo_id'],
            svg_code=logo['svg_code'],
            genome=logo['genome'],
            fitness=logo['fitness'],
            aesthetic_breakdown=logo['aesthetic_breakdown']
        )

    print(f"\n✅ Knowledge base initialized with {len(logos)} logos")

    # Test retrieval
    print("\n" + "="*80)
    print("TESTING RAG RETRIEVAL")
    print("="*80 + "\n")

    test_genome = {
        "company": "TestCompany",
        "industry": "artificial intelligence",
        "style_keywords": ["minimalist", "modern", "elegant"],
        "design_principles": ["symmetry", "golden_ratio"],
        "complexity_target": 20
    }

    examples = rag_system.retrieve_similar_logos(test_genome, k=3)

    print(f"\n📚 Retrieved {len(examples)} similar logos:\n")
    for i, ex in enumerate(examples, 1):
        print(f"{i}. {ex['logo_id']} (fitness: {ex['fitness']}/100)")
        print(f"   Style: {', '.join(ex['genome'].get('style_keywords', []))}")
        print(f"   Principles: {', '.join(ex['genome'].get('design_principles', []))}")
        print()

    # Finalize tracking
    tracker.log_result(
        result_type="kb_initialization_complete",
        metrics={
            "num_logos_added": len(logos),
            "avg_fitness": sum(l['fitness'] for l in logos) / len(logos),
            "max_fitness": max(l['fitness'] for l in logos)
        },
        description="Knowledge base successfully initialized"
    )

    trace_path = tracker.finalize()
    print(f"\n📄 Initialization trace saved to: {trace_path}")

    return rag_system


if __name__ == "__main__":
    initialize_knowledge_base()
