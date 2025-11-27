"""
RAG-Enhanced Evolutionary Logo System
Uses ChromaDB to retrieve successful logo examples for few-shot learning
"""

import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import time
from datetime import datetime

from experiment_tracker import ExperimentTracker
from evolutionary_logo_system import Individual, EvolutionaryLogoSystem


class RAGEvolutionarySystem(EvolutionaryLogoSystem):
    """
    Enhanced evolutionary system with RAG (Retrieval-Augmented Generation).
    Retrieves similar successful logos from ChromaDB to provide few-shot examples.
    """

    def __init__(
        self,
        population_size: int = 10,
        elite_size: int = 2,
        mutation_rate: float = 0.3,
        tournament_size: int = 3,
        use_rag: bool = True,
        top_k_examples: int = 3,
        tracker: Optional[ExperimentTracker] = None
    ):
        super().__init__(
            population_size=population_size,
            elite_size=elite_size,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size
        )

        self.use_rag = use_rag
        self.top_k_examples = top_k_examples

        # Initialize tracker
        self.tracker = tracker or ExperimentTracker(
            experiment_name=f"RAG_Evolution"
        )

        # Initialize ChromaDB for logo knowledge base
        if self.use_rag:
            self._setup_knowledge_base()

        self.tracker.log_decision(
            decision=f"Use RAG: {use_rag}",
            rationale="RAG provides few-shot examples of successful logos to guide LLM generation",
            alternatives=["Pure evolutionary without RAG", "Rule-based generation"],
            metadata={"top_k_examples": top_k_examples}
        )

    def _setup_knowledge_base(self):
        """Setup ChromaDB knowledge base for logo retrieval"""
        self.tracker.log_step(
            step_type="setup",
            description="Setting up ChromaDB knowledge base for RAG"
        )

        chroma_path = Path("/home/luis/svg-logo-ai/chroma_db")
        if not chroma_path.exists():
            self.tracker.log_step(
                step_type="warning",
                description=f"ChromaDB not found at {chroma_path}. Creating new knowledge base.",
                metadata={"path": str(chroma_path)}
            )
            chroma_path.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection for logos
        try:
            self.logo_collection = self.chroma_client.get_collection(name="logo_knowledge")
            num_logos = self.logo_collection.count()
            self.tracker.log_step(
                step_type="setup",
                description=f"Loaded existing knowledge base with {num_logos} logos",
                metadata={"collection_size": num_logos}
            )
        except:
            self.logo_collection = self.chroma_client.create_collection(
                name="logo_knowledge",
                metadata={"description": "Knowledge base of successful logos"}
            )
            self.tracker.log_step(
                step_type="setup",
                description="Created new logo knowledge base collection"
            )

    def add_logo_to_knowledge_base(
        self,
        logo_id: str,
        svg_code: str,
        genome: Dict[str, Any],
        fitness: float,
        aesthetic_breakdown: Dict[str, float]
    ):
        """
        Add a successful logo to the knowledge base

        Args:
            logo_id: Unique identifier
            svg_code: SVG code
            genome: Design genome
            fitness: Fitness score
            aesthetic_breakdown: Detailed aesthetic metrics
        """
        if not self.use_rag:
            return

        # Create document for semantic search
        document = f"""
Logo ID: {logo_id}
Company: {genome.get('company', 'Unknown')}
Industry: {genome.get('industry', 'Unknown')}
Style: {', '.join(genome.get('style_keywords', []))}
Design Principles: {', '.join(genome.get('design_principles', []))}
Colors: {', '.join(genome.get('color_palette', []))}
Complexity: {genome.get('complexity_target', 0)}
Fitness: {fitness}/100

Aesthetic Breakdown:
- Golden Ratio: {aesthetic_breakdown.get('golden_ratio', 0):.1f}
- Color Harmony: {aesthetic_breakdown.get('color_harmony', 0):.1f}
- Visual Interest: {aesthetic_breakdown.get('visual_interest', 0):.1f}
- Professional: {aesthetic_breakdown.get('professional', 0):.1f}
"""

        metadata = {
            "logo_id": logo_id,
            "company": genome.get('company', ''),
            "industry": genome.get('industry', ''),
            "fitness": fitness,
            "timestamp": datetime.now().isoformat(),
            **{f"aesthetic_{k}": v for k, v in aesthetic_breakdown.items()}
        }

        # Store in ChromaDB
        self.logo_collection.add(
            ids=[logo_id],
            documents=[document],
            metadatas=[metadata],
            # Store SVG and genome in metadata (ChromaDB supports this)
        )

        # Also store full data separately
        full_data_path = Path("/home/luis/svg-logo-ai/chroma_db/logos")
        full_data_path.mkdir(exist_ok=True)
        with open(full_data_path / f"{logo_id}.json", 'w') as f:
            json.dump({
                "logo_id": logo_id,
                "svg_code": svg_code,
                "genome": genome,
                "fitness": fitness,
                "aesthetic_breakdown": aesthetic_breakdown
            }, f, indent=2)

        self.tracker.log_step(
            step_type="knowledge_base_update",
            description=f"Added logo {logo_id} to knowledge base",
            metadata={"fitness": fitness, "logo_id": logo_id}
        )

    def retrieve_similar_logos(
        self,
        query_genome: Dict[str, Any],
        k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve k most similar successful logos from knowledge base

        Args:
            query_genome: Genome to find similar logos for
            k: Number of examples to retrieve (default: self.top_k_examples)

        Returns:
            List of similar logo examples with their data
        """
        if not self.use_rag:
            return []

        k = k or self.top_k_examples

        # Create query from genome
        query_text = f"""
Industry: {query_genome.get('industry', '')}
Style: {', '.join(query_genome.get('style_keywords', []))}
Design Principles: {', '.join(query_genome.get('design_principles', []))}
Complexity: {query_genome.get('complexity_target', 0)}
"""

        # Query ChromaDB
        try:
            results = self.logo_collection.query(
                query_texts=[query_text],
                n_results=min(k, self.logo_collection.count()),
                where={"fitness": {"$gte": 85}}  # Only retrieve high-quality logos
            )

            # Load full data for retrieved logos
            examples = []
            for i, logo_id in enumerate(results['ids'][0]):
                full_data_path = Path(f"/home/luis/svg-logo-ai/chroma_db/logos/{logo_id}.json")
                if full_data_path.exists():
                    with open(full_data_path, 'r') as f:
                        data = json.load(f)
                        examples.append(data)

            self.tracker.log_step(
                step_type="rag_retrieval",
                description=f"Retrieved {len(examples)} similar logos for guidance",
                metadata={
                    "query_industry": query_genome.get('industry'),
                    "num_retrieved": len(examples),
                    "avg_fitness": sum(ex['fitness'] for ex in examples) / len(examples) if examples else 0
                }
            )

            return examples

        except Exception as e:
            self.tracker.log_step(
                step_type="error",
                description=f"Error retrieving logos: {str(e)}",
                metadata={"error": str(e)}
            )
            return []

    def generate_svg_with_rag(
        self,
        genome: Dict[str, Any],
        examples: List[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Generate SVG with RAG-enhanced prompt (few-shot examples)

        Args:
            genome: Logo design genome
            examples: Retrieved similar logos (if None, will retrieve)

        Returns:
            SVG code or None if generation fails
        """
        # Retrieve examples if not provided
        if examples is None and self.use_rag:
            examples = self.retrieve_similar_logos(genome)

        # Build enhanced prompt with few-shot examples
        prompt = self._build_rag_prompt(genome, examples)

        # Generate with tracking
        self.tracker.log_step(
            step_type="llm_generation",
            description=f"Generating SVG with {len(examples) if examples else 0} RAG examples",
            metadata={
                "num_examples": len(examples) if examples else 0,
                "company": genome.get('company'),
                "industry": genome.get('industry')
            }
        )

        try:
            response = self.model.generate_content(prompt)
            svg_code = self._extract_svg_from_response(response.text)

            if svg_code:
                self.tracker.log_step(
                    step_type="generation_success",
                    description="Successfully generated SVG with RAG",
                    metadata={"svg_length": len(svg_code)}
                )
            else:
                self.tracker.log_step(
                    step_type="generation_failure",
                    description="Failed to extract valid SVG from response"
                )

            return svg_code

        except Exception as e:
            self.tracker.log_step(
                step_type="error",
                description=f"Error during SVG generation: {str(e)}",
                metadata={"error": str(e)}
            )
            return None

    def _build_rag_prompt(
        self,
        genome: Dict[str, Any],
        examples: List[Dict[str, Any]]
    ) -> str:
        """
        Build enhanced prompt with few-shot examples from RAG
        """
        # Base prompt
        prompt = f"""You are an expert logo designer. Create a professional SVG logo based on the following specifications.

COMPANY: {genome['company']}
INDUSTRY: {genome['industry']}
STYLE: {', '.join(genome.get('style_keywords', []))}
COLORS: {', '.join(genome.get('color_palette', []))}
DESIGN PRINCIPLES: {', '.join(genome.get('design_principles', []))}
COMPLEXITY TARGET: {genome.get('complexity_target', 25)} elements (optimal range: 20-40)
"""

        # Add few-shot examples if available
        if examples:
            prompt += f"\n\n{'='*80}\nSUCCESSFUL LOGO EXAMPLES FOR INSPIRATION:\n{'='*80}\n\n"
            prompt += "Study these high-quality logos (fitness 85+/100) to understand what makes an excellent logo:\n\n"

            for i, example in enumerate(examples, 1):
                ex_genome = example['genome']
                prompt += f"""
EXAMPLE {i} (Fitness: {example['fitness']}/100):
Style: {', '.join(ex_genome.get('style_keywords', []))}
Principles: {', '.join(ex_genome.get('design_principles', []))}
Aesthetic Scores:
  - Golden Ratio: {example['aesthetic_breakdown'].get('golden_ratio', 0):.1f}/100
  - Color Harmony: {example['aesthetic_breakdown'].get('color_harmony', 0):.1f}/100
  - Visual Interest: {example['aesthetic_breakdown'].get('visual_interest', 0):.1f}/100

SVG CODE:
{example['svg_code']}

---
"""

            prompt += f"\n{'='*80}\n"
            prompt += "Now create a NEW logo that:\n"
            prompt += "1. Learns from the design patterns in these examples\n"
            prompt += "2. Achieves similar or better aesthetic scores\n"
            prompt += "3. Is UNIQUE and appropriate for the target company/industry\n"
            prompt += "4. Applies the golden ratio (φ=1.618) for proportions\n"
            prompt += "5. Uses the specified colors and style\n\n"

        else:
            # No examples - standard prompt
            prompt += "\n\nCreate a logo that:\n"
            prompt += "1. Is professional and visually appealing\n"
            prompt += "2. Applies the golden ratio (φ=1.618) for proportions\n"
            prompt += "3. Has strong color harmony\n"
            prompt += "4. Is appropriate for the industry\n\n"

        prompt += """
REQUIREMENTS:
- Output ONLY valid SVG code (no markdown, no explanations)
- Use viewBox="0 0 100 100" for scalability
- Include detailed comments explaining design choices
- Target ~{} elements for optimal complexity
- Ensure high technical quality (valid paths, proper syntax)

Generate the SVG code now:
""".format(genome.get('complexity_target', 25))

        return prompt

    def evolve_with_rag(self) -> Dict[str, Any]:
        """
        Run full evolutionary process with RAG enhancement
        """
        self.tracker.log_step(
            step_type="evolution_start",
            description=f"Starting RAG-enhanced evolution",
            metadata={
                "population_size": self.population_size,
                "use_rag": self.use_rag,
                "top_k_examples": self.top_k_examples
            }
        )

        # Override generation method to use RAG
        original_generate = self.generate_svg
        self.generate_svg = self.generate_svg_with_rag

        # Run evolution
        results = self.evolve()

        # Restore original method
        self.generate_svg = original_generate

        # Add successful logos to knowledge base
        self.tracker.log_step(
            step_type="knowledge_base_update",
            description="Adding successful logos to knowledge base"
        )

        for individual in self.population:
            if individual.fitness >= 85:  # Only store high-quality logos
                self.add_logo_to_knowledge_base(
                    logo_id=individual.id,
                    svg_code=individual.phenotype,  # Use phenotype instead of svg_code
                    genome=individual.genome,
                    fitness=individual.fitness,
                    aesthetic_breakdown=individual.aesthetic_breakdown
                )

        # Log final results
        best_fitness = max(ind.fitness for ind in self.population)
        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)

        self.tracker.log_result(
            result_type="rag_evolution_complete",
            metrics={
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "improvement": best_fitness - avg_fitness
            },
            description=f"RAG-enhanced evolution completed"
        )

        return results
