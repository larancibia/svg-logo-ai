# LLM-Guided Quality-Diversity Logo System Architecture

**Version:** 1.0
**Date:** November 27, 2025
**Author:** Architecture Agent

---

## Executive Summary

This document specifies the architecture for a revolutionary logo generation system that combines:
- **LLM Semantic Intelligence**: Natural language understanding and generation
- **Quality-Diversity Algorithms**: MAP-Elites for systematic design space exploration
- **Enhanced Behavioral Space**: 5D characterization (complexity, style, symmetry, color, emotion)
- **Natural Language Control**: User-driven search via conversational queries

**Key Innovation**: Unlike existing systems (pure evolutionary, RAG-enhanced, or basic MAP-Elites), this LLM-QD system enables users to explore diverse, high-quality logo portfolios through natural language, with the LLM acting as both generator and intelligent search guide.

**Expected Outcomes**:
- 3,000-5,000 diverse logos covering a 5D behavioral space
- Natural language query interface: "Show me minimalist tech logos with warm colors"
- Quality scores of 85-95/100 across the entire archive
- Interactive visualization and filtering

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagrams](#architecture-diagrams)
3. [Component Specifications](#component-specifications)
4. [Enhanced Behavioral Dimensions](#enhanced-behavioral-dimensions)
5. [LLM Integration Points](#llm-integration-points)
6. [Data Structures](#data-structures)
7. [API Interfaces](#api-interfaces)
8. [File Structure](#file-structure)
9. [Implementation Roadmap](#implementation-roadmap)
10. [Experiment Protocol](#experiment-protocol)

---

## 1. System Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                         │
│  ┌───────────────────┐      ┌─────────────────────────────────┐   │
│  │ Natural Language  │      │   Interactive Visualization     │   │
│  │  Query Interface  │──────│   (Grid Explorer + Filters)     │   │
│  └───────────────────┘      └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      INTELLIGENCE LAYER (LLM)                       │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │  Query Parser   │  │ Semantic Mutator │  │  LLM Evaluator  │  │
│  │  (Intent→Goals) │  │ (Directed Mods)  │  │  (Judge+Emotion)│  │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   QUALITY-DIVERSITY LAYER                           │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              Enhanced QD Archive (5D Grid)                   │  │
│  │  Dimensions: [Complexity][Style][Symmetry][Color][Emotion]  │  │
│  │              10 x 10 x 10 x 10 x 10 = 100,000 cells        │  │
│  └─────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────┐          ┌───────────────────────────────┐ │
│  │  MAP-Elites Loop │◄────────►│  Behavior Characterization    │ │
│  │  (Exploration)   │          │  (5D Feature Extraction)      │ │
│  └──────────────────┘          └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       GENERATION LAYER                              │
│  ┌─────────────────┐          ┌───────────────────────────────┐   │
│  │ LLM Logo Gen    │          │   Aesthetic Quality Metrics   │   │
│  │ (Gemini 2.5)    │◄────────►│   (Golden Ratio, Harmony)     │   │
│  └─────────────────┘          └───────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow: Natural Language Query → Diverse Logos

```
User Query: "Show me vibrant, asymmetric tech logos with high complexity"
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ NaturalLanguageQueryParser                                      │
│  - Extract semantic goals: {emotion: "energetic/vibrant",      │
│                            symmetry: "low",                     │
│                            complexity: "high",                  │
│                            industry: "tech"}                    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ LLMGuidedQDSearch                                              │
│  - Target behavioral region: emotion[7-9], complexity[7-9],    │
│                             symmetry[0-3], color[6-9]          │
│  - Run directed MAP-Elites toward this region                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ MAP-Elites Iterations (LLM-Guided)                             │
│  For each iteration:                                           │
│    1. Select source logo from occupied cell                    │
│    2. Identify target cell in desired region                   │
│    3. SemanticMutator: LLM modifies logo toward target         │
│    4. Evaluate: Quality + 5D Behavior                          │
│    5. Add to EnhancedQDArchive if better/new                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ DiversityVisualizer                                            │
│  - Display logos matching query criteria                       │
│  - Interactive 5D grid navigation                              │
│  - Filter by fitness, behavior, or semantic tags               │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
User receives: Portfolio of 20-100 diverse logos matching intent
```

### 1.3 Key Components and Interactions

| Component | Purpose | Dependencies |
|-----------|---------|--------------|
| `NaturalLanguageQueryParser` | Converts user queries to behavioral goals | Gemini 2.5 Flash API |
| `LLMLogoGenerator` | Generates SVGs from text descriptions | Gemini 2.5 Flash API |
| `SemanticMutator` | Intelligently modifies logos toward targets | Gemini 2.5, BehaviorCharacterizer |
| `EnhancedBehaviorCharacterizer` | Extracts 5D behavioral features | LogoValidator, Gemini (for emotion) |
| `EnhancedQDArchive` | 5D MAP-Elites archive | ChromaDB, existing MAPElitesArchive |
| `LLMGuidedQDSearch` | Main search orchestrator | All above components |
| `DiversityVisualizer` | Interactive results display | EnhancedQDArchive |
| `EmotionalToneAnalyzer` | LLM-based emotion classification | Gemini 2.5 Flash API |

---

## 2. Architecture Diagrams

### 2.1 System Component Diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│                        LLM-QD Logo System                             │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ src/llm_qd/                                                  │    │
│  │                                                              │    │
│  │  ├── query_parser.py                                        │    │
│  │  │   └── NaturalLanguageQueryParser                         │    │
│  │  │       • parse_query(text) → BehavioralGoals              │    │
│  │  │       • extract_constraints(query) → Dict                │    │
│  │  │                                                           │    │
│  │  ├── llm_logo_generator.py                                  │    │
│  │  │   └── LLMLogoGenerator                                   │    │
│  │  │       • generate(prompt, genome) → SVG                   │    │
│  │  │       • batch_generate(prompts) → List[SVG]              │    │
│  │  │                                                           │    │
│  │  ├── semantic_mutator.py                                    │    │
│  │  │   └── SemanticMutator                                    │    │
│  │  │       • mutate_toward_emotion(svg, target) → SVG         │    │
│  │  │       • mutate_combined(svg, targets) → SVG              │    │
│  │  │                                                           │    │
│  │  ├── enhanced_behavior_characterizer.py                     │    │
│  │  │   └── EnhancedBehaviorCharacterizer                      │    │
│  │  │       • characterize_5d(svg) → 5D bins                   │    │
│  │  │       • extract_emotion(svg) → emotion_score             │    │
│  │  │                                                           │    │
│  │  ├── enhanced_qd_archive.py                                 │    │
│  │  │   └── EnhancedQDArchive                                  │    │
│  │  │       • add(logo, behavior_5d, fitness) → bool           │    │
│  │  │       • query_by_emotion(range) → List[Logo]             │    │
│  │  │       • get_behavioral_neighbors(cell) → List[Cell]      │    │
│  │  │                                                           │    │
│  │  ├── llm_guided_qd_search.py                                │    │
│  │  │   └── LLMGuidedQDSearch                                  │    │
│  │  │       • run_search(query, iterations) → Archive          │    │
│  │  │       • directed_exploration(goals) → Archive            │    │
│  │  │                                                           │    │
│  │  ├── diversity_visualizer.py                                │    │
│  │  │   └── DiversityVisualizer                                │    │
│  │  │       • generate_html_grid(archive) → HTML               │    │
│  │  │       • export_portfolio(logos, format) → Files          │    │
│  │  │                                                           │    │
│  │  └── emotional_tone_analyzer.py                             │    │
│  │      └── EmotionalToneAnalyzer                              │    │
│  │          • analyze_emotion(svg) → (tone, score)             │    │
│  │          • classify_emotion_bin(score) → bin_index          │    │
│  │                                                              │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  Uses existing components:                                            │
│  • src/logo_validator.py (aesthetic metrics)                          │
│  • src/behavior_characterization.py (4D base)                        │
│  • src/map_elites_archive.py (archive structure)                     │
│  • src/llm_guided_mutation.py (mutation logic)                       │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### 2.2 LLM Integration Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    LLM API Call Types                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ 1. GENERATION                                                    │
│    Input: Text prompt + genome parameters                       │
│    Output: SVG code                                             │
│    Model: Gemini 2.5 Flash                                      │
│    Frequency: High (every logo generation)                      │
│                                                                  │
│ 2. MUTATION                                                      │
│    Input: Source SVG + behavioral delta instructions            │
│    Output: Modified SVG                                         │
│    Model: Gemini 2.5 Flash                                      │
│    Frequency: High (every MAP-Elites iteration)                 │
│                                                                  │
│ 3. EMOTION ANALYSIS                                              │
│    Input: SVG code                                              │
│    Output: Emotion classification (calm/energetic/playful/...)  │
│    Model: Gemini 2.5 Flash                                      │
│    Frequency: High (every evaluation)                           │
│                                                                  │
│ 4. QUERY PARSING                                                 │
│    Input: Natural language query                                │
│    Output: Behavioral goals (JSON)                              │
│    Model: Gemini 2.5 Flash                                      │
│    Frequency: Low (once per user query)                         │
│                                                                  │
│ 5. FITNESS EVALUATION (Optional)                                 │
│    Input: SVG code                                              │
│    Output: Aesthetic critique + score                           │
│    Model: Gemini 2.5 Flash                                      │
│    Frequency: Medium (supplementary to algorithmic metrics)     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

API Call Optimization:
• Batch emotion analysis (analyze 10 SVGs per call)
• Cache query parses for common intents
• Use existing LogoValidator for primary fitness (faster)
• LLM emotion analysis only when not cached
```

---

## 3. Component Specifications

### 3.1 NaturalLanguageQueryParser

**Purpose**: Converts user natural language queries into structured behavioral goals.

**Class Definition**:
```python
class NaturalLanguageQueryParser:
    """
    Parses natural language queries into behavioral search goals.

    Examples:
        "minimalist tech logos" → {complexity: low, style: geometric}
        "vibrant, energetic designs" → {emotion: energetic, color: high}
        "symmetric corporate branding" → {symmetry: high, emotion: professional}
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = genai.GenerativeModel(model)
        self.dimension_mappings = self._load_dimension_vocabulary()

    def parse_query(self, query: str) -> BehavioralGoals:
        """
        Parse query into behavioral goals.

        Args:
            query: Natural language query

        Returns:
            BehavioralGoals with target ranges for each dimension
        """

    def extract_constraints(self, query: str) -> Dict[str, Any]:
        """
        Extract hard constraints (company name, industry, etc.)

        Returns:
            {company, industry, style_keywords, color_palette, must_have, must_not_have}
        """

    def _build_parsing_prompt(self, query: str) -> str:
        """Build LLM prompt for query parsing"""
```

**BehavioralGoals Data Structure**:
```python
@dataclass
class BehavioralGoals:
    """Target behavioral ranges for QD search"""
    complexity: Range  # e.g., Range(7, 9) for "high complexity"
    style: Range       # e.g., Range(0, 3) for "geometric"
    symmetry: Range    # e.g., Range(7, 9) for "symmetric"
    color: Range       # e.g., Range(5, 9) for "colorful"
    emotion: Range     # e.g., Range(6, 8) for "energetic"

    # Constraints
    company: Optional[str] = None
    industry: Optional[str] = None
    style_keywords: List[str] = field(default_factory=list)
    color_palette: List[str] = field(default_factory=list)
```

**Example Usage**:
```python
parser = NaturalLanguageQueryParser()
goals = parser.parse_query("Show me minimalist, calming logos for a meditation app")
# goals.complexity → Range(0, 3)  # minimalist = low complexity
# goals.emotion → Range(0, 2)     # calming = low emotion score
# goals.style → Range(0, 4)       # minimalist = geometric
```

---

### 3.2 LLMLogoGenerator

**Purpose**: Generate SVG logos from text prompts using LLM.

**Class Definition**:
```python
class LLMLogoGenerator:
    """
    LLM-powered SVG logo generation.
    Extends existing gemini_svg_generator.py with QD-specific features.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = genai.GenerativeModel(model)
        self.validator = LogoValidator()

    def generate(self,
                 prompt: str,
                 genome: Optional[Dict] = None,
                 behavioral_hints: Optional[Dict] = None) -> str:
        """
        Generate SVG logo.

        Args:
            prompt: Text description
            genome: Optional genome parameters
            behavioral_hints: Optional hints about desired behavior
                e.g., {"emotion": "calm", "complexity": "low"}

        Returns:
            SVG code string
        """

    def batch_generate(self, prompts: List[str], n_variations: int = 3) -> List[str]:
        """Generate multiple logos in parallel"""

    def _enhance_prompt_with_behaviors(self,
                                       base_prompt: str,
                                       behavioral_hints: Dict) -> str:
        """Add behavioral guidance to prompt"""
```

**Integration with Existing Code**:
- Extends `src/gemini_svg_generator.py` or `src/gemini_svg_generator_v2.py`
- Adds behavioral hint system
- Maintains compatibility with evolutionary_logo_system genome format

---

### 3.3 SemanticMutator

**Purpose**: LLM-guided mutations toward specific emotional tones.

**Class Definition**:
```python
class SemanticMutator:
    """
    Intelligent mutations guided by semantic understanding.
    Extends existing llm_guided_mutation.py with emotion-aware mutations.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.base_mutator = LLMGuidedMutator(model)
        self.emotion_analyzer = EmotionalToneAnalyzer(model)

    def mutate_toward_emotion(self,
                             source_svg: str,
                             current_emotion: str,
                             target_emotion: str,
                             intensity: float = 0.5) -> str:
        """
        Mutate logo to shift emotional tone.

        Args:
            source_svg: Current SVG
            current_emotion: Current emotional classification
            target_emotion: Target emotional tone
            intensity: Mutation strength (0-1)

        Returns:
            Modified SVG

        Example:
            mutate_toward_emotion(svg, "calm", "energetic", intensity=0.7)
            → Increases color saturation, adds dynamic elements
        """

    def mutate_combined(self,
                       source_svg: str,
                       current_behavior: Tuple[int, int, int, int, int],
                       target_behavior: Tuple[int, int, int, int, int],
                       genome: Dict) -> str:
        """
        Mutate toward target across ALL 5 dimensions.

        Uses LLM to intelligently combine:
        - Complexity changes
        - Style shifts (geometric ↔ organic)
        - Symmetry adjustments
        - Color modifications
        - Emotional tone shifts
        """

    def _build_emotion_mutation_prompt(self,
                                      svg: str,
                                      emotion_delta: str) -> str:
        """Build prompt for emotion-driven mutation"""
```

**Emotion Mutation Strategies**:

| Source Emotion | Target Emotion | Mutation Strategy |
|----------------|----------------|-------------------|
| Calm → Energetic | Increase color saturation, add angular shapes, introduce asymmetry |
| Professional → Playful | Add organic curves, use brighter colors, introduce whimsy |
| Elegant → Bold | Increase stroke width, use stronger colors, enlarge elements |
| Minimal → Expressive | Add decorative elements, increase complexity, vary line weights |

---

### 3.4 EnhancedBehaviorCharacterizer

**Purpose**: Extract 5D behavioral features (adds emotion to existing 4D).

**Class Definition**:
```python
class EnhancedBehaviorCharacterizer(BehaviorCharacterizer):
    """
    Extends BehaviorCharacterizer with 5th dimension: Emotional Tone.

    Inherits from existing src/behavior_characterization.py
    Adds LLM-based emotion analysis.
    """

    def __init__(self, num_bins: int = 10):
        super().__init__(num_bins)
        self.emotion_analyzer = EmotionalToneAnalyzer()

    def characterize_5d(self, svg_code: str) -> Dict:
        """
        Compute 5D behavioral features.

        Returns:
            {
                'raw_scores': {
                    'complexity': int,
                    'style': float,
                    'symmetry': float,
                    'color_richness': float,
                    'emotional_tone': float  # NEW: 0-1 scale
                },
                'bins': (complexity_bin, style_bin, symmetry_bin,
                        color_bin, emotion_bin),
                'emotion_label': str,  # NEW: human-readable emotion
                'details': {...}
            }
        """
        # Get 4D features from parent class
        result_4d = self.characterize(svg_code)

        # Add 5th dimension: emotion
        emotion_score, emotion_label = self.emotion_analyzer.analyze_emotion(svg_code)

        # Combine results
        result_5d = {
            'raw_scores': {
                **result_4d['raw_scores'],
                'emotional_tone': emotion_score
            },
            'bins': (*result_4d['bins'], self._discretize_emotion(emotion_score)),
            'emotion_label': emotion_label,
            'details': result_4d['details']
        }

        return result_5d

    def _discretize_emotion(self, emotion_score: float) -> int:
        """Map continuous emotion score to bin index"""
```

**Emotion Score Mapping**:
```
0.0 - 0.1 → Bin 0: Calm/Serene
0.1 - 0.2 → Bin 1: Professional/Corporate
0.2 - 0.3 → Bin 2: Elegant/Refined
0.3 - 0.4 → Bin 3: Friendly/Approachable
0.4 - 0.5 → Bin 4: Balanced/Neutral
0.5 - 0.6 → Bin 5: Confident/Bold
0.6 - 0.7 → Bin 6: Energetic/Dynamic
0.7 - 0.8 → Bin 7: Playful/Whimsical
0.8 - 0.9 → Bin 8: Exciting/Vibrant
0.9 - 1.0 → Bin 9: Intense/Aggressive
```

---

### 3.5 EnhancedQDArchive

**Purpose**: 5D MAP-Elites archive with emotion dimension.

**Class Definition**:
```python
class EnhancedQDArchive(MAPElitesArchive):
    """
    5D MAP-Elites archive: [Complexity][Style][Symmetry][Color][Emotion]

    Extends existing src/map_elites_archive.py
    Grid size: 10x10x10x10x10 = 100,000 cells
    """

    def __init__(self,
                 dimensions: Tuple[int, ...] = (10, 10, 10, 10, 10),
                 chroma_db_path: str = "/home/luis/svg-logo-ai/chroma_db/llm_qd"):
        super().__init__(dimensions, chroma_db_path)
        self.dimension_names = ['complexity', 'style', 'symmetry',
                               'color_richness', 'emotional_tone']

    def query_by_emotion(self,
                         emotion_range: Tuple[int, int]) -> List[ArchiveEntry]:
        """
        Query logos by emotional tone range.

        Args:
            emotion_range: (min_bin, max_bin) for emotion dimension

        Returns:
            All logos with emotion in specified range
        """

    def get_behavioral_slice(self,
                            dimension: str,
                            value: int) -> List[ArchiveEntry]:
        """
        Get all logos with specific value in one dimension.

        Example: get_behavioral_slice('emotion', 7)
                 → All playful/whimsical logos
        """

    def get_region_statistics(self,
                            complexity_range: Optional[Range] = None,
                            style_range: Optional[Range] = None,
                            symmetry_range: Optional[Range] = None,
                            color_range: Optional[Range] = None,
                            emotion_range: Optional[Range] = None) -> Dict:
        """
        Get statistics for a specific behavioral region.

        Returns:
            {
                'count': int,
                'coverage': float,
                'avg_fitness': float,
                'best_logo': ArchiveEntry
            }
        """
```

**ChromaDB Schema Enhancement**:
```python
# Additional metadata for 5D archive
metadata = {
    # Existing 4D
    'behavior_0': complexity_bin,
    'behavior_1': style_bin,
    'behavior_2': symmetry_bin,
    'behavior_3': color_bin,

    # NEW: 5th dimension
    'behavior_4': emotion_bin,
    'emotion_label': emotion_label,  # "energetic", "calm", etc.
    'emotion_score_raw': emotion_score,  # 0-1 continuous

    # Enhanced search
    'behavioral_tags': ["minimalist", "tech", "blue", "calm"],
    'semantic_description': "A calm, minimalist tech logo with blue tones"
}
```

---

### 3.6 LLMGuidedQDSearch

**Purpose**: Main search orchestrator combining QD and LLM guidance.

**Class Definition**:
```python
class LLMGuidedQDSearch:
    """
    Main search algorithm: LLM-Guided Quality-Diversity.

    Combines:
    - MAP-Elites exploration
    - LLM semantic understanding
    - Natural language control
    """

    def __init__(self,
                 company_name: str,
                 industry: str,
                 grid_dimensions: Tuple[int, ...] = (10, 10, 10, 10, 10)):
        self.company_name = company_name
        self.industry = industry

        # Initialize components
        self.query_parser = NaturalLanguageQueryParser()
        self.generator = LLMLogoGenerator()
        self.mutator = SemanticMutator()
        self.characterizer = EnhancedBehaviorCharacterizer()
        self.archive = EnhancedQDArchive(dimensions=grid_dimensions)
        self.validator = LogoValidator()
        self.tracker = ExperimentTracker("llm_qd_search")

    def run_search(self,
                   user_query: Optional[str] = None,
                   n_iterations: int = 5000,
                   n_initial: int = 500) -> EnhancedQDArchive:
        """
        Run full LLM-guided QD search.

        Args:
            user_query: Optional natural language query to guide search
            n_iterations: Number of MAP-Elites iterations
            n_initial: Number of random logos for initialization

        Returns:
            Filled archive with diverse logos
        """
        # Parse query (if provided)
        goals = None
        if user_query:
            goals = self.query_parser.parse_query(user_query)
            self._log_search_goals(goals)

        # Initialize archive
        self._initialize_archive(n_initial, goals)

        # Run MAP-Elites with LLM guidance
        if goals:
            self._directed_exploration(goals, n_iterations)
        else:
            self._undirected_exploration(n_iterations)

        return self.archive

    def _initialize_archive(self,
                           n_logos: int,
                           goals: Optional[BehavioralGoals] = None):
        """Initialize archive with random or goal-biased logos"""

    def _directed_exploration(self,
                             goals: BehavioralGoals,
                             n_iterations: int):
        """
        MAP-Elites with preference for target behavioral region.

        Strategy:
        - 70% of iterations target the goal region
        - 30% explore globally (maintain diversity)
        """

    def _undirected_exploration(self, n_iterations: int):
        """Standard MAP-Elites: uniform exploration"""

    def _map_elites_iteration(self,
                             target_region: Optional[Dict] = None) -> bool:
        """
        Single MAP-Elites iteration.

        1. Select source logo
        2. Select target cell (biased by target_region if provided)
        3. Mutate toward target
        4. Evaluate
        5. Add to archive

        Returns:
            True if logo added, False otherwise
        """
```

**Search Strategies**:

```python
# Strategy 1: Directed Search (user query provided)
# Bias exploration toward target region
def _select_target_cell_directed(self, goals: BehavioralGoals) -> Tuple:
    if random.random() < 0.7:  # 70% directed
        return self._sample_from_goal_region(goals)
    else:  # 30% exploration
        return self._sample_random_empty_cell()

# Strategy 2: Undirected Search (no query)
# Uniform exploration of entire space
def _select_target_cell_undirected(self) -> Tuple:
    source_behavior, _ = self.archive.get_random_occupied()
    return self._sample_empty_neighbor(source_behavior)
```

---

### 3.7 DiversityVisualizer

**Purpose**: Interactive visualization and export of results.

**Class Definition**:
```python
class DiversityVisualizer:
    """
    Visualize and export diverse logo portfolios.

    Generates:
    - Interactive HTML grid explorer
    - PDF/PNG galleries
    - Filtered subsets
    """

    def __init__(self, archive: EnhancedQDArchive):
        self.archive = archive

    def generate_html_grid(self,
                          output_path: str,
                          slice_dims: Tuple[str, str] = ('complexity', 'emotion'),
                          filter_criteria: Optional[Dict] = None) -> str:
        """
        Generate interactive HTML grid visualization.

        Args:
            output_path: Where to save HTML
            slice_dims: Which 2D slice to display
            filter_criteria: Optional filters on other dimensions

        Returns:
            Path to generated HTML file

        Example:
            visualizer.generate_html_grid(
                'output/grid.html',
                slice_dims=('complexity', 'emotion'),
                filter_criteria={'style': Range(0, 3)}  # Only geometric
            )
        """

    def export_portfolio(self,
                        logos: List[ArchiveEntry],
                        output_dir: str,
                        format: str = 'svg') -> str:
        """
        Export selected logos as portfolio.

        Args:
            logos: Logos to export
            output_dir: Output directory
            format: 'svg', 'png', 'pdf', or 'gallery_html'

        Returns:
            Path to output directory
        """

    def generate_behavior_heatmap(self,
                                 dimension1: str,
                                 dimension2: str,
                                 metric: str = 'fitness') -> str:
        """
        Generate heatmap showing metric across 2D slice.

        Args:
            dimension1, dimension2: Behavioral dimensions
            metric: 'fitness', 'coverage', or 'count'

        Returns:
            Path to heatmap image
        """

    def generate_query_results_page(self,
                                   query: str,
                                   matching_logos: List[ArchiveEntry],
                                   output_path: str) -> str:
        """
        Generate HTML page showing query results.

        Displays:
        - Original query
        - Behavioral goals extracted
        - Grid of matching logos with metadata
        - Download links
        """
```

**HTML Grid Example**:
```html
<!-- Interactive 5D Grid Explorer -->
<div id="llm-qd-explorer">
  <div class="controls">
    <select id="x-axis">
      <option value="complexity">Complexity</option>
      <option value="emotion" selected>Emotion</option>
      ...
    </select>
    <select id="y-axis">
      <option value="style" selected>Style</option>
      ...
    </select>

    <!-- Filters for other dimensions -->
    <input type="range" id="filter-symmetry" min="0" max="9" />
  </div>

  <div class="grid-container">
    <!-- 10x10 grid of logos -->
    <div class="grid-cell" data-behavior="3,5,7,2,8" data-fitness="92.3">
      <img src="logo_xyz.svg" />
      <div class="metadata">
        Fitness: 92.3/100<br/>
        Emotion: Playful<br/>
        Complexity: Moderate
      </div>
    </div>
    ...
  </div>
</div>
```

---

### 3.8 EmotionalToneAnalyzer

**Purpose**: LLM-based emotional classification of logos.

**Class Definition**:
```python
class EmotionalToneAnalyzer:
    """
    Analyzes emotional tone of logos using LLM vision capabilities.

    Maps visual characteristics to emotional impressions.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = genai.GenerativeModel(model)
        self.emotion_taxonomy = self._load_emotion_taxonomy()

    def analyze_emotion(self, svg_code: str) -> Tuple[float, str]:
        """
        Analyze emotional tone of a logo.

        Args:
            svg_code: SVG code to analyze

        Returns:
            (emotion_score, emotion_label)
            emotion_score: 0-1 (calm → intense)
            emotion_label: Human-readable classification

        Example:
            score, label = analyzer.analyze_emotion(svg)
            # (0.75, "playful")
        """

    def batch_analyze(self, svg_codes: List[str]) -> List[Tuple[float, str]]:
        """Analyze multiple logos in single API call (efficiency)"""

    def _build_emotion_prompt(self, svg_code: str) -> str:
        """
        Build prompt for emotion analysis.

        Prompt structure:
        - Explain emotion taxonomy (calm → intense spectrum)
        - Show SVG code
        - Ask for classification with reasoning
        """

    def _parse_emotion_response(self, response: str) -> Tuple[float, str]:
        """Parse LLM response into score and label"""

    def _load_emotion_taxonomy(self) -> Dict:
        """
        Load emotion taxonomy.

        Returns:
            {
                'calm': {'score': 0.0, 'characteristics': [...]},
                'professional': {'score': 0.1, 'characteristics': [...]},
                ...
                'intense': {'score': 1.0, 'characteristics': [...]}
            }
        """
```

**Emotion Analysis Prompt**:
```python
EMOTION_ANALYSIS_PROMPT = """
You are a professional brand designer analyzing the emotional tone of a logo.

EMOTION SPECTRUM (0-1):
0.0 - Calm/Serene: Soft colors, minimal elements, gentle curves
0.1 - Professional/Corporate: Clean lines, conservative colors, balanced
0.2 - Elegant/Refined: Sophisticated shapes, tasteful colors, harmony
0.3 - Friendly/Approachable: Rounded shapes, warm colors, inviting
0.4 - Balanced/Neutral: No strong emotional bias
0.5 - Confident/Bold: Strong shapes, solid colors, assertive
0.6 - Energetic/Dynamic: Angular shapes, bright colors, movement
0.7 - Playful/Whimsical: Organic curves, fun colors, creative
0.8 - Exciting/Vibrant: High saturation, complex forms, lively
0.9 - Intense/Aggressive: Sharp angles, strong contrasts, dramatic

LOGO TO ANALYZE:
{svg_code}

TASK:
1. Analyze the visual characteristics
2. Determine the dominant emotional tone
3. Provide a score (0-1) and label

RESPOND IN JSON:
{{
  "emotion_score": 0.7,
  "emotion_label": "playful",
  "reasoning": "The logo uses organic curves, bright colors, and playful shapes..."
}}
"""
```

---

## 4. Enhanced Behavioral Dimensions (5D)

### 4.1 Dimension Definitions

| Dimension | Range | Metric | Description |
|-----------|-------|--------|-------------|
| **Complexity** | 0-9 | Element count | Number of SVG geometric elements (paths, circles, rects, etc.) |
| **Style** | 0-9 | Geometric ↔ Organic | 0 = Pure geometric (lines, circles), 9 = Organic curves (beziers) |
| **Symmetry** | 0-9 | Asymmetric ↔ Symmetric | Degree of reflective/rotational symmetry |
| **Color Richness** | 0-9 | Monochrome ↔ Polychromatic | 0 = 1 color, 9 = Many colors |
| **Emotional Tone** | 0-9 | Calm ↔ Intense | LLM-judged emotional impression |

### 4.2 Complexity Bins

```
Bin 0: 0-15 elements   (Ultra minimal)
Bin 1: 15-20 elements  (Very simple)
Bin 2: 20-25 elements  (Simple)
Bin 3: 25-30 elements  (Moderate-low)
Bin 4: 30-35 elements  (Moderate)
Bin 5: 35-40 elements  (Moderate-high)
Bin 6: 40-45 elements  (Complex)
Bin 7: 45-50 elements  (Very complex)
Bin 8: 50-55 elements  (Highly complex)
Bin 9: 55+ elements    (Ultra complex)
```

### 4.3 Style Bins

```
Bin 0-2: Geometric (rectangles, circles, straight lines)
Bin 3-4: Mixed geometric (some curves)
Bin 5-6: Balanced (mix of geometric and organic)
Bin 7-8: Organic (bezier curves, flowing shapes)
Bin 9:   Pure organic (no straight lines)
```

### 4.4 Symmetry Bins

```
Bin 0-2: Asymmetric (< 30% symmetry match)
Bin 3-4: Slightly asymmetric (30-50%)
Bin 5-6: Partial symmetry (50-70%)
Bin 7-8: Mostly symmetric (70-90%)
Bin 9:   Perfect symmetry (> 90%)
```

### 4.5 Color Richness Bins

```
Bin 0-1: Monochrome (1 color)
Bin 2-3: Duotone (2 colors)
Bin 4-5: Tritone (3 colors)
Bin 6-7: Polychromatic (4 colors)
Bin 8-9: Highly polychromatic (5+ colors)
```

### 4.6 Emotional Tone Bins (NEW)

```
Bin 0: Calm/Serene
Bin 1: Professional/Corporate
Bin 2: Elegant/Refined
Bin 3: Friendly/Approachable
Bin 4: Balanced/Neutral
Bin 5: Confident/Bold
Bin 6: Energetic/Dynamic
Bin 7: Playful/Whimsical
Bin 8: Exciting/Vibrant
Bin 9: Intense/Aggressive
```

**Emotional Tone Characteristics**:

| Emotion | Color Palette | Shapes | Complexity | Typical Use Cases |
|---------|--------------|--------|------------|-------------------|
| Calm | Pastels, blues, greens | Soft curves, circles | Low-moderate | Meditation, wellness, healthcare |
| Professional | Navy, gray, black | Clean lines, rectangles | Low-moderate | Finance, law, consulting |
| Elegant | Black, gold, white | Refined curves, balance | Moderate | Luxury, fashion, jewelry |
| Friendly | Warm colors, orange | Rounded shapes | Moderate | Education, social, community |
| Confident | Bold primary colors | Strong geometric | Moderate-high | Tech, sports, automotive |
| Energetic | Bright, saturated | Angular, dynamic | High | Fitness, energy drinks, sports |
| Playful | Rainbow, pastels | Organic, whimsical | High | Kids, games, creative |
| Intense | High contrast | Sharp angles | Very high | Gaming, extreme sports, nightlife |

---

## 5. LLM Integration Points

### 5.1 Integration Point 1: Initial Generation

**When**: Archive initialization, creating seed population
**LLM Task**: Generate SVG from text prompt
**Input**: Prompt + genome parameters
**Output**: SVG code
**Frequency**: 500-1000 calls (initialization phase)

**Example**:
```python
generator = LLMLogoGenerator()
prompt = "Create a minimalist tech logo with calm, professional tone using blue colors"
svg = generator.generate(prompt, behavioral_hints={'emotion': 'calm', 'complexity': 'low'})
```

### 5.2 Integration Point 2: Semantic Mutation

**When**: Every MAP-Elites iteration
**LLM Task**: Modify SVG toward behavioral target
**Input**: Source SVG + behavioral delta instructions
**Output**: Modified SVG
**Frequency**: 5,000-10,000 calls (main search loop)

**Example**:
```python
mutator = SemanticMutator()
current_behavior = (2, 3, 7, 1, 2)  # Simple, mixed, symmetric, duotone, professional
target_behavior = (5, 3, 7, 4, 6)   # More complex, same style/symmetry, more colors, energetic

mutated_svg = mutator.mutate_combined(
    source_svg=svg,
    current_behavior=current_behavior,
    target_behavior=target_behavior,
    genome=genome
)
```

### 5.3 Integration Point 3: Emotion Analysis

**When**: Every logo evaluation
**LLM Task**: Classify emotional tone
**Input**: SVG code
**Output**: (emotion_score, emotion_label)
**Frequency**: 5,000-10,000 calls (can be batched)

**Optimization**: Batch analysis
```python
analyzer = EmotionalToneAnalyzer()
emotions = analyzer.batch_analyze([svg1, svg2, svg3, ...])  # Analyze 10 at once
```

### 5.4 Integration Point 4: Query Parsing

**When**: User submits natural language query
**LLM Task**: Extract behavioral goals
**Input**: Natural language query
**Output**: Structured BehavioralGoals
**Frequency**: 1-10 calls per experiment (low frequency)

**Example**:
```python
parser = NaturalLanguageQueryParser()
goals = parser.parse_query("I need vibrant, playful logos for a kids' app")
# goals.emotion → Range(7, 8)  # Playful
# goals.color → Range(6, 9)    # Vibrant = high color richness
```

### 5.5 Integration Point 5: Fitness Evaluation (Optional)

**When**: Supplementary quality check
**LLM Task**: Aesthetic critique
**Input**: SVG code
**Output**: Textual critique + score
**Frequency**: Optional, for high-value candidates only

**Note**: Primary fitness uses existing `LogoValidator` (faster). LLM evaluation is supplementary.

---

## 6. Data Structures

### 6.1 Core Data Structures

```python
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

@dataclass
class Range:
    """Behavioral range for search goals"""
    min: int
    max: int

    def contains(self, value: int) -> bool:
        return self.min <= value <= self.max

@dataclass
class BehavioralGoals:
    """Target behavioral ranges for QD search"""
    complexity: Range
    style: Range
    symmetry: Range
    color: Range
    emotion: Range

    # Constraints
    company: Optional[str] = None
    industry: Optional[str] = None
    style_keywords: List[str] = field(default_factory=list)
    color_palette: List[str] = field(default_factory=list)

    def matches(self, behavior: Tuple[int, int, int, int, int]) -> bool:
        """Check if behavior tuple matches goals"""
        return (
            self.complexity.contains(behavior[0]) and
            self.style.contains(behavior[1]) and
            self.symmetry.contains(behavior[2]) and
            self.color.contains(behavior[3]) and
            self.emotion.contains(behavior[4])
        )

@dataclass
class Enhanced5DEntry:
    """Archive entry with 5D behavior"""
    logo_id: str
    svg_code: str
    genome: Dict
    fitness: float
    aesthetic_breakdown: Dict
    behavior_5d: Tuple[int, int, int, int, int]
    raw_behavior: Dict  # Includes 'emotional_tone'
    emotion_label: str  # Human-readable: "playful", "calm", etc.
    generation: int
    parent_ids: List[str] = field(default_factory=list)
    semantic_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize to dict for JSON export"""
        return {
            'logo_id': self.logo_id,
            'genome': self.genome,
            'fitness': self.fitness,
            'aesthetic_breakdown': self.aesthetic_breakdown,
            'behavior_5d': list(self.behavior_5d),
            'raw_behavior': self.raw_behavior,
            'emotion_label': self.emotion_label,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'semantic_tags': self.semantic_tags
        }

@dataclass
class SearchResult:
    """Results from LLM-guided QD search"""
    archive: 'EnhancedQDArchive'
    user_query: Optional[str]
    behavioral_goals: Optional[BehavioralGoals]
    statistics: Dict
    best_logos: List[Enhanced5DEntry]
    coverage_by_region: Dict

    def summary(self) -> str:
        """Generate human-readable summary"""
        return f"""
LLM-Guided QD Search Results
============================
Query: {self.user_query or 'General exploration'}
Coverage: {self.statistics['coverage']*100:.1f}%
Occupied Cells: {self.statistics['num_occupied']:,} / {self.statistics['total_cells']:,}
Avg Fitness: {self.statistics['avg_fitness']:.1f}/100
Max Fitness: {self.statistics['max_fitness']:.1f}/100
Best Logo: {self.best_logos[0].logo_id} (fitness={self.best_logos[0].fitness:.1f})
"""
```

### 6.2 Genome Structure (Extended)

```python
# Existing genome format (from evolutionary_logo_system.py)
genome = {
    'company': str,
    'industry': str,
    'style_keywords': List[str],
    'color_palette': List[str],
    'design_principles': List[str],
    'complexity_target': int,
    'golden_ratio_weight': float,
    'color_harmony_type': str
}

# NEW: Extended with emotion hints
genome_extended = {
    **genome,  # All existing fields
    'emotional_target': str,  # "calm", "energetic", "playful", etc.
    'emotional_intensity': float,  # 0-1 scale
    'behavioral_hints': Dict[str, str]  # Additional LLM guidance
}
```

---

## 7. API Interfaces

### 7.1 Main Search API

```python
# Primary interface for running LLM-guided QD search
from src.llm_qd.llm_guided_qd_search import LLMGuidedQDSearch

# Example 1: General exploration (no query)
search = LLMGuidedQDSearch(
    company_name="TechCorp",
    industry="technology"
)
archive = search.run_search(
    user_query=None,
    n_iterations=5000,
    n_initial=500
)

# Example 2: Directed search (with query)
archive = search.run_search(
    user_query="minimalist, calming logos with blue tones",
    n_iterations=3000,
    n_initial=300
)

# Access results
stats = archive.get_statistics()
best_logos = archive.get_best_logos(n=20)
calm_logos = archive.query_by_emotion(emotion_range=(0, 2))
```

### 7.2 Visualization API

```python
from src.llm_qd.diversity_visualizer import DiversityVisualizer

visualizer = DiversityVisualizer(archive)

# Generate interactive HTML grid
visualizer.generate_html_grid(
    output_path='output/llm_qd_grid.html',
    slice_dims=('complexity', 'emotion'),
    filter_criteria={'style': Range(0, 3)}  # Only geometric
)

# Export portfolio
visualizer.export_portfolio(
    logos=best_logos,
    output_dir='output/portfolio',
    format='gallery_html'
)

# Generate heatmap
visualizer.generate_behavior_heatmap(
    dimension1='complexity',
    dimension2='emotion',
    metric='fitness'
)
```

### 7.3 Query API

```python
from src.llm_qd.query_parser import NaturalLanguageQueryParser

parser = NaturalLanguageQueryParser()

# Parse query
goals = parser.parse_query("vibrant tech logos for startups")
print(goals.emotion)  # Range(6, 8) - energetic/vibrant
print(goals.color)    # Range(6, 9) - colorful
print(goals.industry) # "technology"
```

### 7.4 Emotion Analysis API

```python
from src.llm_qd.emotional_tone_analyzer import EmotionalToneAnalyzer

analyzer = EmotionalToneAnalyzer()

# Analyze single logo
emotion_score, emotion_label = analyzer.analyze_emotion(svg_code)
print(f"Emotion: {emotion_label} (score: {emotion_score:.2f})")

# Batch analysis (efficient)
emotions = analyzer.batch_analyze([svg1, svg2, svg3, svg4, svg5])
```

---

## 8. File Structure

### 8.1 New Directory Structure

```
/home/luis/svg-logo-ai/
├── src/
│   ├── llm_qd/                              # NEW: LLM-QD System
│   │   ├── __init__.py
│   │   ├── query_parser.py                  # Natural language query parsing
│   │   ├── llm_logo_generator.py            # LLM-powered SVG generation
│   │   ├── semantic_mutator.py              # Emotion-aware mutations
│   │   ├── enhanced_behavior_characterizer.py  # 5D behavioral features
│   │   ├── enhanced_qd_archive.py           # 5D MAP-Elites archive
│   │   ├── llm_guided_qd_search.py          # Main search orchestrator
│   │   ├── diversity_visualizer.py          # Interactive visualization
│   │   ├── emotional_tone_analyzer.py       # LLM emotion classification
│   │   └── emotion_taxonomy.json            # Emotion definitions
│   │
│   ├── evolutionary_logo_system.py          # EXISTING: Base evolutionary
│   ├── rag_experiment_runner.py             # EXISTING: RAG system
│   ├── map_elites_experiment.py             # EXISTING: Basic MAP-Elites
│   ├── behavior_characterization.py         # EXISTING: 4D behaviors
│   ├── llm_guided_mutation.py               # EXISTING: LLM mutations
│   ├── map_elites_archive.py                # EXISTING: Archive base
│   ├── logo_validator.py                    # EXISTING: Quality metrics
│   └── ...
│
├── experiments/
│   ├── llm_qd_experiment_YYYYMMDD_HHMMSS/  # NEW: LLM-QD experiments
│   │   ├── config.json
│   │   ├── archive.json
│   │   ├── statistics.json
│   │   ├── query_log.json
│   │   ├── logos/
│   │   │   ├── logo_0001.svg
│   │   │   ├── logo_0002.svg
│   │   │   └── ...
│   │   └── visualizations/
│   │       ├── grid_complexity_emotion.html
│   │       ├── heatmap_fitness.png
│   │       └── query_results.html
│   │
│   ├── experiment_YYYYMMDD_HHMMSS/         # EXISTING: Evolutionary
│   ├── map_elites_YYYYMMDD_HHMMSS/         # EXISTING: MAP-Elites
│   └── rag_experiment_YYYYMMDD_HHMMSS/     # EXISTING: RAG
│
├── chroma_db/
│   ├── llm_qd/                              # NEW: 5D archive ChromaDB
│   ├── logos/                               # EXISTING: RAG knowledge base
│   └── map_elites/                          # EXISTING: 4D archive
│
├── docs/
│   ├── LLM_QD_ARCHITECTURE.md               # NEW: This document
│   ├── LLM_QD_USER_GUIDE.md                 # NEW: User documentation
│   ├── LLM_QD_API_REFERENCE.md              # NEW: API documentation
│   ├── EVOLUTIONARY_PAPER_DRAFT.md          # EXISTING
│   ├── LOGO_DESIGN_PRINCIPLES.md            # EXISTING
│   └── ...
│
└── tests/
    ├── test_llm_qd/                         # NEW: LLM-QD tests
    │   ├── test_query_parser.py
    │   ├── test_emotion_analyzer.py
    │   ├── test_enhanced_characterizer.py
    │   └── test_llm_qd_search.py
    └── ...
```

### 8.2 Configuration Files

```python
# config/llm_qd_config.yaml
llm_qd_search:
  # Grid configuration
  grid_dimensions: [10, 10, 10, 10, 10]  # 5D grid

  # Search parameters
  n_initial_logos: 500
  n_iterations: 5000
  directed_exploration_ratio: 0.7  # 70% directed, 30% exploration

  # LLM configuration
  model_name: "gemini-2.5-flash"
  temperature: 0.7
  max_retries: 3
  batch_size: 10  # For batch emotion analysis

  # Quality thresholds
  min_fitness_for_archive: 70.0
  target_avg_fitness: 85.0

  # Behavioral biases
  emotion_weight: 1.0  # Weight emotion equally with other dimensions
  prefer_diversity: true

  # Caching
  cache_emotion_analysis: true
  cache_query_parses: true
  cache_ttl_hours: 24
```

### 8.3 Dependencies

```python
# requirements_llm_qd.txt (additions to existing requirements.txt)
google-generativeai>=0.3.0  # EXISTING (already used)
chromadb>=0.4.0             # EXISTING (already used)
numpy>=1.24.0               # EXISTING
matplotlib>=3.7.0           # For heatmaps
plotly>=5.18.0              # For interactive visualizations
jinja2>=3.1.0               # For HTML template rendering
pyyaml>=6.0                 # For config files
```

---

## 9. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Goal**: Implement 5D behavioral characterization and archive

**Tasks**:
1. Create `EmotionalToneAnalyzer` class
   - Implement emotion analysis prompt
   - Test on sample logos
   - Validate classification accuracy

2. Extend `BehaviorCharacterizer` to 5D
   - Add emotion dimension
   - Update bin discretization
   - Test on diverse logos

3. Extend `MAPElitesArchive` to 5D
   - Update dimensions to (10,10,10,10,10)
   - Add emotion queries
   - Update ChromaDB schema

4. Create test suite
   - Test emotion analysis
   - Test 5D characterization
   - Test archive operations

**Deliverable**: Working 5D behavioral characterization system

---

### Phase 2: LLM Integration (Week 3-4)

**Goal**: Implement LLM-guided generation and mutation

**Tasks**:
1. Create `NaturalLanguageQueryParser`
   - Design query parsing prompt
   - Implement goal extraction
   - Test on sample queries

2. Create `LLMLogoGenerator`
   - Extend existing generator
   - Add behavioral hints
   - Test generation quality

3. Create `SemanticMutator`
   - Implement emotion-aware mutations
   - Add combined 5D mutations
   - Test mutation effectiveness

4. Integration testing
   - End-to-end query → logo pipeline
   - Validate LLM outputs
   - Performance optimization

**Deliverable**: Full LLM-guided generation and mutation system

---

### Phase 3: QD Search (Week 5-6)

**Goal**: Implement LLM-guided MAP-Elites search

**Tasks**:
1. Create `LLMGuidedQDSearch` orchestrator
   - Implement initialization
   - Implement directed exploration
   - Implement undirected exploration

2. Search strategy optimization
   - Tune directed/exploration ratio
   - Optimize target cell selection
   - Balance quality vs diversity

3. Large-scale testing
   - Run 5,000 iteration searches
   - Validate coverage and quality
   - Compare against baselines

4. Performance optimization
   - Batch LLM API calls
   - Cache emotion analyses
   - Parallelize evaluations

**Deliverable**: Working LLM-guided QD search system

---

### Phase 4: Visualization & Export (Week 7)

**Goal**: Create interactive visualization and export tools

**Tasks**:
1. Create `DiversityVisualizer`
   - Implement HTML grid generator
   - Add interactive filters
   - Create heatmaps

2. Portfolio export
   - SVG gallery generation
   - PDF export
   - PNG rasterization

3. Query results page
   - Display matching logos
   - Show behavioral statistics
   - Enable downloads

4. User interface polish
   - Responsive design
   - Accessibility
   - Documentation

**Deliverable**: Complete visualization and export system

---

### Phase 5: Integration & Testing (Week 8)

**Goal**: Full system integration and validation

**Tasks**:
1. End-to-end testing
   - Multiple query types
   - Different company/industry combinations
   - Edge cases

2. Comparison experiments
   - LLM-QD vs baseline evolutionary
   - LLM-QD vs RAG
   - LLM-QD vs basic MAP-Elites

3. Documentation
   - User guide
   - API reference
   - Tutorial notebooks

4. Performance benchmarking
   - API call costs
   - Runtime analysis
   - Quality metrics

**Deliverable**: Production-ready LLM-QD system

---

## 10. Experiment Protocol

### 10.1 Test Queries

**Category 1: Emotional Tone**
1. "Calm, serene logos for meditation app"
2. "Energetic, vibrant logos for fitness brand"
3. "Professional, corporate logos for consulting firm"
4. "Playful, whimsical logos for kids' toys"

**Category 2: Behavioral Constraints**
1. "Minimalist logos with high symmetry"
2. "Complex, asymmetric tech logos"
3. "Geometric monochrome designs"
4. "Organic, colorful nature logos"

**Category 3: Combined Criteria**
1. "Bold, confident tech logos with geometric style"
2. "Elegant, refined fashion logos with minimal colors"
3. "Friendly, approachable education logos with warm tones"
4. "Intense, aggressive gaming logos with high complexity"

**Category 4: Open-Ended**
1. "Innovative startup logos"
2. "Luxury brand identity"
3. "Eco-friendly product logos"
4. "Modern minimalist design"

### 10.2 Success Metrics

**Primary Metrics**:

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Archive Coverage** | > 10% | Percentage of 100,000 cells occupied |
| **Average Fitness** | > 85/100 | Mean fitness across all archived logos |
| **Query Precision** | > 80% | % of results matching query intent |
| **Emotional Accuracy** | > 75% | Human validation of emotion labels |
| **Diversity Score** | > 0.8 | Behavioral distance between logos |

**Secondary Metrics**:

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Generation Quality** | > 90/100 | Best logo fitness |
| **Coverage Speed** | < 3000 iter | Iterations to reach 5% coverage |
| **API Efficiency** | < 20k calls | Total LLM API calls |
| **Runtime** | < 6 hours | Time for 5000 iterations |

### 10.3 Baseline Comparisons

**Comparison 1: vs. Baseline Evolutionary**
- Metric: Final average fitness
- Expectation: LLM-QD ≥ Evolutionary (both ~85-90/100)
- Advantage: LLM-QD provides diversity, evolutionary provides optimization

**Comparison 2: vs. RAG-Enhanced**
- Metric: Quality + Diversity
- Expectation: LLM-QD > RAG in diversity, similar quality
- Advantage: LLM-QD explores entire space, RAG focuses on known good regions

**Comparison 3: vs. Basic MAP-Elites**
- Metric: Coverage + Quality
- Expectation: LLM-QD > MAP-Elites in both
- Advantage: LLM semantic understanding guides better mutations

**Comparison Matrix**:

|  | Baseline Evo | RAG | MAP-Elites (4D) | **LLM-QD (5D)** |
|---|---|---|---|---|
| **Avg Fitness** | 85-90 | 90-95 | 80-85 | **85-90** |
| **Diversity** | Low | Medium | High | **Very High** |
| **Coverage** | N/A | N/A | 5-10% | **10-15%** |
| **Query Control** | No | No | No | **Yes** |
| **Emotion Aware** | No | No | No | **Yes** |
| **Runtime** | 2h | 3h | 4h | **6h** |

### 10.4 Validation Protocol

**Step 1: Automated Validation**
- Run 10 test queries
- Generate 3,000-5,000 logos per query
- Measure coverage, fitness, precision

**Step 2: Human Validation** (Sample-based)
- Select 100 random logos from archive
- Human judges rate:
  - Emotion accuracy (does "playful" label match perception?)
  - Quality (professional usability)
  - Diversity (how distinct are logos in same query?)

**Step 3: Ablation Studies**
- Run LLM-QD without emotion dimension (4D) → measure impact
- Run LLM-QD without directed search → measure query precision loss
- Run LLM-QD without semantic mutations → measure quality loss

**Step 4: Stress Testing**
- Ambiguous queries: "modern logos"
- Conflicting constraints: "minimalist but highly complex"
- Edge cases: "logos with no colors" (invalid)

### 10.5 Expected Outcomes

**Quantitative Outcomes**:
1. **Archive Size**: 10,000-15,000 unique logos
2. **Coverage**: 10-15% of 100,000 cells (10,000-15,000 cells)
3. **Average Fitness**: 85-90/100
4. **Query Precision**: 80-85% match rate
5. **Runtime**: ~6 hours for full experiment

**Qualitative Outcomes**:
1. **Diverse Portfolio**: Logos span calm→intense, simple→complex
2. **Query Control**: Users can explore specific emotional tones
3. **Professional Quality**: 80%+ of logos suitable for real use
4. **Novel Designs**: Non-obvious combinations discovered by QD

**Key Innovations Demonstrated**:
1. **First LLM-QD Logo System**: Combines semantic intelligence with QD
2. **Emotion-Aware Design Space**: 5D behavioral characterization
3. **Natural Language Control**: Conversational design exploration
4. **Systematic Coverage**: Guaranteed diversity through QD algorithm

---

## Appendix A: Example Workflows

### Workflow 1: General Exploration

```python
# User wants to explore all possible tech logos
from src.llm_qd.llm_guided_qd_search import LLMGuidedQDSearch
from src.llm_qd.diversity_visualizer import DiversityVisualizer

# Initialize search
search = LLMGuidedQDSearch(
    company_name="TechVentures",
    industry="technology"
)

# Run undirected search (no query)
archive = search.run_search(
    user_query=None,
    n_iterations=5000,
    n_initial=500
)

# Visualize results
viz = DiversityVisualizer(archive)
viz.generate_html_grid(
    output_path='output/tech_exploration.html',
    slice_dims=('complexity', 'emotion')
)

print(f"Explored {archive.get_statistics()['num_occupied']} unique design niches")
```

### Workflow 2: Directed Search with Query

```python
# User wants specific type of logos
search = LLMGuidedQDSearch(
    company_name="ZenLife",
    industry="wellness"
)

archive = search.run_search(
    user_query="calm, minimalist logos with soft blue and green tones",
    n_iterations=3000,
    n_initial=300
)

# Filter results matching query
from src.llm_qd.query_parser import NaturalLanguageQueryParser
parser = NaturalLanguageQueryParser()
goals = parser.parse_query("calm, minimalist logos with soft blue and green tones")

matching_logos = [
    entry for entry in archive.archive.values()
    if goals.matches(entry.behavior_5d)
]

print(f"Found {len(matching_logos)} logos matching your criteria")

# Export portfolio
viz = DiversityVisualizer(archive)
viz.export_portfolio(
    logos=matching_logos,
    output_dir='output/zenlife_portfolio',
    format='gallery_html'
)
```

### Workflow 3: Interactive Exploration

```python
# User explores interactively via HTML interface
viz = DiversityVisualizer(archive)

# Generate multiple views
viz.generate_html_grid(
    'output/view_complexity_emotion.html',
    slice_dims=('complexity', 'emotion')
)

viz.generate_html_grid(
    'output/view_style_color.html',
    slice_dims=('style', 'color_richness')
)

# Filter for specific region
viz.generate_html_grid(
    'output/view_geometric_calm.html',
    slice_dims=('complexity', 'symmetry'),
    filter_criteria={
        'style': Range(0, 3),  # Geometric
        'emotion': Range(0, 2)  # Calm
    }
)

# User browses HTML grids, clicks logos, downloads favorites
```

---

## Appendix B: Performance Optimization

### B.1 LLM API Call Optimization

**Problem**: 5,000 iterations × 3 LLM calls/iteration = 15,000 API calls

**Solutions**:

1. **Batch Emotion Analysis**
   ```python
   # Instead of: analyze_emotion(svg) per logo
   # Do: batch_analyze([svg1, svg2, ..., svg10])
   # Reduces calls by 10x
   ```

2. **Cache Emotion Results**
   ```python
   # Cache emotion by SVG hash
   @lru_cache(maxsize=10000)
   def get_emotion(svg_hash: str) -> Tuple[float, str]:
       return emotion_analyzer.analyze_emotion(svg)
   ```

3. **Lazy Emotion Analysis**
   ```python
   # Only analyze emotion when:
   # - Logo is candidate for archive (passes fitness threshold)
   # - Archive cell is empty or current logo is better
   # Reduces unnecessary analyses by ~70%
   ```

4. **Use Cheaper Models for Bulk Operations**
   ```python
   # Generation/mutation: Gemini 2.5 Flash (fast, cheap)
   # Critical evaluation: Gemini 2.5 Pro (high quality, expensive)
   ```

### B.2 Parallelization

```python
# Parallelize logo generation during initialization
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [
        executor.submit(generate_logo, genome)
        for genome in initial_genomes
    ]
    logos = [f.result() for f in futures]
```

### B.3 ChromaDB Optimization

```python
# Batch upserts instead of individual adds
entries_to_add = []
for logo in new_logos:
    entries_to_add.append(create_entry(logo))
    if len(entries_to_add) >= 100:
        collection.upsert(entries_to_add)
        entries_to_add = []
```

---

## Appendix C: Error Handling

### C.1 LLM API Failures

```python
def generate_with_fallback(prompt: str, max_retries: int = 3) -> str:
    """Generate with exponential backoff and fallback"""
    for attempt in range(max_retries):
        try:
            return llm_generator.generate(prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                # Fallback to simpler generation
                return fallback_generator.generate(prompt)
            time.sleep(2 ** attempt)  # Exponential backoff
```

### C.2 Invalid SVG Handling

```python
def validate_and_fix_svg(svg_code: str) -> Optional[str]:
    """Validate SVG and attempt automatic fixes"""
    try:
        ET.fromstring(svg_code)
        return svg_code
    except ET.ParseError:
        # Attempt fixes
        fixed_svg = add_missing_namespace(svg_code)
        fixed_svg = close_unclosed_tags(fixed_svg)
        try:
            ET.fromstring(fixed_svg)
            return fixed_svg
        except:
            return None  # Unfixable, discard
```

### C.3 Emotion Analysis Failures

```python
def analyze_emotion_safe(svg_code: str) -> Tuple[float, str]:
    """Emotion analysis with fallback to heuristics"""
    try:
        return emotion_analyzer.analyze_emotion(svg_code)
    except Exception as e:
        # Fallback to algorithmic heuristics
        return heuristic_emotion_estimate(svg_code)

def heuristic_emotion_estimate(svg_code: str) -> Tuple[float, str]:
    """Estimate emotion from color saturation and complexity"""
    colors = extract_colors(svg_code)
    saturation_avg = np.mean([rgb_to_hsv(*c)[1] for c in colors])
    complexity = count_elements(svg_code)

    # High saturation + high complexity → energetic
    # Low saturation + low complexity → calm
    emotion_score = (saturation_avg + complexity / 100) / 2
    return emotion_score, classify_emotion(emotion_score)
```

---

## Appendix D: Future Enhancements

### D.1 Multi-Modal Emotion Analysis

**Current**: Text-based LLM analysis of SVG code
**Future**: Vision-based analysis using rendered images

```python
class VisionBasedEmotionAnalyzer:
    """Use Gemini vision capabilities on rendered SVGs"""

    def analyze_emotion(self, svg_code: str) -> Tuple[float, str]:
        # Render SVG to PNG
        png_image = render_svg_to_png(svg_code)

        # Use Gemini vision model
        response = vision_model.generate_content([
            "Analyze the emotional tone of this logo",
            png_image
        ])

        return parse_emotion(response.text)
```

### D.2 User Feedback Loop

**Concept**: Learn from user preferences

```python
class AdaptiveSearch:
    """Adjust search based on user feedback"""

    def incorporate_feedback(self,
                            liked_logos: List[str],
                            disliked_logos: List[str]):
        # Extract common behavioral patterns from liked logos
        liked_behaviors = [
            self.archive.get_entry(logo_id).behavior_5d
            for logo_id in liked_logos
        ]

        # Update search goals to bias toward liked region
        self.adaptive_goals = infer_goals_from_examples(liked_behaviors)

        # Continue search with updated goals
        self.run_search(behavioral_goals=self.adaptive_goals)
```

### D.3 Style Transfer

**Concept**: "Make this logo more [emotion]"

```python
class StyleTransferMutator:
    """Transfer emotional style from reference logo"""

    def transfer_emotion(self,
                        source_svg: str,
                        reference_svg: str) -> str:
        # Analyze reference emotion
        ref_emotion = self.emotion_analyzer.analyze_emotion(reference_svg)

        # Extract stylistic features from reference
        ref_features = extract_style_features(reference_svg)

        # Apply features to source
        return self.llm_mutator.mutate_with_style(
            source_svg,
            target_emotion=ref_emotion,
            style_hints=ref_features
        )
```

### D.4 Ensemble LLM Evaluation

**Concept**: Multiple LLM judges for more robust evaluation

```python
class EnsembleEvaluator:
    """Use multiple LLMs to evaluate logos"""

    def __init__(self):
        self.judges = [
            genai.GenerativeModel('gemini-2.5-flash'),
            genai.GenerativeModel('gemini-2.5-pro'),
            # Could add other models (Claude, GPT-4, etc.)
        ]

    def evaluate_consensus(self, svg_code: str) -> Dict:
        scores = []
        for judge in self.judges:
            score = judge.evaluate(svg_code)
            scores.append(score)

        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'agreement': calculate_agreement(scores)
        }
```

---

## Conclusion

This architecture document specifies a complete LLM-Guided Quality-Diversity Logo System that:

1. **Extends existing systems** with 5D behavioral space (adding emotion)
2. **Integrates LLM intelligence** at generation, mutation, and evaluation
3. **Enables natural language control** via query parsing
4. **Ensures systematic diversity** through MAP-Elites algorithm
5. **Maintains high quality** via existing LogoValidator metrics

**Key Innovations**:
- First logo system combining LLM + QD
- Emotion-aware behavioral characterization
- Natural language design space exploration
- Guaranteed diversity with quality control

**Next Steps**:
1. Review and approve architecture
2. Begin Phase 1 implementation (5D characterization)
3. Iterate based on experimental results

**Questions for Review**:
1. Are 5 behavioral dimensions sufficient, or should we add more?
2. Should emotion be LLM-only or hybrid (LLM + heuristics)?
3. What grid size is optimal? (Current: 10^5 cells, could reduce to 5^5 = 3,125)
4. Should we support multi-objective optimization (Pareto fronts)?

---

**End of Architecture Document**
