# LLM Components Quick Start Guide

**Fast setup guide for using LLM-powered logo generation components**

---

## 1. Setup (1 minute)

### Install Dependencies
Already installed in your environment:
```bash
# Activate virtual environment
source venv/bin/activate

# Dependencies already in requirements.txt:
# - google-generativeai (Gemini API)
# - chromadb (for RAG)
# - numpy, pandas
```

### Set API Key
```bash
# Set your Google Gemini API key
export GOOGLE_API_KEY=your_gemini_api_key_here

# Verify it's set
echo $GOOGLE_API_KEY
```

---

## 2. Quick Examples

### Example 1: Generate Logos from Natural Language (30 seconds)

```python
from src.nl_query_parser import NLQueryParser
from src.llm_logo_generator import LLMLogoGenerator

# Parse natural language query
parser = NLQueryParser()
parsed = parser.parse_query("10 minimalist tech logos in blue tones")

# Generate logos
generator = LLMLogoGenerator()
variations = generator.generate_from_prompt(
    user_query="minimalist tech logos in blue tones",
    num_variations=10
)

# Save results
generator.save_variations(variations, "output/my_logos", prefix="tech_logo")

print(f"Generated {len(variations)} logos!")
print(f"Saved to: output/my_logos")
```

**Output:**
- `output/my_logos/tech_logo_001.svg` through `tech_logo_010.svg`
- `output/my_logos/tech_logo_summary.json` with metadata

### Example 2: Targeted Logo Generation (20 seconds)

```python
from src.llm_logo_generator import LLMLogoGenerator

generator = LLMLogoGenerator()

# Generate logo with specific behavioral characteristics
logo = generator.generate_targeted(
    base_prompt="Professional logo for DataFlow - analytics company",
    behavioral_target={
        'complexity': 0.6,   # Moderate complexity
        'style': 0.3,        # Geometric
        'symmetry': 0.8,     # Highly symmetric
        'color_richness': 0.25  # Duotone
    }
)

# Save
with open("output/dataflow_logo.svg", 'w') as f:
    f.write(logo.svg_code)

print(f"Design: {logo.design_rationale}")
print(f"Estimated fitness: {logo.estimated_fitness}/100")
```

### Example 3: Semantic Mutation (15 seconds)

```python
from src.semantic_mutator import SemanticMutator

mutator = SemanticMutator()

# Load your current logo
with open("my_logo.svg") as f:
    current_svg = f.read()

# Mutate it toward specific goals
mutated = mutator.mutate_toward_behavior(
    logo_svg=current_svg,
    current_behavior={'complexity': 0.2, 'style': 0.1, 'symmetry': 1.0, 'color_richness': 0.0},
    target_behavior={'complexity': 0.6, 'style': 0.4, 'symmetry': 0.8, 'color_richness': 0.5},
    user_intent="Modern tech company logo"
)

# Save mutated version
with open("my_logo_evolved.svg", 'w') as f:
    f.write(mutated)
```

### Example 4: Evaluate Logo Quality (10 seconds)

```python
from src.llm_evaluator import LLMLogoEvaluator

evaluator = LLMLogoEvaluator()

# Load logo to evaluate
with open("my_logo.svg") as f:
    svg_code = f.read()

# Get multi-dimensional evaluation
scores = evaluator.evaluate_fitness(
    logo_svg=svg_code,
    user_query="Professional tech company logo"
)

print(f"Aesthetic: {scores['aesthetic']}/100")
print(f"Professionalism: {scores['professionalism']}/100")
print(f"Originality: {scores['originality']}/100")
print(f"OVERALL: {scores['overall']}/100")

# Get detailed critique
critique = evaluator.critique_and_suggest(svg_code, "Tech company logo")
print("\nStrengths:")
for s in critique['strengths']:
    print(f"  + {s}")
print("\nSuggestions:")
for s in critique['suggestions']:
    print(f"  → {s}")
```

---

## 3. Complete Workflow Example

**Goal:** Generate 50 logos, evaluate them, pick best 10

```python
from src.nl_query_parser import NLQueryParser
from src.llm_logo_generator import LLMLogoGenerator
from src.llm_evaluator import LLMLogoEvaluator

# 1. Parse user query
parser = NLQueryParser()
parsed = parser.parse_query("50 modern healthcare logos with leaf motifs")

print(f"Generating {parsed.quantity} logos...")
print(f"Style: {', '.join(parsed.style_keywords)}")
print(f"Emotion: {parsed.emotion_target}")

# 2. Generate logos
generator = LLMLogoGenerator()
variations = generator.generate_from_prompt(
    user_query=parsed.original_query,
    num_variations=parsed.quantity
)

print(f"Generated {len(variations)} variations")

# 3. Evaluate each logo
evaluator = LLMLogoEvaluator()
evaluated = []

for i, var in enumerate(variations, 1):
    print(f"Evaluating {i}/{len(variations)}...")
    scores = evaluator.evaluate_fitness(var.svg_code, parsed.original_query)
    evaluated.append({
        'variation': var,
        'scores': scores
    })

# 4. Sort by quality
evaluated.sort(key=lambda x: x['scores']['overall'], reverse=True)

# 5. Save top 10
best_10 = [e['variation'] for e in evaluated[:10]]
generator.save_variations(best_10, "output/best_healthcare_logos", prefix="best")

print(f"\nTop 10 logos saved!")
print(f"Best score: {evaluated[0]['scores']['overall']:.1f}/100")
print(f"Worst of top 10: {evaluated[9]['scores']['overall']:.1f}/100")
```

---

## 4. Integration with QD System

### Use with MAP-Elites

```python
from src.llm_logo_generator import LLMLogoGenerator
from src.behavior_characterization import BehaviorCharacterizer
from src.map_elites_archive import MAPElitesArchive

# Initialize
generator = LLMLogoGenerator()
characterizer = BehaviorCharacterizer()
archive = MAPElitesArchive(bins_per_dimension=10)

# Fill archive with targeted generation
for complexity in [0.2, 0.4, 0.6, 0.8]:
    for style in [0.2, 0.5, 0.8]:
        for symmetry in [0.3, 0.7]:
            # Generate logo for this behavioral region
            logo = generator.generate_targeted(
                base_prompt="Modern tech logo",
                behavioral_target={
                    'complexity': complexity,
                    'style': style,
                    'symmetry': symmetry,
                    'color_richness': 0.5
                }
            )

            # Characterize and add to archive
            behavior = characterizer.characterize(logo.svg_code)
            archive.add(logo.svg_code, behavior['bins'], logo.estimated_fitness)

print(f"Archive coverage: {archive.coverage():.1%}")
```

### Use Semantic Mutations in Evolution

```python
from src.semantic_mutator import SemanticMutator
from src.evolutionary_logo_system import EvolutionaryLogoSystem

# Replace random mutation with semantic mutation
class SemanticEvolutionarySystem(EvolutionaryLogoSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.semantic_mutator = SemanticMutator()

    def mutate(self, genome):
        # Get current SVG
        current_svg = self.genome_to_svg(genome)
        current_behavior = self.get_behavior(current_svg)

        # Random target behavior (could be smarter)
        target_behavior = self.random_behavior_nearby(current_behavior)

        # Semantic mutation
        mutated_svg = self.semantic_mutator.mutate_toward_behavior(
            logo_svg=current_svg,
            current_behavior=current_behavior,
            target_behavior=target_behavior,
            user_intent=self.design_intent
        )

        return self.svg_to_genome(mutated_svg)
```

---

## 5. Tips & Best Practices

### For Best Quality

1. **Be Specific in Queries:**
   - ❌ "logo"
   - ✓ "minimalist tech logo with circular motifs in blue tones"

2. **Use Behavioral Targets:**
   - Better control over output characteristics
   - Fill specific gaps in MAP-Elites archive

3. **Iterate with Mutations:**
   - Start with good base logo
   - Use semantic mutations to refine
   - Faster than generating from scratch

4. **Combine LLM + Objective Metrics:**
   - LLM for aesthetic quality
   - LogoValidator for technical quality
   - Best of both worlds

### For Performance

1. **Batch Generation:**
   - Generate multiple logos in one call
   - More efficient than serial calls

2. **Cache Results:**
   - Reuse successful logos
   - Build knowledge base (ChromaDB)

3. **Adjust Retries:**
   - Reduce retries for fast iteration (1-2)
   - Increase retries for final generation (3-5)

### For Cost Optimization

1. **Use Query Parser First:**
   - No API cost
   - Validates query before generation

2. **Generate Fewer Variations:**
   - 10-20 variations usually enough
   - Use mutations to explore from there

3. **Evaluate Selectively:**
   - Pre-filter with objective metrics
   - Only LLM-evaluate promising candidates

---

## 6. Troubleshooting

### API Key Issues
```bash
# Check if key is set
echo $GOOGLE_API_KEY

# Set it if missing
export GOOGLE_API_KEY=your_key_here

# Verify in Python
python -c "import os; print(os.getenv('GOOGLE_API_KEY'))"
```

### Import Errors
```bash
# Make sure you're in project root
cd /home/luis/svg-logo-ai

# Add src to path in your script
import sys
sys.path.insert(0, 'src')
```

### Rate Limiting
```python
# Add delays between calls
import time

for i in range(10):
    logo = generator.generate_from_prompt(query)
    time.sleep(2)  # 2 second delay
```

### Low Quality Output
```python
# Try targeted generation instead
logo = generator.generate_targeted(
    base_prompt=detailed_description,
    behavioral_target=specific_targets
)

# Or add more context to query
query = "minimalist tech logo for AI startup focusing on healthcare data analytics, clean and trustworthy"
```

---

## 7. Running Tests

```bash
# Set API key first
export GOOGLE_API_KEY=your_key_here

# Run all tests
python src/test_llm_components.py

# Or just query parser (no API needed)
python src/nl_query_parser.py

# Individual component demos
python src/llm_logo_generator.py  # Demo logo generation
python src/semantic_mutator.py     # Demo mutations
python src/llm_evaluator.py        # Demo evaluation
```

---

## 8. File Locations

**Source Code:**
- `src/llm_logo_generator.py` - Logo generation
- `src/semantic_mutator.py` - Intelligent mutations
- `src/llm_evaluator.py` - Quality evaluation
- `src/nl_query_parser.py` - Query parsing
- `src/test_llm_components.py` - Test suite

**Documentation:**
- `LLM_COMPONENTS_IMPLEMENTATION_REPORT.md` - Full technical documentation
- `LLM_COMPONENTS_QUICKSTART.md` - This file

**Outputs:**
- `output/llm_tests/` - Test results
- `output/my_logos/` - Your generated logos

---

## 9. Next Steps

1. **Run your first generation:**
   ```bash
   export GOOGLE_API_KEY=your_key
   python -c "from src.llm_logo_generator import LLMLogoGenerator; \
              g = LLMLogoGenerator(); \
              v = g.generate_from_prompt('tech logo', 5); \
              g.save_variations(v, 'output/test', 'logo'); \
              print('Done!')"
   ```

2. **Integrate with your workflow:**
   - Replace manual logo creation
   - Use in evolutionary experiments
   - Build interactive tools

3. **Experiment and iterate:**
   - Try different queries
   - Explore semantic mutations
   - Combine with existing tools

---

**Ready to generate amazing logos with AI!**

For detailed documentation, see `LLM_COMPONENTS_IMPLEMENTATION_REPORT.md`
