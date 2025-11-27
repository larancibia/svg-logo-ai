# Examples Directory

Code examples demonstrating how to use the LLM-QD Logo System.

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API key
export GOOGLE_API_KEY="your-gemini-api-key"

# Initialize knowledge base (for RAG examples)
python src/initialize_rag_kb.py
```

### 2. Run Examples

```bash
# Basic example (simplest)
python examples/example_basic.py

# Advanced example (all features)
python examples/example_advanced.py

# Natural language queries
python examples/example_custom_query.py
python examples/example_custom_query.py --interactive
```

## Available Examples

### 1. Basic Example (`example_basic.py`)

**Time: 2-3 minutes**

The simplest possible usage - generates a few logos with Quality-Diversity.

```bash
python examples/example_basic.py
```

**What it demonstrates:**
- Minimal system initialization
- Running a QD experiment
- Viewing results
- Basic output interpretation

**Perfect for:**
- First-time users
- Quick testing
- Understanding core concepts

**Output:**
- 10-20 logos
- Basic metrics (coverage, fitness)
- Simple results summary

---

### 2. Advanced Example (`example_advanced.py`)

**Time: 15-20 minutes**

Comprehensive demonstration of all system features.

```bash
python examples/example_advanced.py
```

**What it demonstrates:**
1. **Basic QD**: Default Quality-Diversity
2. **Curiosity-Driven Search**: Exploration strategy
3. **Novelty Search**: Maximum diversity
4. **RAG Enhancement**: Higher quality logos
5. **Custom Configuration**: Targeted constraints
6. **Advanced Visualization**: All plot types
7. **Strategy Comparison**: Side-by-side comparison
8. **Natural Language**: Query-based generation

**Perfect for:**
- Learning all features
- Understanding different strategies
- Comparing approaches
- Production preparation

**Output:**
- 100+ logos across experiments
- Detailed visualizations
- Comparative analysis
- Performance benchmarks

---

### 3. Custom Query Example (`example_custom_query.py`)

**Time: Variable (2-10 minutes per query)**

Natural language interface for logo generation.

```bash
# Predefined demos
python examples/example_custom_query.py

# Interactive mode
python examples/example_custom_query.py --interactive

# Specific demo
python examples/example_custom_query.py --demo variations
```

**What it demonstrates:**
- Natural language query parsing
- Behavioral constraint extraction
- Interactive query builder
- Style variations
- Color preferences

**Predefined Queries:**
1. Geometric, symmetric tech logo
2. Organic, flowing coffee shop logo
3. Minimal, iconic finance logo
4. Complex, detailed creative agency logo
5. Simple, clean health app logo

**Perfect for:**
- User-friendly interface
- Non-technical users
- Exploring design space
- Custom requirements

**Interactive Commands:**
- Enter natural language queries
- Specify company name
- See parsed constraints
- Generate matching logos

---

## Example Outputs

### Basic Example Output

```
===========================================================
LLM-QD Logo System - Basic Example
===========================================================

Initializing LLM-QD system...
Running Quality-Diversity experiment...
This will take about 2-3 minutes...

===========================================================
Results Summary
===========================================================

Logos Generated: 18
Coverage: 14.4%
Best Fitness: 87.3/100
Average Fitness: 83.5/100
QD-Score: 1502.3

Top 3 Logos:
----------------------------------------------------------
1. Fitness: 87.3
   Behavior: {'complexity': 0.45, 'style': 0.32, ...}
   File: experiments/llm_qd_*/logos/gen5_*.svg

...
```

### Advanced Example Output

```
===========================================================
Example 1: Basic Quality-Diversity
===========================================================

Generated 25 logos
Coverage: 16.0%
Best fitness: 88.5

===========================================================
Example 2: Curiosity-Driven Exploration
===========================================================
Focuses on under-explored regions of behavioral space

Coverage: 22.5%
This should show higher coverage than random search

===========================================================
Strategy Comparison Results:
----------------------------------------------------------
Strategy        Coverage     Best Fitness    QD-Score
----------------------------------------------------------
Random           16.0%          88.5         1420.3
Curiosity        22.5%          86.2         1938.7
Novelty          28.3%          84.1         2375.9
```

### Interactive Query Example

```
===========================================================
Natural Language Query - Interactive Mode
===========================================================

Enter natural language queries to generate logos.
Type 'quit' or 'exit' to stop.

----------------------------------------------------------------------

Enter your query: Create a geometric, minimal logo with blue colors
Company name (default: 'CustomCo'): TechStartup

Parsing query...

Parsed Constraints:
  style_range: (0.0, 0.3)
  complexity_range: (10, 25)
  color_palette: ['#0066cc', '#004999']

Generate logo with these constraints? (Y/n): y

Generating logo (this takes 2-3 minutes)...

===========================================================
Logo Generated!
===========================================================

Fitness: 85.7/100
File: experiments/llm_qd_*/logos/gen3_*.svg

Behavioral Characteristics:
  complexity: 0.18
  style: 0.12
  symmetry: 0.78
  color_richness: 0.32
```

## Understanding the Examples

### System Initialization

All examples follow this pattern:

```python
from llm_qd_logo_system import LLMQDLogoSystem

system = LLMQDLogoSystem(
    company_name="YourCompany",      # Logo target
    archive_dimensions=(10,10,10,10),# Grid size
    num_iterations=20,               # QD iterations
    batch_size=10                    # Logos per iteration
)
```

### Running Experiments

```python
results = system.run_experiment()

# Results dictionary contains:
# - archive: All generated logos
# - coverage: % of behavioral space filled
# - best_fitness: Highest quality logo
# - avg_fitness: Average quality
# - qd_score: Coverage × Quality
# - top_logos: Best logos sorted
# - output_dir: Where files are saved
```

### Customization Options

```python
system = LLMQDLogoSystem(
    # Basic settings
    company_name="TechCorp",
    archive_dimensions=(10, 10, 10, 10),
    num_iterations=20,

    # Behavioral ranges (0-1)
    complexity_range=(0.2, 0.8),      # Medium complexity
    style_range=(0.0, 0.5),           # Geometric bias
    symmetry_range=(0.5, 1.0),        # Prefer symmetric
    color_range=(0.3, 0.7),           # Moderate colors

    # Quality constraints
    min_fitness=75,                   # Minimum acceptable quality

    # Search strategy
    search_strategy=CuriosityDrivenSearch(),

    # Mutation settings
    mutation_rate=0.2,
    semantic_mutation_prob=0.6,

    # Output settings
    save_intermediate=True,
    verbose=True
)
```

## Common Patterns

### Pattern 1: Quick Test

```python
# Fast test with small archive
system = LLMQDLogoSystem(
    company_name="Test",
    archive_dimensions=(5, 5, 5, 5),  # 625 cells
    num_iterations=5,
    batch_size=5
)
```

### Pattern 2: Full Experiment

```python
# Comprehensive exploration
system = LLMQDLogoSystem(
    company_name="Production",
    archive_dimensions=(10, 10, 10, 10),  # 10k cells
    num_iterations=50,
    batch_size=20
)
```

### Pattern 3: High Quality Focus

```python
# Prioritize quality over diversity
system = LLMQDLogoSystem(
    company_name="Premium",
    num_iterations=30,
    min_fitness=85,                    # High bar
    search_strategy=None,              # Random (quality-focused)
    use_rag=True                       # Enable RAG
)
```

### Pattern 4: Maximum Diversity

```python
# Maximize behavioral coverage
system = LLMQDLogoSystem(
    company_name="Diverse",
    num_iterations=50,
    search_strategy=NoveltySearch(),   # Novelty search
    mutation_rate=0.4,                 # High mutation
    min_fitness=70                     # Lower bar
)
```

## Troubleshooting

### Issue: "ModuleNotFoundError"

```bash
# Make sure you're in the project root
cd /home/luis/svg-logo-ai

# Run with python path
PYTHONPATH=src python examples/example_basic.py
```

### Issue: API Rate Limits

```python
# Add delays between calls
system = LLMQDLogoSystem(
    ...
    api_delay=2.0  # 2 seconds between calls
)
```

### Issue: Low Fitness Scores

```python
# Enable RAG for better quality
from rag_evolutionary_system import RAGEvolutionarySystem

system = RAGEvolutionarySystem(
    company_name="HighQuality",
    use_rag=True,
    top_k_examples=3
)
```

### Issue: Out of Memory

```python
# Reduce archive size
system = LLMQDLogoSystem(
    ...
    archive_dimensions=(5, 5, 5, 5),  # Smaller
    batch_size=5                      # Fewer at once
)
```

## Performance Benchmarks

### Example Timing (on typical hardware)

| Example | Iterations | Logos | Duration | API Calls |
|---------|-----------|-------|----------|-----------|
| Basic | 10 | 15-20 | 2-3 min | 20-30 |
| Advanced (full) | Mixed | 100+ | 15-20 min | 150-200 |
| Custom Query | 5 | 8-12 | 2 min | 15-20 |
| Interactive (per query) | 5 | 8-12 | 2 min | 15-20 |

*Assumes gemini-1.5-flash with good API latency*

## Next Steps

After running examples:

1. **Explore Results:**
   - Browse generated SVG files
   - View heatmap visualizations
   - Read experiment reports

2. **Customize:**
   - Modify parameters
   - Try different strategies
   - Add your own queries

3. **Integrate:**
   - Use in your own code
   - Build web interfaces
   - Automate workflows

4. **Learn More:**
   - Read [User Guide](../docs/LLM_QD_USER_GUIDE.md)
   - Study [Architecture](../docs/LLM_QD_ARCHITECTURE.md)
   - Review [Paper Draft](../docs/EVOLUTIONARY_PAPER_DRAFT.md)

## Contributing Examples

Want to add more examples? See [CONTRIBUTING.md](../CONTRIBUTING.md)

Good example contributions:
- Industry-specific use cases
- Integration with other tools
- Advanced visualization techniques
- Novel search strategies
- Real-world applications

## Getting Help

- **Documentation:** See [docs/INDEX.md](../docs/INDEX.md)
- **Issues:** GitHub Issues
- **Questions:** GitHub Discussions

## License

All examples are licensed under MIT License. See [LICENSE](../LICENSE).
