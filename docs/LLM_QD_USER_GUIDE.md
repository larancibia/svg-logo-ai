# LLM-Guided Quality-Diversity Logo System: User Guide

**Version:** 1.0
**Date:** November 27, 2025

---

## Table of Contents

1. [Introduction](#introduction)
2. [What Makes This Revolutionary](#what-makes-this-revolutionary)
3. [Quick Start](#quick-start)
4. [How It Works](#how-it-works)
5. [Using the System](#using-the-system)
6. [Understanding Results](#understanding-results)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

---

## Introduction

The LLM-Guided Quality-Diversity (LLM-QD) Logo System is a revolutionary approach to automated logo design that combines:

- **Natural Language Understanding** (LLMs) - Understands design concepts semantically
- **Quality-Diversity Optimization** (MAP-Elites) - Systematically explores design space
- **Intelligent Search** (Curiosity-driven) - Actively seeks under-explored regions

### What You Can Do

- **Generate diverse logos** from natural language descriptions
- **Explore design space** systematically across multiple dimensions
- **Get high-quality results** with professional aesthetic metrics
- **Choose from hundreds** of unique designs in a single run
- **No expertise required** - just describe what you want

---

## What Makes This Revolutionary

### 1. Natural Language Interface

Traditional evolutionary systems require parameter tuning. LLM-QD understands plain English:

```
"minimalist tech logo with circular motifs conveying innovation and trust"
→ System generates 50+ diverse variations automatically
```

### 2. Systematic Diversity

Most systems generate variations randomly. LLM-QD explores systematically:

- **Traditional approach:** Generate 50 logos, get random variations
- **LLM-QD approach:** Generate 50 logos covering 50 distinct design niches

### 3. Intelligent Mutations

Random mutations are inefficient. LLM-QD understands what changes mean:

- **Random:** Change parameter X by ±0.1
- **LLM-QD:** "Make it more complex" → adds 10-15 SVG elements intelligently

### 4. Quality-Diversity Optimization

Optimizes for TWO objectives simultaneously:

1. **Quality:** Each logo scored by professional aesthetic metrics
2. **Diversity:** Logos distributed across 4D behavior space

**Result:** Not just ONE good logo, but HUNDREDS of good logos, each unique.

### 5. Curiosity-Driven Search

The system actively seeks novelty:

- Identifies under-explored regions
- Preferentially explores new niches
- Prevents premature convergence
- Similar to human creative exploration

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/svg-logo-ai.git
cd svg-logo-ai

# Install dependencies
pip install -r requirements.txt

# Set API key
export GOOGLE_API_KEY='your-gemini-api-key'
```

### Basic Usage

```python
from llm_qd_logo_system import LLMGuidedQDLogoSystem

# Initialize system
system = LLMGuidedQDLogoSystem()

# Run search with natural language query
archive = system.search(
    user_query="minimalist tech logo with circular motifs",
    iterations=100
)

# Get best diverse logos
best_logos = archive.get_best_logos(n=50)

# Save results
system.save_results()
```

### Command Line

```bash
# Run demo
python src/demo_llm_qd.py --run

# Run full experiment
python src/run_llm_qd_experiment.py

# Analyze results
python src/llm_qd_analysis.py experiments/llm_qd_20251127_120000
```

---

## How It Works

### Architecture Overview

```
Natural Language Query
        ↓
   Parse to Genome
        ↓
Generate Initial Population (20 diverse individuals)
        ↓
┌──────────────────────────┐
│   Main Search Loop       │
│  (100-1000 iterations)   │
│                          │
│  1. Select Parent        │ ← Curiosity-driven
│  2. Select Target        │ ← Empty neighbor cell
│  3. LLM Mutation         │ ← Intelligent, directed
│  4. Evaluate             │ ← Fitness + Behavior
│  5. Archive              │ ← Keep best per niche
└──────────────────────────┘
        ↓
  MAP-Elites Archive
  (10x10x10x10 grid)
        ↓
  50+ Diverse Logos
```

### Behavior Dimensions

The system explores a **4-dimensional behavior space**:

| Dimension | Description | Range |
|-----------|-------------|-------|
| **Complexity** | Number of SVG elements | Simple (10-15) ↔ Complex (55+) |
| **Style** | Geometric vs. Organic | Geometric (straight) ↔ Organic (curves) |
| **Symmetry** | Asymmetric vs. Symmetric | Asymmetric (0) ↔ Symmetric (1.0) |
| **Color Richness** | Monochrome vs. Colorful | Mono (1 color) ↔ Poly (5+ colors) |

Each dimension is divided into **10 bins**, creating a **10^4 = 10,000 cell grid**.

### Quality Evaluation

Each logo is evaluated on:

- **Aesthetic Score** (Golden ratio, color harmony, visual interest)
- **Professional Quality** (Scalability, technical correctness)
- **Design Principles** (Balance, symmetry, negative space)

Score range: **0-100** (higher is better)

### Intelligent Mutation

When mutating from cell (2,1,5,2) to (6,1,5,2):

```
Behavior delta: complexity +4, style 0, symmetry 0, color 0

LLM instruction:
"INCREASE COMPLEXITY: Add 12-20 more SVG elements (circles, rectangles, paths)
 while maintaining geometric style, symmetric layout, and duotone color scheme."

Result: Logo with ~15 new elements, similar style
```

This is **much more efficient** than random mutations.

---

## Using the System

### Natural Language Queries

#### Good Queries

✅ Specific but flexible:
```
"minimalist tech logo with circular motifs conveying innovation and trust"
"organic nature-inspired logo with flowing shapes and earth tones"
"bold geometric fintech logo conveying security and professionalism"
```

✅ Include:
- Style keywords (minimalist, modern, bold, organic)
- Industry context (tech, finance, healthcare)
- Desired characteristics (professional, playful, elegant)
- Color hints (earth tones, vibrant, monochrome)

#### Avoid

❌ Too vague:
```
"make me a logo"
"something cool"
```

❌ Too specific (over-constrains):
```
"logo with exactly 3 circles at coordinates (50,50), (100,50), (150,50)"
```

### Configuration Options

```python
system = LLMGuidedQDLogoSystem(
    grid_dimensions=(10, 10, 10, 10),  # 4D grid size
    experiment_name="my_experiment",    # For tracking
    model_name="gemini-2.0-flash-exp"  # LLM model
)

archive = system.search(
    user_query="your query here",
    iterations=100  # More = better coverage
)
```

### Recommended Settings

| Use Case | Iterations | Grid Size | Expected Coverage |
|----------|-----------|-----------|-------------------|
| **Quick test** | 25-50 | (5,5,5,5) | 10-20% |
| **Standard** | 100-200 | (10,10,10,10) | 15-30% |
| **Comprehensive** | 500-1000 | (10,10,10,10) | 30-50% |
| **High-res** | 1000+ | (15,15,15,15) | 10-20% |

---

## Understanding Results

### Archive Statistics

```python
stats = archive.get_statistics()

{
    'coverage': 0.23,          # 23% of cells filled
    'num_occupied': 2300,      # 2300 out of 10,000 cells
    'avg_fitness': 72.5,       # Average quality across all niches
    'max_fitness': 89.3,       # Best quality achieved
    'min_fitness': 58.1,       # Worst quality (still decent)
}
```

### Interpreting Coverage

- **<10%:** Under-explored, increase iterations
- **10-20%:** Good coverage for 100 iterations
- **20-30%:** Excellent coverage
- **>30%:** Exceptional, diminishing returns beyond this

### Quality Metrics

| Fitness Range | Interpretation |
|--------------|----------------|
| **90-100** | Exceptional professional quality |
| **80-90** | High quality, suitable for production |
| **70-80** | Good quality, minor refinements needed |
| **60-70** | Acceptable, may need revision |
| **<60** | Poor quality, needs significant work |

### Behavior Characterization

Each logo's behavior is characterized:

```python
behavior_data = characterizer.characterize(svg_code)

{
    'raw_scores': {
        'complexity': 28,      # 28 SVG elements
        'style': 0.3,          # 0.3 = geometric (0.0=pure geometric, 1.0=pure organic)
        'symmetry': 0.85,      # 0.85 = highly symmetric
        'color_richness': 0.5  # 0.5 = 3 colors (tritone)
    },
    'bins': (3, 3, 8, 5),      # Grid coordinates
    'details': {
        'complexity_rating': 'moderate',
        'style_rating': 'geometric',
        'symmetry_rating': 'symmetric',
        'color_rating': 'tritone'
    }
}
```

---

## Advanced Features

### Custom Genome Manipulation

```python
# Parse query to genome
genome = system.parse_query("minimalist tech logo")

# Customize genome
genome['complexity_target'] = 35  # More complex
genome['color_palette'] = ['#2563eb', '#3b82f6', '#60a5fa']  # Custom colors
genome['design_principles'].append('golden_ratio')  # Add principle

# Generate with custom genome
svg = system.mutator.generate_from_genome(genome)
```

### Filtering Results

```python
# Get best overall
best_logos = archive.get_best_logos(n=10)

# Filter by behavior
for behavior, entry in archive.archive.items():
    complexity, style, symmetry, color = behavior

    # Find highly symmetric geometric logos
    if symmetry >= 7 and style <= 3:
        print(f"Found: {entry.logo_id}")
```

### Export Options

```python
# Save all results
output_dir = system.save_results()

# Output structure:
# output_dir/
#   ├── archive.json          # Full archive metadata
#   ├── config.json           # Experiment configuration
#   ├── history.json          # Iteration-by-iteration history
#   ├── logo_001.svg          # Individual SVG files
#   ├── logo_002.svg
#   └── ...
```

### Batch Processing

```python
# Process multiple queries
queries = [
    "minimalist tech logo",
    "organic nature logo",
    "geometric finance logo"
]

results = []
for query in queries:
    archive = system.search(query, iterations=100)
    results.append({
        'query': query,
        'archive': archive,
        'stats': archive.get_statistics()
    })
```

---

## Troubleshooting

### Common Issues

#### "GOOGLE_API_KEY not set"

```bash
export GOOGLE_API_KEY='your-key-here'
```

#### Low Coverage (<5%)

**Problem:** Not exploring enough
**Solution:** Increase iterations (100 → 200+)

#### Poor Quality (fitness <60)

**Problem:** Query too vague or constraints too loose
**Solution:** Be more specific in query, add style keywords

#### Slow Performance

**Problem:** Network latency
**Solutions:**
- Use smaller grid for testing: (5,5,5,5)
- Reduce iterations for quick tests
- Check internet connection

#### Out of Memory

**Problem:** Large archive with many high-res SVGs
**Solution:**
- Process in batches
- Clear archive periodically
- Use smaller grid dimensions

### Debugging

```python
# Enable verbose logging
system.tracker.log_step(
    step_type="debug",
    description="Debug message",
    data={"variable": value}
)

# Check archive state
print(f"Occupied cells: {len(archive.archive)}")
print(f"Coverage: {archive.get_coverage()}")

# Inspect specific cell
entry = archive.get((5, 5, 5, 5))
if entry:
    print(f"Cell (5,5,5,5): fitness={entry.fitness}")
```

---

## API Reference

### LLMGuidedQDLogoSystem

Main system class.

#### `__init__(grid_dimensions, experiment_name, model_name)`

Initialize system.

**Parameters:**
- `grid_dimensions` (tuple): Grid size per dimension, e.g., (10,10,10,10)
- `experiment_name` (str): Name for tracking
- `model_name` (str): LLM model to use

#### `search(user_query, iterations)`

Main search algorithm.

**Parameters:**
- `user_query` (str): Natural language description
- `iterations` (int): Number of search iterations

**Returns:** `MAPElitesArchive`

#### `generate_diverse_logos(query, n)`

User-facing API for getting N diverse logos.

**Parameters:**
- `query` (str): Natural language description
- `n` (int): Number of logos to return

**Returns:** List of logo dictionaries

#### `save_results(output_dir)`

Save complete results.

**Parameters:**
- `output_dir` (str, optional): Where to save

**Returns:** Path to output directory

### MAPElitesArchive

Archive maintaining diverse logo population.

#### `add(...)`

Add logo to archive (if better than existing).

**Returns:** `bool` - True if added

#### `get(behavior)`

Get logo at specific behavior coordinates.

**Returns:** `ArchiveEntry` or `None`

#### `get_best_logos(n)`

Get top N logos by fitness.

**Returns:** List of `ArchiveEntry`

#### `get_statistics()`

Get archive statistics.

**Returns:** Dict with coverage, fitness stats, etc.

---

## Best Practices

### For Best Results

1. **Be specific in queries:** Include style, industry, mood
2. **Run enough iterations:** At least 100 for meaningful coverage
3. **Review multiple options:** Check various behavior niches
4. **Iterate on favorites:** Use best logos as starting points
5. **Track experiments:** Use meaningful experiment names

### For Efficiency

1. **Start with small grid:** Test with (5,5,5,5) before (10,10,10,10)
2. **Quick iterations:** Run 25-50 iterations for prototyping
3. **Batch similar queries:** Process related queries together
4. **Reuse successful patterns:** Learn from previous runs

### For Production

1. **Validate outputs:** Always check SVG validity
2. **Test at multiple sizes:** Ensure scalability
3. **Review aesthetic scores:** Aim for fitness >75
4. **Get diverse options:** Explore multiple behavior regions
5. **Document choices:** Track which designs work best

---

## Citation

If you use this system in research, please cite:

```bibtex
@software{llm_qd_logo_2025,
  title={LLM-Guided Quality-Diversity Logo Generation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/svg-logo-ai}
}
```

---

## Support

- **GitHub Issues:** https://github.com/yourusername/svg-logo-ai/issues
- **Documentation:** https://github.com/yourusername/svg-logo-ai/docs
- **Email:** your.email@example.com

---

## License

MIT License - see LICENSE file for details.

---

**User Guide Version 1.0 - November 27, 2025**
