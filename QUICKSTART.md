# Quick Start Guide

Get up and running with the LLM-QD Logo System in 5 minutes.

## Prerequisites

- Python 3.10 or higher
- Google API key (for Gemini)
- 2GB+ free disk space

## Installation (2 minutes)

### 1. Clone and Setup Environment

```bash
# Clone repository (if not already cloned)
cd /home/luis/svg-logo-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Google API key
# GOOGLE_API_KEY=your_actual_api_key_here
```

**Get your API key:** https://ai.google.dev/

## Running Your First Experiment (3 minutes)

### Option A: Quick Demo (Fastest - 30 seconds)

Test the system with a simple demo:

```bash
python src/demo_llm_qd.py
```

This will:
- Generate 5 logos with different styles
- Show behavioral diversity metrics
- Save results to `output/demo/`

### Option B: Full LLM-QD Experiment (3 minutes)

Run a complete Quality-Diversity experiment:

```bash
python src/run_llm_qd_experiment.py
```

Parameters (edit in script):
- `num_iterations`: Number of QD iterations (default: 20)
- `archive_size`: Grid dimensions (default: 10x10x10x10)
- `company_name`: Logo target (default: "TechCorp")

### Option C: RAG-Enhanced Evolution (5 minutes)

Run evolutionary algorithm with retrieval-augmented generation:

```bash
python src/rag_experiment_runner.py
```

This typically produces logos with 90-92/100 fitness.

### Option D: Natural Language Query (Advanced)

Generate logos using natural language:

```bash
# Interactive mode
python src/nl_query_parser.py

# Example queries:
# "Create a geometric, symmetric logo with high color richness"
# "Generate an organic, asymmetric logo with minimal colors"
# "Find logos similar to tech startups but more playful"
```

## Understanding Results (5 minutes)

### Output Structure

```
experiments/
└── llm_qd_20251127_HHMMSS/
    ├── logos/              # Generated SVG files
    ├── metrics.json        # Quality metrics for each logo
    ├── archive.json        # MAP-Elites archive state
    ├── heatmaps/           # Behavioral space visualizations
    └── report.md           # Human-readable summary
```

### Key Metrics Explained

**Fitness Score (0-100)**
- 85-89: Good quality
- 90-94: Excellent quality
- 95+: Exceptional quality

**Behavioral Dimensions:**
1. **Complexity**: Number of SVG elements (10-55+)
2. **Style**: Geometric (0) to Organic (1)
3. **Symmetry**: Asymmetric (0) to Symmetric (1)
4. **Color Richness**: Monochrome (0) to Polychrome (1)

**Quality Components:**
- **Aesthetic**: Visual appeal (0-100)
- **Golden Ratio**: Proportion harmony (0-100)
- **Color Harmony**: Color theory compliance (0-100)
- **Technical**: SVG validity and structure (0-100)

### View Results

1. **Browse Logos:**
   ```bash
   # Open in file explorer
   xdg-open experiments/llm_qd_*/logos/  # Linux
   open experiments/llm_qd_*/logos/      # macOS
   explorer experiments\llm_qd_*\logos\  # Windows
   ```

2. **Read Report:**
   ```bash
   cat experiments/llm_qd_*/report.md
   ```

3. **Visualize Archive:**
   ```bash
   python src/visualize_llm_qd.py experiments/llm_qd_*/
   ```

   This generates heatmaps showing:
   - Coverage across behavioral dimensions
   - Fitness distribution
   - Diversity statistics

### Analyzing Specific Logos

```bash
# Get detailed metrics for a logo
python src/analyze_experiment.py --logo experiments/llm_qd_*/logos/gen5_*.svg

# Compare multiple logos
python src/generate_comparison_logos.py --experiment experiments/llm_qd_*/
```

## Common Issues & Solutions

### Issue 1: API Rate Limits

**Symptom:** `429 Too Many Requests` error

**Solution:**
```python
# In experiment script, add delays:
import time
time.sleep(2)  # Wait 2 seconds between API calls
```

Or use `gemini-1.5-flash` instead of `gemini-2.0-flash-exp`:
```python
model = "gemini-1.5-flash"  # More stable, higher rate limits
```

### Issue 2: ChromaDB Lock Error

**Symptom:** `sqlite3.OperationalError: database is locked`

**Solution:**
```bash
# Close all running experiments, then:
rm -rf chroma_db/
python src/initialize_rag_kb.py  # Rebuild knowledge base
```

### Issue 3: Low Fitness Scores

**Symptom:** All logos scoring below 80

**Solution:**
1. Check API key is valid
2. Ensure you're using `gemini-1.5-flash` or newer
3. Increase `num_iterations` for more optimization time
4. Enable RAG for better results:
   ```python
   use_rag = True  # In experiment script
   ```

### Issue 4: Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'chromadb'`

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue 5: Out of Memory

**Symptom:** Crashes during large experiments

**Solution:**
```python
# Reduce archive size
archive_dimensions = (5, 5, 5, 5)  # Instead of (10, 10, 10, 10)

# Or run fewer iterations
num_iterations = 10  # Instead of 100
```

## Next Steps

### For Researchers

1. **Read the paper draft:** `docs/EVOLUTIONARY_PAPER_DRAFT.md`
2. **Understand the architecture:** `docs/LLM_QD_ARCHITECTURE.md`
3. **Review experimental results:** `REPORTE_RESULTADOS_ESPAÑOL.md`
4. **Explore advanced optimization:** `docs/ADVANCED_OPTIMIZATION.md`

### For Developers

1. **API reference:** `docs/LLM_QD_USER_GUIDE.md`
2. **Extend the system:** `CONTRIBUTING.md`
3. **Run full experiments:** See `experiments/` directory
4. **Customize metrics:** Edit `src/llm_evaluator.py`

### For Practitioners

1. **Generate production logos:**
   ```bash
   python src/run_llm_qd_experiment.py --company "YourCompany" --iterations 50
   ```

2. **Fine-tune style preferences:**
   Edit `src/logo_metadata.py` to adjust:
   - Color palettes
   - Style preferences
   - Design constraints

3. **Export best logos:**
   ```bash
   # Find top 10 logos by fitness
   python src/llm_qd_analysis.py --top 10 --export output/best/
   ```

## Quick Reference

### Essential Commands

```bash
# Setup
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key"

# Run experiments
python src/demo_llm_qd.py                    # Quick demo
python src/run_llm_qd_experiment.py          # Full QD
python src/rag_experiment_runner.py          # RAG evolution

# Analyze results
python src/visualize_llm_qd.py <experiment_dir>
python src/analyze_experiment.py <experiment_dir>

# Utilities
python src/initialize_rag_kb.py              # Setup knowledge base
python src/test_llm_components.py            # Test system components
```

### Key Files

- `src/llm_qd_logo_system.py` - Main QD system
- `src/llm_guided_mutation.py` - Semantic mutations
- `src/behavior_characterization.py` - 4D behavioral features
- `src/llm_evaluator.py` - Quality metrics
- `src/map_elites_archive.py` - Archive management

### Performance Benchmarks

| Experiment Type | Duration | Logos Generated | Best Fitness | Coverage |
|----------------|----------|-----------------|--------------|----------|
| Quick Demo | 30s | 5 | 75-85 | N/A |
| Small QD (10 iter) | 3 min | 20-30 | 82-88 | 5-10% |
| Full QD (50 iter) | 15 min | 100-150 | 88-92 | 15-30% |
| RAG Evolution | 5 min | 25-35 | 90-92 | N/A |

*Timings assume `gemini-1.5-flash` with good API latency*

## Getting Help

- **Documentation:** See `docs/INDEX.md` for complete documentation index
- **Examples:** Check `examples/` directory for code samples
- **Issues:** Search existing issues or open a new one on GitHub
- **Community:** Join discussions in GitHub Discussions

## Success Criteria

You've successfully set up the system when:

1. Demo runs without errors
2. Generates SVG files in `output/`
3. Fitness scores are above 75
4. Visualizations display correctly

**Ready to dive deeper?** Continue to the [User Guide](docs/LLM_QD_USER_GUIDE.md) for comprehensive documentation.
