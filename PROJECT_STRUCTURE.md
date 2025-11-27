# Project Structure

Complete directory organization and navigation guide for the LLM-QD Logo System.

## Overview

This project is organized into clear functional areas: core source code, experiments, documentation, configuration, and output. Total size: ~17,500 lines of production code + 24,000 words of documentation.

## Directory Tree

```
svg-logo-ai/
├── src/                          # Core source code (42 Python files, 17,500 LOC)
├── experiments/                  # Experiment results and data
├── docs/                         # Comprehensive documentation (24K+ words)
├── output/                       # Generated logos and artifacts
├── data/                         # Data files and resources
├── notebooks/                    # Jupyter notebooks for exploration
├── deploy/                       # Deployment configurations
├── models/                       # Placeholder for trained models
├── venv/                         # Python virtual environment (not committed)
├── chroma_db/                    # ChromaDB vector database (not committed)
├── chroma_experiments/           # Experiment-specific databases (not committed)
│
├── README.md                     # Project overview and quick start
├── QUICKSTART.md                 # 5-minute getting started guide
├── CONTRIBUTING.md               # Contribution guidelines
├── CHANGELOG.md                  # Version history and changes
├── LICENSE                       # MIT License
├── CITATION.cff                  # Academic citation information
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── .gitignore                    # Git ignore rules
│
└── [Additional Documentation]    # Implementation reports and guides
```

## Core Source Code (`src/`)

**42 Python files, ~17,500 lines of code**

### Main QD System (4 files)
- `llm_qd_logo_system.py` (1,200 LOC) - **Main LLM-QD system**
  - Integrates all components into unified QD framework
  - Implements MAP-Elites with LLM guidance
  - Manages archive, mutations, and search strategies

- `map_elites_archive.py` (600 LOC) - **Archive management**
  - 10×10×10×10 grid (10,000 cells)
  - Behavioral descriptor mapping
  - Efficient storage and retrieval

- `behavior_characterization.py` (800 LOC) - **4D behavioral features**
  - Complexity: Element count (10-55+)
  - Style: Geometric ↔ Organic (0-1)
  - Symmetry: Asymmetric ↔ Symmetric (0-1)
  - Color Richness: Monochrome ↔ Polychrome (0-1)

- `qd_search_strategies.py` (500 LOC) - **Search algorithms**
  - Random search baseline
  - Curiosity-driven exploration
  - Novelty search
  - Quality-diversity balance

### LLM Integration (4 files)
- `llm_guided_mutation.py` (900 LOC) - **Semantic mutations**
  - Style transformations
  - Complexity control
  - Color palette mutations
  - Symmetry adjustments

- `semantic_mutator.py` (700 LOC) - **Low-level mutations**
  - SVG element manipulation
  - Geometric transformations
  - Color space operations

- `nl_query_parser.py` (600 LOC) - **Natural language interface**
  - Query parsing and understanding
  - Behavioral constraint extraction
  - Interactive query builder

- `llm_evaluator.py` (1,000 LOC) - **Quality evaluation**
  - Multi-dimensional fitness (4 components)
  - Aesthetic, golden ratio, color harmony, technical
  - LLM-based quality assessment

### Logo Generation (5 files)
- `gemini_svg_generator_v2.py` (800 LOC) - **Primary generator**
  - Improved prompts and error handling
  - SVG syntax validation
  - Retry logic and rate limiting

- `gemini_svg_generator.py` (600 LOC) - **Original generator**
  - Legacy implementation
  - Baseline generation

- `llm_logo_generator.py` (500 LOC) - **Generic LLM wrapper**
  - Multi-provider support (Gemini, OpenAI)
  - Unified interface

- `logo_validator.py` (400 LOC) - **SVG validation**
  - Syntax checking
  - Semantic validation
  - Quality constraints

- `logo_metadata.py` (300 LOC) - **Metadata management**
  - Structured metadata extraction
  - Design principle encoding

### RAG System (3 files)
- `rag_evolutionary_system.py` (700 LOC) - **RAG-enhanced evolution**
  - Few-shot learning from successful designs
  - 2.2% fitness improvement (90→92/100)
  - Semantic similarity search

- `knowledge_base.py` (600 LOC) - **Vector database**
  - ChromaDB integration
  - Logo storage and retrieval
  - Design principle embeddings

- `initialize_rag_kb.py` (400 LOC) - **KB initialization**
  - Populate knowledge base
  - Index existing logos
  - Setup embeddings

### Experiment Framework (5 files)
- `run_llm_qd_experiment.py` (500 LOC) - **Main experiment runner**
  - Command-line interface
  - Configuration management
  - Result aggregation

- `map_elites_experiment.py` (600 LOC) - **MAP-Elites experiments**
  - Standard QD experiments
  - Baseline comparisons

- `rag_experiment_runner.py` (500 LOC) - **RAG experiments**
  - RAG-enhanced evolution
  - Comparative analysis

- `experiment_tracker.py` (700 LOC) - **Experiment logging**
  - ChromaDB tracking
  - Complete experiment history
  - Reproducibility support

- `evolutionary_logo_system.py` (800 LOC) - **Baseline EA**
  - Standard genetic algorithm
  - Baseline for comparisons

### Visualization & Analysis (6 files)
- `qd_visualization.py` (800 LOC) - **QD visualizations**
  - 2D heatmaps
  - Coverage evolution plots
  - Fitness distributions

- `visualize_llm_qd.py` (500 LOC) - **LLM-QD specific plots**
  - Archive visualizations
  - Behavioral space exploration

- `visualize_map_elites.py` (400 LOC) - **MAP-Elites plots**
  - Standard QD visualizations

- `llm_qd_analysis.py` (600 LOC) - **Statistical analysis**
  - Coverage metrics
  - QD-score calculation
  - Performance comparisons

- `analyze_experiment.py` (500 LOC) - **Experiment analysis**
  - Result parsing
  - Metric extraction

- `generate_comparison_logos.py` (400 LOC) - **Comparative visualization**
  - Side-by-side comparisons
  - Visual similarity analysis

### Demonstration & Testing (6 files)
- `demo_llm_qd.py` (300 LOC) - **Quick demo**
  - 5-logo demonstration
  - 30-second test

- `demo_experiment_with_mock_data.py` (400 LOC) - **Mock testing**
  - No API calls required
  - Fast development testing

- `demo_new_metrics.py` (300 LOC) - **Metrics demo**
  - Showcase evaluation system

- `test_llm_components.py` (600 LOC) - **Component tests**
  - Unit tests for LLM integration

- `test_qd_system.py` (500 LOC) - **QD system tests**
  - Integration tests

- `test_map_elites_small.py` (400 LOC) - **Small-scale tests**
  - Quick MAP-Elites verification

### Utilities (8 files)
- `logo_examples.py` (200 LOC) - **Example logos**
  - Sample SVG templates
  - Design patterns

- `populate_knowledge.py` (300 LOC) - **KB population**
  - Bulk logo import

- `reevaluate_logos.py` (400 LOC) - **Batch re-evaluation**
  - Update fitness scores

- `example_usage.py` (200 LOC) - **Usage examples**
  - API demonstrations

- `update_research_findings.py` (300 LOC) - **Documentation updater**
  - Automated doc generation

- `gallery_generator.py` (600 LOC) - **Web gallery**
  - HTML gallery generation

- `cloudflare_deployer.py` (500 LOC) - **Deployment automation**
  - Cloudflare Workers deployment

- `purge_cloudflare_cache.py` (200 LOC) - **Cache management**
  - CDN cache purging

### Deployment Scripts (3 files)
- `deploy_api.py` (300 LOC) - **API deployment**
- `deploy_direct.py` (200 LOC) - **Direct deployment**
- `add_dns.py` (150 LOC) - **DNS configuration**

## Experiments (`experiments/`)

Organized by timestamp and experiment type.

### Structure
```
experiments/
├── experiment_20251127_053108/      # Baseline EA (best: 90/100)
│   ├── gen5_*.svg                   # Generated logos
│   ├── history.json                 # Generation history
│   ├── comparison.json              # Comparative metrics
│   └── *.png                        # Visualization plots
│
├── rag_experiment_20251127_090317/  # RAG-enhanced (best: 92/100)
│   ├── logos/                       # Generated logos
│   ├── metrics.json                 # Detailed metrics
│   └── convergence_plot.png         # Convergence visualization
│
├── map_elites_20251127_074420/      # MAP-Elites test (4% coverage)
│   ├── archive.json                 # Archive state
│   ├── heatmaps/                    # Behavioral space heatmaps
│   └── logos/                       # Archive contents
│
├── llm_qd_*/                        # Full LLM-QD experiments
│   ├── archive.json                 # Final archive
│   ├── logos/                       # All generated logos
│   ├── metrics.json                 # Comprehensive metrics
│   ├── heatmaps/                    # Coverage visualizations
│   ├── report.md                    # Human-readable summary
│   └── config.json                  # Experiment configuration
│
└── demo_experiment_*/               # Demo runs
    ├── comparison.json
    ├── history.json
    └── *.png
```

### Experiment Types

1. **Baseline EA**: Standard evolutionary algorithm
2. **RAG-Enhanced**: Retrieval-augmented generation
3. **MAP-Elites**: Quality-diversity baseline
4. **LLM-QD**: Full system with all features
5. **Demo**: Quick demonstrations

## Documentation (`docs/`)

**24,000+ words across 12 files**

### Research Documentation (4 files)
- **`EVOLUTIONARY_PAPER_DRAFT.md`** (5,000 words) - Publication-ready paper
- **`LLM_QD_ARCHITECTURE.md`** (18,000 words) - Complete system architecture
- **`RESEARCH_FINDINGS.md`** (2,000 words) - Initial feasibility study
- **`LLM_QD_PAPER_DRAFT.md`** (6,000 words) - Alternative paper draft

### Technical Documentation (4 files)
- **`LLM_QD_USER_GUIDE.md`** (3,500 words) - API and usage guide
- **`ADVANCED_OPTIMIZATION.md`** (14,000 words) - Optimization techniques
- **`PROMPT_ENGINEERING.md`** (14,000 words) - Prompt design guide
- **`QUALITY_METRICS_ANALYSIS.md`** (13,000 words) - Metrics deep dive

### Design Documentation (3 files)
- **`LOGO_DESIGN_PRINCIPLES.md`** (26,000 words) - Design theory
- **`DATASETS.md`** (5,000 words) - Available datasets
- **`METRICS_COMPARISON_VISUAL.md`** (5,000 words) - Visual comparisons

### Meta Documentation (1 file)
- **`LEARNING_SYSTEMS.md`** (26,000 words) - Learning mechanisms

## Root Documentation

### Essential Files
- **`README.md`** (1,000 words) - Project overview, quick start, results
- **`QUICKSTART.md`** (2,000 words) - 5-minute getting started guide
- **`CONTRIBUTING.md`** (3,000 words) - Contribution guidelines
- **`CHANGELOG.md`** (4,000 words) - Complete version history
- **`LICENSE`** - MIT License

### Research Reports (Root)
- **`REVOLUTIONARY_METHODS_RESEARCH.md`** (15,000 words) - Literature survey (50+ papers)
- **`REPORTE_RESULTADOS_ESPAÑOL.md`** (5,000 words) - Results report (Spanish)
- **`RESEARCH_EXECUTIVE_SUMMARY.md`** (3,000 words) - Executive summary

### Implementation Reports (Root)
- **`LLM_COMPONENTS_IMPLEMENTATION_REPORT.md`** (7,000 words) - LLM components
- **`LLM_QD_INTEGRATION_REPORT.md`** (5,000 words) - Integration details
- **`QD_SYSTEM_IMPLEMENTATION_REPORT.md`** (6,000 words) - QD specifics
- **`RAG_SYSTEM_IMPLEMENTATION_SUMMARY.md`** (3,000 words) - RAG details
- **`GALLERY_SYSTEM.md`** (4,000 words) - Gallery documentation
- **`CLOUDFLARE_DEPLOYMENT.md`** (3,000 words) - Deployment guide

### Quick Reference Guides (Root)
- **`LLM_COMPONENTS_QUICKSTART.md`** (2,000 words) - LLM components quick start
- **`LLM_QD_QUICKSTART.md`** (1,500 words) - LLM-QD quick start
- **`IMPLEMENTATION_COMPLETE.txt`** (2,000 words) - Implementation checklist
- **`IMPLEMENTATION_SUMMARY.md`** (3,000 words) - Summary overview

## Output (`output/`)

Generated artifacts organized by type.

### Structure
```
output/
├── logos/                    # Generated SVG files
│   ├── techflow_*.svg
│   ├── healthplus_*.svg
│   ├── finvest_*.svg
│   └── ...
│
├── deploy/                   # Deployment-ready files
│   ├── index.html
│   ├── gallery.html
│   ├── *.svg
│   └── logos_metadata.json
│
├── llm_tests/                # LLM component tests
│   ├── test_report.json
│   └── TEST_REPORT.md
│
├── gallery.html              # Logo gallery
├── logos_metadata.json       # Metadata index
└── logo-gallery-deploy.zip   # Deployment package
```

## Configuration Files

### Environment Configuration
- **`.env.example`** - Template for environment variables
- **`.env.cloudflare.example`** - Cloudflare-specific configuration
- **`.env`** - Actual credentials (not committed)

### Project Configuration
- **`requirements.txt`** - Python dependencies
- **`.gitignore`** - Git ignore rules
- **`CITATION.cff`** - Academic citation metadata

### Deployment Configuration
- **`deploy/wrangler.toml`** - Cloudflare Workers config
- **`nginx-config.conf`** - Nginx configuration
- **`nginx-config-temp.conf`** - Temporary nginx config

## Data Files (`data/`)

Resources and reference data.

```
data/
├── reference_logos/          # Example logos for training
├── design_principles/        # Design rule databases
└── color_palettes/           # Color scheme collections
```

## Notebooks (`notebooks/`)

Jupyter notebooks for exploration and analysis.

```
notebooks/
└── 01_explore_knowledge_base.ipynb    # Knowledge base exploration
```

## Not Committed (`.gitignore`)

### Large Databases
- `chroma_db/` - Main ChromaDB database
- `chroma_experiments/` - Experiment-specific databases

### Virtual Environment
- `venv/` - Python virtual environment
- `__pycache__/` - Python bytecode
- `*.pyc` - Compiled Python files

### Secrets
- `.env` - Environment variables with API keys
- `*.key` - Key files
- `credentials*.json` - Credential files

### Temporary Files
- `*.log` - Log files
- `*.tmp` - Temporary files
- `nohup.out` - Background process output

## Navigation Guide

### For First-Time Users
1. Start with `README.md` for overview
2. Follow `QUICKSTART.md` for 5-minute setup
3. Run `src/demo_llm_qd.py` for quick demo
4. Explore `experiments/` for example results

### For Developers
1. Read `CONTRIBUTING.md` for guidelines
2. Study `docs/LLM_QD_ARCHITECTURE.md` for system design
3. Review `src/llm_qd_logo_system.py` for main logic
4. Check `tests/` for testing approach

### For Researchers
1. Read `docs/EVOLUTIONARY_PAPER_DRAFT.md` for research
2. Review `REVOLUTIONARY_METHODS_RESEARCH.md` for literature
3. Analyze `experiments/` for results
4. Study `REPORTE_RESULTADOS_ESPAÑOL.md` for detailed results

### For System Architects
1. Study `docs/LLM_QD_ARCHITECTURE.md` for design
2. Review component implementations in `src/`
3. Understand data flow between modules
4. Check deployment configurations

## File Counts & Statistics

### Source Code
- **Total Python files**: 42
- **Total lines of code**: ~17,500
- **Average file size**: ~400 LOC
- **Largest file**: `llm_qd_logo_system.py` (1,200 LOC)
- **Test coverage**: TBD (tests in development)

### Documentation
- **Total documentation files**: 30+
- **Total word count**: 24,000+
- **Average document**: 800 words
- **Largest document**: `LOGO_DESIGN_PRINCIPLES.md` (26,000 words)

### Experiments
- **Completed experiments**: 10+
- **Generated logos**: 67 unique SVGs
- **Best fitness achieved**: 92/100
- **Maximum coverage**: 15% (test runs)

## Key Entry Points

### Running Experiments
```bash
python src/demo_llm_qd.py                # Quick demo
python src/run_llm_qd_experiment.py      # Full experiment
python src/rag_experiment_runner.py      # RAG experiment
```

### Analysis & Visualization
```bash
python src/visualize_llm_qd.py <exp_dir>     # Visualize results
python src/analyze_experiment.py <exp_dir>   # Analyze metrics
python src/llm_qd_analysis.py --top 10       # Top logos
```

### Utilities
```bash
python src/initialize_rag_kb.py          # Setup knowledge base
python src/gallery_generator.py          # Generate gallery
python src/test_llm_components.py        # Test system
```

## Dependencies

### Core
- `chromadb==0.4.22` - Vector database
- `google-cloud-aiplatform==1.42.1` - Gemini API
- `numpy==1.26.4` - Numerical computing
- `pandas==2.2.0` - Data analysis

### Visualization & Processing
- `pillow==10.2.0` - Image processing
- `cairosvg==2.7.1` - SVG rendering
- `svgwrite==1.4.3` - SVG generation
- `beautifulsoup4==4.12.3` - SVG parsing

### Utilities
- `requests==2.31.0` - HTTP client
- `python-dotenv==1.0.1` - Environment config

## License

All files are licensed under MIT License. See `LICENSE` file.

## Contributing

Follow guidelines in `CONTRIBUTING.md`. Key points:
- PEP 8 style (100 char lines)
- Type hints required
- Tests for new features
- Documentation updates

## Citation

Use `CITATION.cff` for academic citations:
```
Arancibia, L. (2025). LLM-QD Logo System: Large Language Model
Guided Quality-Diversity for Logo Generation.
GitHub: https://github.com/larancibia/svg-logo-ai
```

---

**Last Updated**: 2025-11-27
**Version**: 1.0.0
**Total Project Size**: 17,500 LOC + 24,000 words documentation
