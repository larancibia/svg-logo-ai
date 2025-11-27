# LLM-QD Logo Paper: Reproducibility and Submission Guide

**Paper:** LLM-Guided Quality-Diversity for Evolutionary Logo Generation
**Authors:** Luis @ GuanacoLabs, Claude (Anthropic)
**Target Venues:** ICLR 2026 / GECCO 2026
**Date:** November 27, 2025

---

## Overview

This document provides instructions for:
1. Reproducing all experiments in the paper
2. Generating all figures
3. Preparing the paper for submission
4. Submission checklist

---

## 1. Reproducing Experiments

### 1.1 Environment Setup

```bash
# Clone repository
git clone https://github.com/larancibia/svg-logo-ai.git
cd svg-logo-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
export GOOGLE_API_KEY="your-gemini-api-key"
```

### 1.2 Baseline Evolutionary Experiment

**Command:**
```bash
python src/evolutionary_logo_system.py \
    --company "NeuralFlow" \
    --industry "artificial intelligence" \
    --population 10 \
    --generations 5 \
    --output experiments/baseline_reproduction
```

**Expected Results:**
- 50 logos generated
- Max fitness: ~90/100
- Mean fitness: ~88.2/100
- Runtime: ~45 minutes
- Cost: ~$0.03

**Output Files:**
- `experiments/baseline_reproduction/final_population.json`
- `experiments/baseline_reproduction/history.json`
- `experiments/baseline_reproduction/gen5_*.svg` (10 SVG files)

### 1.3 RAG-Enhanced Evolutionary Experiment

**Step 1: Initialize Knowledge Base**
```bash
python src/initialize_knowledge_base.py \
    --source experiments/baseline_reproduction/final_population.json \
    --min_fitness 85
```

**Step 2: Run RAG Experiment**
```bash
python src/rag_experiment_runner.py \
    --company "NeuralFlow" \
    --industry "artificial intelligence" \
    --population 20 \
    --generations 5 \
    --kb_path chroma_db/logos \
    --output experiments/rag_reproduction
```

**Expected Results:**
- 100 logos generated
- Max fitness: ~92/100
- Mean fitness: ~88.5/100
- RAG retrievals: ~60
- Runtime: ~90 minutes
- Cost: ~$0.07

**Output Files:**
- `experiments/rag_reproduction/final_population.json`
- `experiments/rag_reproduction/history.json`
- `experiments/rag_reproduction/kb_stats.json`
- `experiments/rag_reproduction/gen*_*.svg` (20+ SVG files)

### 1.4 MAP-Elites (Random Mutations) Experiment

```bash
python src/map_elites_experiment.py \
    --company "NeuralFlow" \
    --industry "artificial intelligence" \
    --grid_dims 5 5 5 5 \
    --n_init 50 \
    --n_iter 100 \
    --mutation_type "random" \
    --output experiments/map_elites_random_reproduction
```

**Expected Results:**
- 150 logos evaluated
- Coverage: 4.0-4.5%
- Mean fitness: ~87/100
- Runtime: ~30 minutes
- Cost: ~$0.04

**Output Files:**
- `experiments/map_elites_random_reproduction/archive.json`
- `experiments/map_elites_random_reproduction/experiment_summary.json`
- `experiments/map_elites_random_reproduction/*.svg` (25-30 SVG files)

### 1.5 LLM-QD Logo (Full System) Experiment

**Test Configuration (5×5×5×5 grid):**
```bash
python src/map_elites_experiment.py \
    --company "NeuralFlow" \
    --industry "artificial intelligence" \
    --grid_dims 5 5 5 5 \
    --n_init 50 \
    --n_iter 150 \
    --mutation_type "llm_guided" \
    --output experiments/llm_qd_test_reproduction
```

**Full-Scale Configuration (10×10×10×10 grid):**
```bash
python src/map_elites_experiment.py \
    --company "NeuralFlow" \
    --industry "artificial intelligence" \
    --grid_dims 10 10 10 10 \
    --n_init 200 \
    --n_iter 500 \
    --mutation_type "llm_guided" \
    --target_selection_strategy "curiosity" \
    --output experiments/llm_qd_full_reproduction
```

**Expected Results (Full-Scale):**
- 700 logos evaluated (200 init + 500 iter)
- Coverage: 10-30% (1,000-3,000 occupied cells)
- Mean fitness: 85-89/100
- Runtime: ~2-3 hours
- Cost: ~$0.30-0.50

**Output Files:**
- `experiments/llm_qd_full_reproduction/archive.json`
- `experiments/llm_qd_full_reproduction/experiment_summary.json`
- `experiments/llm_qd_full_reproduction/*.svg` (1,000-3,000 SVG files)

---

## 2. Generating Figures

All figures can be generated using the visualization scripts:

### 2.1 Figure 1: System Architecture

**Manual Creation Required**
- Use draw.io or similar tool
- Components:
  - LLM Generation
  - Behavioral Characterization
  - MAP-Elites Archive (4D grid)
  - Target Selection
  - LLM-Guided Mutation
  - Loop arrows
- Export as PNG (300 DPI) or PDF (vector)

**Reference:** See `/docs/LLM_QD_ARCHITECTURE.md` Section 2.1 for diagram structure

### 2.2 Figure 2: 4D Behavioral Space Visualization

```bash
python scripts/visualize_behavioral_space.py \
    --archive experiments/llm_qd_full_reproduction/archive.json \
    --output figures/behavioral_space_3d.png \
    --mode "3d_scatter" \
    --dimensions complexity style symmetry \
    --color_by color_richness
```

**Output:** `figures/behavioral_space_3d.png` (3D scatter plot)

### 2.3 Figure 3: Coverage Comparison Bar Chart

```bash
python scripts/generate_coverage_comparison.py \
    --experiments \
        experiments/baseline_reproduction \
        experiments/rag_reproduction \
        experiments/map_elites_random_reproduction \
        experiments/llm_qd_test_reproduction \
        experiments/llm_qd_full_reproduction \
    --output figures/coverage_comparison.png
```

**Output:** `figures/coverage_comparison.png` (bar chart)

### 2.4 Figure 4: Quality-Diversity Scatter Plot

```bash
python scripts/generate_qd_scatter.py \
    --experiments \
        experiments/baseline_reproduction \
        experiments/rag_reproduction \
        experiments/map_elites_random_reproduction \
        experiments/llm_qd_full_reproduction \
    --output figures/qd_scatter.png \
    --x_metric "behavioral_diversity" \
    --y_metric "mean_fitness"
```

**Output:** `figures/qd_scatter.png` (scatter plot)

### 2.5 Figure 5: Heatmap (Complexity × Style)

```bash
python scripts/visualize_map_elites.py \
    --archive experiments/llm_qd_full_reproduction/archive.json \
    --output figures/heatmap_complexity_style.png \
    --type "heatmap" \
    --dimensions complexity style \
    --metric "fitness"
```

**Output:** `figures/heatmap_complexity_style.png` (2D heatmap)

### 2.6 Figure 6: Example Logos from Different Niches

```bash
python scripts/extract_niche_examples.py \
    --archive experiments/llm_qd_full_reproduction/archive.json \
    --output figures/niche_examples/ \
    --cells \
        "[2,0,8,0]" \
        "[7,8,2,6]" \
        "[4,4,5,2]" \
        "[1,0,9,0]" \
        "[8,5,5,8]" \
        "[3,7,3,4]" \
        "[6,2,1,1]" \
        "[5,5,7,5]" \
        "[9,9,0,9]"
```

**Output:**
- `figures/niche_examples/grid.png` (3×3 grid of logos)
- Individual SVG files for each niche

### 2.7 Figure 7: Convergence Curves

```bash
python scripts/plot_convergence.py \
    --experiments \
        experiments/baseline_reproduction \
        experiments/rag_reproduction \
        experiments/map_elites_random_reproduction \
        experiments/llm_qd_full_reproduction \
    --output figures/convergence_curves.png \
    --metric "mean_fitness"
```

**Output:** `figures/convergence_curves.png` (line plot)

### 2.8 Figure 8: Ablation Study Results

```bash
python scripts/plot_ablation_study.py \
    --experiments \
        experiments/map_elites_random_reproduction \
        experiments/llm_qd_no_rag \
        experiments/llm_qd_no_llm \
        experiments/llm_qd_full_reproduction \
    --output figures/ablation_study.png \
    --metrics coverage mean_fitness qd_score
```

**Output:** `figures/ablation_study.png` (grouped bar chart)

**Note:** Ablation experiments need to be run first:
```bash
# MAP-Elites + RAG (no LLM mutations)
python src/map_elites_experiment.py --mutation_type "random" --use_rag --output experiments/llm_qd_no_llm

# MAP-Elites + LLM (no RAG)
python src/map_elites_experiment.py --mutation_type "llm_guided" --no_rag --output experiments/llm_qd_no_rag
```

### 2.9 Generate All Figures at Once

```bash
bash scripts/generate_all_figures.sh
```

This script runs all visualization commands and outputs figures to `/figures/`.

---

## 3. Statistical Analysis

### 3.1 Significance Testing

```bash
python scripts/statistical_analysis.py \
    --baseline experiments/baseline_reproduction/final_population.json \
    --rag experiments/rag_reproduction/final_population.json \
    --output results/statistical_tests.json
```

**Output:** `results/statistical_tests.json` containing:
- t-test results (t-statistic, p-value)
- Cohen's d effect size
- Confidence intervals
- Mean differences

### 3.2 Diversity Metrics

```bash
python scripts/compute_diversity_metrics.py \
    --experiments \
        experiments/baseline_reproduction \
        experiments/rag_reproduction \
        experiments/map_elites_random_reproduction \
        experiments/llm_qd_full_reproduction \
    --output results/diversity_metrics.json
```

**Output:** `results/diversity_metrics.json` containing:
- Uniqueness rates
- Behavioral distances (pairwise)
- Coverage per dimension
- QD-Score

### 3.3 Efficiency Analysis

```bash
python scripts/analyze_efficiency.py \
    --trace_files \
        experiments/*/trace.json \
    --output results/efficiency_analysis.json
```

**Output:** `results/efficiency_analysis.json` containing:
- API call counts
- Token usage
- Estimated costs
- Runtime statistics
- Coverage per API call

---

## 4. Preparing for Submission

### 4.1 LaTeX Compilation

The paper is provided in Markdown format. To convert to LaTeX:

**Option 1: Pandoc (Automated)**
```bash
pandoc docs/LLM_QD_PAPER_DRAFT.md \
    -o paper/llm_qd_paper.tex \
    --template=templates/iclr2026.tex \
    --bibliography=references.bib \
    --citeproc
```

**Option 2: Manual Conversion**
- Use the ICLR 2026 LaTeX template
- Copy section content from Markdown
- Format according to ICLR style guide

**Compile:**
```bash
cd paper/
pdflatex llm_qd_paper.tex
bibtex llm_qd_paper
pdflatex llm_qd_paper.tex
pdflatex llm_qd_paper.tex
```

**Output:** `paper/llm_qd_paper.pdf`

### 4.2 Supplementary Materials

Create supplementary PDF with:
- Appendix A: Additional genome examples
- Appendix B: Complete algorithm pseudocode
- Appendix C: Full experimental data tables
- Appendix D: Additional figures (6 heatmap projections)
- Appendix E: Hyperparameters and configuration details

```bash
python scripts/generate_supplementary.py \
    --experiments experiments/*_reproduction \
    --output supplementary/supplementary.pdf
```

### 4.3 Code Release

Prepare code release repository:

```bash
# Create release branch
git checkout -b paper-submission-v1

# Clean up sensitive data
python scripts/clean_for_release.py

# Create release archive
git archive --format=zip --prefix=llm-qd-logo/ -o llm_qd_logo_code.zip HEAD

# Generate README for code release
python scripts/generate_code_readme.py > CODE_README.md
```

**What to include:**
- All source code (`src/`)
- Reproduction scripts (`scripts/`)
- Requirements (`requirements.txt`)
- Documentation (`docs/`)
- Example experiments (selected, not all)
- LICENSE file (MIT)

**What to exclude:**
- API keys and credentials
- Large experiment outputs (provide download link)
- ChromaDB databases (too large; provide scripts to rebuild)

### 4.4 Data Release

Prepare data archive:

```bash
# Archive experimental results
python scripts/archive_experiments.py \
    --experiments experiments/*_reproduction \
    --output data_release/experimental_results.tar.gz

# Archive generated logos (sample)
python scripts/sample_logos.py \
    --archives experiments/*/archive.json \
    --n_per_experiment 50 \
    --output data_release/sample_logos.zip

# Generate data documentation
python scripts/generate_data_documentation.py \
    --output data_release/DATA_README.md
```

**Upload to:**
- Zenodo (DOI for permanent archival)
- Google Drive (supplementary large files)
- GitHub Releases (code + small data)

---

## 5. Submission Checklist

### 5.1 Paper Content

- [ ] Abstract (200-250 words)
- [ ] Introduction with clear contributions
- [ ] Related work (20-30 citations)
- [ ] Methodology fully specified
- [ ] Experimental setup described
- [ ] Results with statistical tests
- [ ] Discussion of limitations
- [ ] Conclusion and future work
- [ ] References formatted (ICLR style)
- [ ] Acknowledgments
- [ ] 8 figures generated and captioned
- [ ] 12 data tables formatted
- [ ] Page limit: 8-10 pages (ICLR main, excluding refs)

### 5.2 Supplementary Materials

- [ ] Supplementary PDF (no page limit)
- [ ] Complete algorithm pseudocode
- [ ] Additional experimental data
- [ ] Extra visualizations (6+ heatmaps)
- [ ] Hyperparameter details
- [ ] Ablation study details
- [ ] Error analysis

### 5.3 Code and Data

- [ ] Code repository cleaned and documented
- [ ] README with reproduction instructions
- [ ] All dependencies listed (requirements.txt)
- [ ] Example scripts for each experiment
- [ ] Visualization scripts included
- [ ] LICENSE file (MIT recommended)
- [ ] Experimental results archived
- [ ] Sample logos provided (50-100 per experiment)
- [ ] Data documentation (DATA_README.md)
- [ ] Upload to permanent repository (Zenodo, GitHub)
- [ ] DOI obtained for citation

### 5.4 Experiments

- [ ] Baseline evolutionary (50 logos, ~90 fitness)
- [ ] RAG-enhanced (100 logos, ~92 fitness)
- [ ] MAP-Elites random (150 evals, 4% coverage)
- [ ] LLM-QD test (150 evals, 4% coverage, 87 fitness)
- [ ] **LLM-QD full-scale (700 evals, 10-30% coverage)** ← CRITICAL
- [ ] Ablation: MAP-Elites only
- [ ] Ablation: +LLM mutations
- [ ] Ablation: +RAG
- [ ] Ablation: Full system
- [ ] Multi-industry validation (10+ industries) ← RECOMMENDED
- [ ] Human evaluation study (20-30 designers) ← RECOMMENDED

### 5.5 Figures and Visualizations

- [ ] Figure 1: System architecture diagram
- [ ] Figure 2: 4D behavioral space (3D visualization)
- [ ] Figure 3: Coverage comparison (bar chart)
- [ ] Figure 4: Quality-diversity scatter plot
- [ ] Figure 5: Heatmap (complexity × style)
- [ ] Figure 6: Example logos (3×3 grid)
- [ ] Figure 7: Convergence curves (line plot)
- [ ] Figure 8: Ablation study (grouped bars)
- [ ] All figures high resolution (300 DPI for raster, vector for plots)
- [ ] All figures captioned and referenced in text

### 5.6 Statistical Analysis

- [ ] t-tests computed (RAG vs baseline, p-values)
- [ ] Effect sizes calculated (Cohen's d)
- [ ] Confidence intervals reported
- [ ] Significance level α = 0.05 stated
- [ ] Multiple testing correction if applicable
- [ ] Diversity metrics computed
- [ ] Efficiency metrics computed

### 5.7 Ethical Considerations

- [ ] Broader impact statement (positive and negative)
- [ ] Bias analysis (LLM training data biases)
- [ ] Job displacement discussion
- [ ] Mitigation strategies proposed
- [ ] Copyright and originality addressed
- [ ] Human evaluation IRB if applicable

### 5.8 Submission Requirements (ICLR 2026)

- [ ] Anonymous submission (no author names in paper)
- [ ] Code/data links anonymized (anonymous GitHub, anonymous Zenodo)
- [ ] Submitted via OpenReview
- [ ] Conflict of interest declared
- [ ] Author agreement signed
- [ ] Supplementary materials uploaded
- [ ] Submission deadline met (typically January)

### 5.9 Pre-Submission Review

- [ ] Internal review by co-authors
- [ ] Check for typos and grammar
- [ ] Verify all citations correct
- [ ] Ensure all figures render correctly
- [ ] Test code reproduction (fresh environment)
- [ ] Verify data downloads work
- [ ] Spell-check
- [ ] Grammar-check (Grammarly, LanguageTool)
- [ ] Consistency check (terminology, notation)

---

## 6. Timeline to Submission

### Week 1-2: Full-Scale Experiment
- [ ] Run LLM-QD full-scale (10×10×10×10 grid)
- [ ] Run ablation studies
- [ ] Validate 10-30% coverage hypothesis
- [ ] Compute all metrics

### Week 3-4: Human Evaluation (Optional but Recommended)
- [ ] Recruit 20-30 professional designers
- [ ] Create blind comparison survey
- [ ] Collect preference data
- [ ] Analyze results
- [ ] Add to paper

### Week 5: Multi-Industry Validation (Optional)
- [ ] Test on 10 diverse industries
- [ ] Validate generalization
- [ ] Add results to paper

### Week 6: Figure Generation and Statistical Analysis
- [ ] Generate all 8 main figures
- [ ] Run statistical tests
- [ ] Compute diversity metrics
- [ ] Create supplementary figures

### Week 7: Paper Writing and Formatting
- [ ] Convert Markdown to LaTeX
- [ ] Format according to ICLR template
- [ ] Add all figures and captions
- [ ] Write supplementary materials
- [ ] Internal review

### Week 8: Code and Data Release Preparation
- [ ] Clean code repository
- [ ] Write documentation
- [ ] Archive experiments
- [ ] Upload to Zenodo
- [ ] Test reproduction

### Week 9: Final Review and Submission
- [ ] Final proofreading
- [ ] Check submission requirements
- [ ] Anonymize submission
- [ ] Upload to OpenReview
- [ ] Submit before deadline

**Total Time:** ~9 weeks (can be compressed to 5-6 weeks if skipping optional items)

---

## 7. Contact and Support

**Questions about experiments:**
- Check `/docs/LLM_QD_ARCHITECTURE.md` for system details
- Check `/docs/REPORTE_RESULTADOS_ESPAÑOL.md` for baseline results
- Email: luis@guanacolabs.com

**Questions about code:**
- GitHub Issues: https://github.com/larancibia/svg-logo-ai/issues
- See `/src/README.md` for module documentation

**Questions about paper:**
- See `/docs/LLM_QD_PAPER_DRAFT.md` for full draft
- Email for collaboration inquiries

---

## 8. Additional Resources

### 8.1 Related Work Papers (for Citation)
- EvoPrompt (ICLR 2024): LLM-guided evolution
- LLMatic (2024): LLM + QD for NAS
- MEliTA (2024): MAP-Elites for images
- MAP-Elites (2015): Original QD algorithm
- RAG (2020): Retrieval-augmented generation

### 8.2 Venue Information

**ICLR 2026 (International Conference on Learning Representations)**
- Website: https://iclr.cc/
- Submission deadline: ~January 2026
- Notification: ~April 2026
- Conference: ~May 2026
- Page limit: 8 pages main + unlimited appendix
- Review process: Double-blind
- Acceptance rate: ~25-30%

**GECCO 2026 (Genetic and Evolutionary Computation Conference)**
- Website: https://gecco-2026.sigevo.org/
- Submission deadline: ~January 2026
- Notification: ~March 2026
- Conference: ~July 2026
- Page limit: 8 pages
- Review process: Double-blind
- Acceptance rate: ~35-40%

### 8.3 Alternative Venues

If ICLR/GECCO are too competitive or timing doesn't work:

**Top-Tier Alternatives:**
- ICML 2026 (International Conference on Machine Learning)
- NeurIPS 2026 (Neural Information Processing Systems)
- IJCAI 2026 (International Joint Conference on AI)

**Domain-Specific:**
- IEEE CEC 2026 (Congress on Evolutionary Computation)
- AAAI 2026 (Association for Advancement of AI)
- CVPR 2026 (Computer Vision and Pattern Recognition) - if framed as visual generation

**Journals (Longer Timeline but Archival):**
- Evolutionary Computation (MIT Press)
- IEEE Transactions on Evolutionary Computation
- Artificial Intelligence Journal
- Journal of Machine Learning Research (JMLR)

---

## 9. Version History

**v1.0 (November 27, 2025)**
- Initial paper draft completed
- Baseline and RAG experiments validated
- Test-scale LLM-QD experiment completed
- Figures described (not yet generated)
- Estimated 75% submission-ready

**TODO for v2.0 (Full Submission-Ready):**
- [ ] Full-scale LLM-QD experiment (10×10×10×10)
- [ ] Human evaluation study
- [ ] All figures generated
- [ ] LaTeX formatting complete
- [ ] Code and data released

---

**END OF PAPER README**
