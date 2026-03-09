# Predicting Prompt Engineering Failure Modes via Attention Circuit Geometry

Investigating whether geometric properties of transformer attention patterns can predict which prompt formats will fail before deployment.

## Key Findings

- **21% accuracy spread** across 20 prompt formats on GPT-2 Small / SST-2 (0.500 to 0.710)
- **6 geometric features** significantly correlate with prompt performance (p < 0.05):
  - Attention Frobenius norm (ρ = +0.474, p = 0.035)
  - Cross-layer variance spread (ρ = -0.472, p = 0.036)
  - Mean attention variance (ρ = -0.460, p = 0.041)
- **Permutation test confirms** top correlation is non-spurious (p = 0.036, 5000 permutations)
- **Logit confidence baseline completely fails** — geometric features provide information that output confidence does not
- **Classifier limited by small n** (20 formats): best LOO accuracy = 60% vs 50% random

## Project Structure

```
├── REPORT.md              # Full research report with methodology and results
├── planning.md            # Research plan and motivation
├── src/
│   ├── experiment.py      # Main experiment: feature extraction + classification
│   └── analysis.py        # Follow-up: feature selection, bootstrap CIs, permutation tests
├── results/
│   ├── final_results.json # Complete numerical results
│   ├── analysis_results.json  # Follow-up analysis results
│   ├── feature_data.csv   # Per-format features and accuracy
│   └── figures/           # All visualizations (11 plots)
├── datasets/              # SST-2, AG News, MMLU (pre-downloaded)
├── papers/                # 34 research papers
├── code/                  # TransformerLens, ACDC, PromptBench, pyvene
├── literature_review.md   # Comprehensive literature review
└── resources.md           # Resource catalog
```

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add torch transformers transformer-lens numpy scipy scikit-learn matplotlib pandas seaborn datasets

# Run experiments
export USER=researcher TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache
python src/experiment.py   # ~5 min on GPU
python src/analysis.py     # ~10 sec
```

Requires: Python 3.10+, CUDA GPU recommended (CPU possible but slow).

## See Also

- [REPORT.md](REPORT.md) for full methodology, results, and analysis
- [planning.md](planning.md) for research motivation and experimental design
