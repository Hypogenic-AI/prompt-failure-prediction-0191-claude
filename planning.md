# Research Plan: Predicting Prompt Engineering Failure Modes via Attention Circuit Geometry

## Motivation & Novelty Assessment

### Why This Research Matters
Prompt engineering failures (76+ accuracy point swings from formatting alone, per Sclar 2023) cause safety and reliability issues in production LLM deployments. Current practice relies on expensive trial-and-error testing. A predictive geometric signature would enable pre-deployment failure detection, saving time and preventing harmful outputs.

### Gap in Existing Work
- Ricci curvature of attention graphs linked to model-level robustness (ICLR 2025), but NOT to prompt-level failure prediction
- Attention head stability shows U-shaped layer pattern (Bali 2026), but hasn't been used for prompt failure prediction
- Prompt sensitivity work (Sclar, Mizrahi) documents the problem but provides no mechanistic explanation
- **No existing work connects attention circuit geometry to prompt failure prediction** — this is the core gap

### Our Novel Contribution
We propose that geometric features of attention patterns (curvature, singular value spectrum, entropy, stability) computed at inference time can predict which prompt formulations will fail, bridging mechanistic interpretability and prompt robustness.

### Experiment Justification
- **Experiment 1 (Geometric Feature Extraction)**: Extract attention geometry features to establish the measurement framework
- **Experiment 2 (Correlation Analysis)**: Test whether geometric features correlate with prompt performance variance
- **Experiment 3 (Failure Prediction Classifier)**: Build and evaluate a classifier using geometric features to predict prompt failures

## Research Question
Can geometric properties of attention patterns in transformer models predict which prompt formulations will produce failures (low accuracy) before deployment?

## Hypothesis Decomposition
H1: Attention patterns vary geometrically across prompt formats in measurable ways
H2: Geometric features (curvature, singular values, entropy) correlate with prompt performance
H3: A classifier using geometric features outperforms baselines (random, logit confidence, embedding similarity) at predicting prompt failures

## Proposed Methodology

### Approach
Use GPT-2 Small (well-studied circuits) via TransformerLens. For SST-2 sentiment classification:
1. Create 20+ prompt format variants (systematic perturbations)
2. Measure accuracy of each format on validation set
3. Extract attention patterns and compute geometric features
4. Correlate features with performance; train failure predictor

### Experimental Steps
1. **Data Preparation**: Load SST-2, create prompt format variants
2. **Performance Measurement**: Run each format through GPT-2, measure accuracy
3. **Feature Extraction**: For each prompt format × example, extract:
   - Attention entropy per head per layer
   - Singular value spectrum of attention matrices
   - Attention pattern variance across positions
   - Frobenius norm of attention differences between formats
   - Mid-layer head stability (cosine similarity)
   - Effective rank of attention matrices
4. **Analysis**: Correlate geometric features with performance
5. **Classifier**: Train logistic regression/random forest to predict failure from geometry
6. **Validation**: Hold-out evaluation, cross-validation

### Baselines
- Random prediction
- Logit confidence (max logit value as proxy for stability)
- Prompt embedding cosine similarity to best-performing prompt

### Evaluation Metrics
- Spearman correlation between geometric features and accuracy
- AUC-ROC for binary failure detection (accuracy < median = failure)
- Precision/Recall/F1 for failure prediction
- R² for continuous performance prediction

### Statistical Analysis Plan
- Spearman rank correlation with bootstrap confidence intervals
- Paired t-test / Wilcoxon for classifier vs baselines
- 5-fold cross-validation for classifier evaluation
- Bonferroni correction for multiple comparisons

## Expected Outcomes
- H1 likely supported (attention patterns clearly differ across prompts)
- H2 partially supported (some geometric features should correlate, but may be noisy)
- H3 uncertain (depends on signal strength; may beat random but possibly not logit confidence)

## Timeline (within ~60 min)
- Setup + Data Prep: 10 min
- Performance Measurement: 15 min
- Feature Extraction: 15 min
- Analysis + Classifier: 10 min
- Documentation: 10 min

## Potential Challenges
- GPT-2 Small may not show strong prompt sensitivity on SST-2 (mitigate: use diverse formats)
- Geometric features may be noisy (mitigate: aggregate across examples)
- Limited prompt variants (mitigate: systematic grammar-based generation)

## Success Criteria
- At least one geometric feature shows |ρ| > 0.3 correlation with prompt performance
- Failure classifier achieves AUC > 0.6 (above random baseline of 0.5)
- Results are reproducible across random seeds
