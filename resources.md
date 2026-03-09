# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Predicting Prompt Engineering Failure Modes via Attention Circuit Geometry." Resources include papers, datasets, and code repositories.

## Papers
Total papers downloaded: 34

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Constrained Belief Updates Explain Geometric Structures | Piotrowski et al. | 2025 | constrained_belief_geometric_2025.pdf | **Core**: attention geometry = constrained Bayesian belief |
| Towards Automated Circuit Discovery (ACDC) | Conmy et al. | 2023 | conmy2023_automated_circuit_discovery.pdf | Automated circuit discovery tool |
| IOI Circuit in GPT-2 | Wang et al. | 2022 | wang2022_ioi_circuit.pdf | Canonical circuit analysis |
| In-context Learning and Induction Heads | Olsson et al. | 2022 | olsson2022_induction_heads.pdf | Fundamental attention mechanism |
| GPT-2 Greater-Than | Hanna et al. | 2023 | hanna2023_gpt2_greater_than.pdf | Circuit analysis methodology |
| Practical Review of Mech Interp | Ferrando et al. | 2024 | ferrando2024_practical_review_mech_interp.pdf | Survey of methods |
| Circuit Component Reuse | Merullo et al. | 2023 | merullo2023_circuit_component_reuse.pdf | Cross-task circuits |
| Successor Heads | Gould et al. | 2023 | gould2023_successor_heads.pdf | Interpretable attention heads |
| Semantic Induction Heads | Wang et al. | 2024 | wang2024_semantic_induction_heads.pdf | ICL mechanism |
| Circuit Consistency Across Scale | Biran et al. | 2024 | biran2024_circuit_consistency_scale.pdf | Scaling behavior of circuits |
| Interpreting Attention with SAEs | Kissane et al. | 2024 | kissane2024_interpreting_attention_sae.pdf | SAE-based attention analysis |
| Chain-of-Thought Mechanistic | Jin et al. | 2024 | jin2024_chain_of_thought_mechanistic.pdf | CoT circuit analysis |
| Progress Measures for Grokking | Nanda et al. | 2023 | nanda2023_progress_measures_grokking.pdf | Training dynamics |
| SVD-Based Circuit Interpretability | Bussman et al. | 2025 | bussman2025_singular_vector_interpretability.pdf | SVD decomposition of circuits |
| Causal Head Gating | DeBenedetti et al. | 2025 | debenedetti2025_causal_head_gating.pdf | Head importance framework |
| Talking Heads | He et al. | 2024 | he2024_talking_heads.pdf | Inter-layer communication |
| Stacked Attention Heads | Todd et al. | 2024 | todd2024_stacked_attention_heads.pdf | Multi-layer attention |
| SAEs Enable Circuit Identification | Marks et al. | 2024 | marks2024_sparse_autoencoders_circuits.pdf | Scalable circuit methods |
| Attribution Patching | Syed et al. | 2024 | syed2024_attribution_patching.pdf | Efficient activation patching |
| Attention Head Stability | Bali et al. | 2026 | attention_head_stability_2026.pdf | **Core**: U-shaped stability curve, code at github.com/karanbali/attention_head_seed_stability |
| Circuit Fingerprints | - | 2026 | circuit_fingerprints_2026.pdf | **Core**: geometric path encoding |
| Geometry of Attention | Park et al. | 2024 | park2024_geometry_of_attention.pdf | **Core**: formal attention geometry |
| Prompt Format Sensitivity | Sclar et al. | 2023 | sclar2023_prompt_format_sensitivity.pdf | **Core**: prompt sensitivity analysis |
| Prompt Fragility | Mizrahi et al. | 2024 | mizrahi2024_state_of_prompt_fragility.pdf | Multi-prompt evaluation |
| PromptBench | Zhu et al. | 2023 | zhu2023_promptbench.pdf | Prompt robustness benchmark |
| Prompt Sensitivity | Lu et al. | 2023 | lu2023_prompt_sensitivity.pdf | Sensitivity vs robustness |
| Prompt Sensitivity NLU | Webson et al. | 2022 | webson2022_prompt_sensitivity_nlu.pdf | NLU prompt analysis |
| Mind Your Format | Voronov et al. | 2024 | voronov2024_mind_your_format.pdf | Format impact on performance |
| Quantifying Prompt Sensitivity | Yang et al. | 2024 | yang2024_prompt_sensitivity_quantifying.pdf | Sensitivity quantification |
| ProSA | - | 2024 | prosa_2024_prompt_sensitivity.pdf | Prompt sensitivity assessment |
| Not All Features Linear | Engels et al. | 2024 | engels2024_not_all_features_linear.pdf | Non-linear features in LLMs |
| Linear Representation Hypothesis | Park et al. | 2024 | park2024_linear_representation_hypothesis.pdf | Representation geometry |
| Geometry of Hidden Reps | Valeriani et al. | 2023 | valeriani2023_geometry_of_hidden_reps.pdf | Intrinsic dimensionality |
| Representation Geometry | Ethayarajh | 2022 | ethayarajh2022_representation_geometry.pdf | Contextual embedding geometry |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| SST-2 | HuggingFace glue/sst2 | 872 val | Sentiment classification | datasets/sst2/ | Primary prompt sensitivity dataset |
| AG News | HuggingFace ag_news | 1000 test | Topic classification | datasets/agnews/ | Used in PromptBench |
| MMLU (Abstract Algebra) | HuggingFace cais/mmlu | 100 test | Multiple choice QA | datasets/mmlu_sample/ | Knowledge task sample |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | Mechanistic interpretability toolkit | code/TransformerLens/ | Primary analysis tool |
| ACDC | github.com/ArthurConmy/Automatic-Circuit-Discovery | Automated circuit discovery | code/ACDC/ | Circuit identification |
| PromptBench | github.com/microsoft/promptbench | Prompt robustness benchmark | code/promptbench/ | Prompt perturbation methods |
| pyvene | github.com/stanfordnlp/pyvene | Causal interventions | code/pyvene/ | Activation patching |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder with "attention head circuits transformer mechanistic interpretability" (diligent mode)
2. Searched arxiv for "prompt sensitivity robustness failure LLM" and "attention geometry manifold curvature transformer"
3. Used Semantic Scholar API to find correct arxiv IDs for papers from paper-finder results
4. Targeted specific papers from citations in key works

### Selection Criteria
- Papers directly addressing attention circuit geometry (highest priority)
- Papers on prompt sensitivity/robustness (needed for experimental framework)
- Papers on representation geometry in transformers (theoretical foundation)
- Established tools for mechanistic interpretability (practical implementation)

### Challenges Encountered
- Several arxiv ID mappings from paper-finder Semantic Scholar results were incorrect (different papers had those IDs). Required verification via Semantic Scholar API and manual checking.
- Some papers from 2026 have limited availability and few citations
- No existing dataset specifically designed for attention-circuit-geometry-to-prompt-failure mapping

### Gaps and Workarounds
- **No direct geometry→failure dataset**: Must be constructed during experiments by combining prompt perturbation data with attention pattern analysis
- **No curvature analysis tooling**: Must implement custom curvature computation on attention manifolds
- **Limited to GPT-2 scale**: Larger models require more compute; start with GPT-2 Small (well-studied circuit-wise)

## Recommendations for Experiment Design

1. **Primary dataset(s)**: SST-2 (most studied for prompt sensitivity, clear baseline from Sclar 2023)
2. **Model**: GPT-2 Small (extensively studied circuits: IOI, greater-than, induction heads)
3. **Baseline methods**:
   - Random prediction of prompt failure
   - Logit confidence as failure predictor
   - Prompt embedding similarity as stability proxy
4. **Evaluation metrics**:
   - Correlation between geometric features and performance variance (R²)
   - AUC for binary prompt failure detection
   - Spearman correlation of predicted vs actual performance ranking
5. **Code to adapt/reuse**:
   - TransformerLens for attention extraction (primary)
   - ACDC for circuit discovery
   - PromptBench for systematic prompt perturbation
   - Piotrowski et al.'s constrained belief framework (theoretical basis)
6. **Proposed geometric features to extract** (informed by literature):
   - Singular value spectrum of attention matrices per head (Bussman 2025)
   - Attention pattern entropy across positions
   - Frobenius norm difference between prompt variants
   - Query-weight Frobenius norm per layer (Bali 2026 — correlates with instability)
   - Mid-layer attention head cosine similarity stability (Bali 2026 — U-shaped curve)
   - Contrastive direction alignment: δr = r(success) - r(failure) projection onto circuit components (Circuit Fingerprints 2026)
   - Q/K/V Shapley channel decomposition to detect pathway misrouting (Circuit Fingerprints 2026)
   - Cosine similarity of OV circuit outputs across prompt variants
   - Eigenvalue structure of effective transition matrices (Piotrowski 2025)
   - CKA of residual stream activations across prompt variants (Bali 2026)
