# Literature Review: Predicting Prompt Engineering Failure Modes via Attention Circuit Geometry

## Research Area Overview

This research sits at the intersection of three active areas: (1) **mechanistic interpretability** of transformer models, particularly circuit-level analysis of attention heads; (2) **prompt sensitivity/robustness**, understanding why small prompt changes cause dramatic performance shifts; and (3) **geometric analysis of neural representations**, studying the manifold structure of attention patterns and hidden states. Our hypothesis posits that the geometric structure of attention circuits contains predictive signatures for prompt failures—specifically, that high curvature or discontinuities in the attention pattern manifold correspond to unstable regions.

## Key Papers

### 1. Constrained Belief Updates Explain Geometric Structures in Transformer Representations
- **Authors**: Piotrowski, Riechers, Filan, Shai (2025, ICML)
- **ArXiv**: 2502.01954
- **Key Contribution**: Shows transformers implement *constrained Bayesian belief updating*—a parallelized approximation of Bayesian inference shaped by architectural constraints. The geometry of internal representations is a mathematical signature of how constraints warp optimal inference.
- **Methodology**: Trains single-layer transformers on HMM-generated data (Mess3 family). Uses PCA of residual stream activations, spectral decomposition of transition matrices, and mechanistic interpretability of OV/QK circuits.
- **Key Findings**:
  - Intermediate representations (post-attention, pre-MLP) form **fractal structures** in the belief simplex
  - Attention weights decay as eigenvalues raised to the power of positional distance: A_{d,s} ∝ ζ^{d-s}
  - Negative eigenvalues require multiple attention heads (non-negativity constraint)
  - Single-head models show **incomplete representations** and degraded performance when eigenvalues are negative
  - MLP transforms constrained fractals into full Bayesian belief geometry
- **Datasets**: Synthetic data from Mess3 HMMs with parameters α (emission clarity) and x (state persistence)
- **Code Available**: Referenced but not linked explicitly
- **Relevance**: **Critical.** Provides the theoretical framework for understanding attention geometry. The constrained belief update equation (Eq. 5) defines exactly what geometry attention "should" produce—deviations from this geometry may predict failures. The breakdown of the approximation at extreme parameter values directly maps to our hypothesis about curvature/discontinuity causing instability.

### 2. Towards Automated Circuit Discovery for Mechanistic Interpretability (ACDC)
- **Authors**: Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso (2023)
- **ArXiv**: 2304.14997, Citations: 494
- **Key Contribution**: Systematizes mechanistic interpretability by automating circuit discovery. ACDC algorithm identified 68/32,000 edges in GPT-2 Small for the Greater-Than circuit.
- **Methodology**: Activation patching on computational graph edges. Iteratively removes edges that don't significantly affect task performance.
- **Code Available**: https://github.com/ArthurConmy/Automatic-Circuit-Discovery
- **Relevance**: Provides tools for identifying which attention heads form circuits for specific tasks. Essential for extracting the circuits whose geometry we want to analyze.

### 3. Interpretability in the Wild: Circuit for Indirect Object Identification in GPT-2 Small
- **Authors**: Wang et al. (2022)
- **ArXiv**: 2211.00593, Citations: 854
- **Key Contribution**: Identified the complete circuit for IOI task in GPT-2, including name movers, backup name movers, induction heads, and S-inhibition heads.
- **Methodology**: Causal tracing, activation patching, knock-out experiments.
- **Relevance**: The IOI circuit is the canonical example for circuit analysis. Understanding how this circuit's geometry changes with prompt perturbations is a natural starting point for our research.

### 4. In-context Learning and Induction Heads
- **Authors**: Olsson et al. (2022)
- **ArXiv**: 2209.11895, Citations: 763
- **Key Contribution**: Identifies induction heads as a key mechanism for in-context learning. Shows a phase change during training.
- **Relevance**: Induction heads form a fundamental circuit type. Their geometric behavior under prompt variation is a natural target for our analysis.

### 5. Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design
- **Authors**: Sclar et al. (2023, ICLR 2024)
- **ArXiv**: 2310.11324
- **Key Contribution**: Proposes FORMATSPREAD, an algorithm using Thompson Sampling to efficiently search the space of meaning-preserving prompt format variations. Finds up to **76 accuracy point** swings from formatting alone.
- **Methodology**: Defines a formal grammar over prompt formats (separators, spacing, casing, enumerations). Uses multi-armed bandit to identify best/worst formats. Tests with 10-320 formats per task.
- **Key Findings**:
  - Median spread of 7.5 accuracy points; 20% of tasks show 15+ point spreads
  - Sensitivity NOT eliminated by scaling, few-shot, or instruction tuning
  - Format performance only weakly correlates between models — rankings frequently reversed
  - Accuracy landscape is highly non-monotonic (~33% monotonicity, same as random)
  - Single atomic changes (adding/removing a space) can swing accuracy by 78+ points
  - Prompt format embeddings are highly separable (0.98+ classification accuracy)
- **Datasets**: 53 tasks from Super-NaturalInstructions (1000 samples per task)
- **Models**: LLaMA-2 (7B/13B/70B), Falcon-7B, GPT-3.5-Turbo
- **Relevance**: **Critical.** The formal grammar over formats provides a systematic perturbation space. The non-monotonicity finding suggests the performance landscape has complex geometry. Embedding separability suggests geometric features ARE predictive.

### 6. State of What Art? A Call for Multi-Prompt LLM Evaluation (Prompt Fragility)
- **Authors**: Mizrahi et al. (2024, TACL)
- **ArXiv**: 2402.10679
- **Key Contribution**: Large-scale study (6.5M instances) of prompt paraphrasing effects on evaluation reliability. Proposes CPS metric combining peak capability with robustness.
- **Key Findings**:
  - 10/25 tasks show weak ranking agreement across prompts (Kendall's W < 0.55)
  - 15/25 tasks had anti-correlated prompt pair rankings
  - Single-word changes ("excludes" vs "lacks") cause 28-46% swings
  - Edit distance does NOT predict degradation magnitude
  - 72.5% of the time, original benchmark prompts outperform average (selection bias)
- **Datasets**: LMentry (10 tasks), BIG-bench Lite (14 tasks), BIG-bench Hard (15 tasks)
- **Models**: 20 LLMs total
- **Metrics**: MaxP, AvgP, CPS (Combined Performance Score), Kendall's W
- **Relevance**: Establishes the problem we aim to solve. The finding that edit distance doesn't predict degradation strongly motivates our geometric approach — we need features beyond surface-level prompt similarity.

### 7. PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts
- **Authors**: Zhu et al. (2023)
- **ArXiv**: 2306.04528
- **Key Contribution**: Comprehensive benchmark for prompt robustness including character-level, word-level, sentence-level, and semantic-level perturbations.
- **Code Available**: https://github.com/microsoft/promptbench
- **Relevance**: Provides systematic perturbation methods and evaluation framework we can use.

### 8. The Geometry of Attention: Ricci Curvature and Transformers Training and Robustness
- **Source**: OpenReview (ICLR 2025 submission)
- **Key Contribution**: **The most directly related prior work.** Treats attention maps as weighted graphs and uses Ollivier-Ricci curvature (ORC) to characterize their geometry. Establishes theoretical links between Ricci curvature, gradient descent convergence, and transformer robustness.
- **Methodology**: Computes ORC on attention graph edges using 1-Wasserstein distance. Links minimum eigenvalue of attention matrix to convergence via PL condition. Proposes variance-based regularization to manipulate curvature.
- **Key Findings**:
  - ORC distribution shifts toward more positive values through forward pass (first→last layer)
  - Higher positive Ricci curvature → more robust transformers (via entropy/Fluctuation Theorem)
  - Negative ORC corresponds to weight imbalance (unstable attention patterns)
  - Variance of attention scores is a computationally efficient proxy for ORC
  - Performance-robustness trade-off: decreasing variance improves performance but reduces generalizability
- **Lemma 4.2**: Minimum attention eigenvalue is positively correlated with probability of exponential gradient descent convergence
- **Models**: L-ViT, BERT-Tiny
- **Datasets**: MNIST, Fashion-MNIST, CIFAR10, SST-2, IMDb, Headlines
- **Relevance**: **Critical.** Establishes curvature→robustness link at model level. Our contribution extends this to **prompt-level** failure prediction at **inference time** (not training time). Key insight: negative ORC edges = unstable attention patterns = potential failure points. We can compute ORC of attention graphs for specific prompts to predict failures.

### 9. Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits
- **Authors**: Bussman et al. (2025)
- **ArXiv**: 2501.11029
- **Key Contribution**: SVD-based analysis of transformer circuits, decomposing attention heads into singular vector components.
- **Relevance**: SVD provides a natural way to measure the "curvature" or complexity of attention patterns—high singular value spread could indicate instability.

### 10. Circuit Fingerprints: How Answer Tokens Encode Their Geometrical Path (2026)
- **Authors**: Saurez, Sengar, Har (KAIST)
- **ArXiv**: 2602.09784
- **Key Contribution**: Proposes that circuit discovery (reading) and activation steering (writing) are dual operations on the same geometric structure. Answer tokens processed in isolation trace the same computational pathways that produce them. Uses Shapley decomposition of Q/K/V channels for principled circuit scoring.
- **Key Findings**:
  - Gradient-free geometric circuit discovery matches gradient-based methods (EAP, EAP-IG)
  - Q/K/V Shapley values reveal functional head clustering (Name Movers are Q-dominated, S-Inhibition heads are K-dominated)
  - **Negative-valence steering is fragile**—causes semantic degradation, suggesting geometric regions of instability
  - Contrastive direction δr = r(a+) - r(a-) defines task-relevant geometric directions
- **Models**: GPT-2 Small, Qwen2.5-0.5B, Llama 3.2-1B, OPT-1.3B
- **Relevance**: **Critical.** The fragility of steering along certain geometric directions directly demonstrates that circuit geometry has unstable regions. The contrastive direction methodology (success vs failure prompts) is directly applicable to our failure prediction approach. Gradient-free method enables scalable deployment.

### 11. Quantifying LLM Attention-Head Stability: Implications for Circuit Universality (2026)
- **Authors**: Bali, Stanley, Suresh, Bzdok (Mila / McGill)
- **ArXiv**: 2602.16740
- **Key Contribution**: Measures attention head stability across 50 random seed refits of GPT-like models. Finds a **U-shaped stability curve** across layers—middle layers are least stable. Unstable mid-layer heads are paradoxically the most functionally important. Stability degrades with prompt length.
- **Methodology**: Cosine similarity of flattened attention score matrices across refits, CKA on residual streams, ablation-based importance, meta-SNE visualization.
- **Key Findings**:
  - Middle-layer heads show stability as low as 0.70 (vs ~1.0 for early/late layers)
  - AdamW improves stability; query-weight Frobenius norm correlates negatively with stability
  - Longer prompts (5→50 tokens) sharply degrade stability
  - Residual stream is more stable than individual attention heads
- **Datasets**: C4 (2B tokens), OpenWebText (9B tokens)
- **Code**: https://github.com/karanbali/attention_head_seed_stability
- **Relevance**: **Critical.** The stability-dip in middle layers provides a geometric signature for prompt fragility. The finding that prompt length degrades stability directly supports our hypothesis. Query-weight Frobenius norm offers a weight-space geometric measure computable without multiple refits.

### 12. Not All Language Model Features Are Linear (Engels et al., 2024)
- **ArXiv**: 2405.14860
- **Key Contribution**: Shows that some features in LLMs are represented as multi-dimensional, non-linear structures, not just linear directions.
- **Relevance**: Challenges the linear representation hypothesis, suggesting that attention circuit geometry may have complex, non-linear manifold structure relevant to our curvature analysis.

### 13. The Linear Representation Hypothesis and the Geometry of Large Language Models
- **Authors**: Park et al. (2024)
- **ArXiv**: 2310.16764
- **Key Contribution**: Analyzes when and how linear representations emerge in LLMs.
- **Relevance**: Understanding the geometry of representations helps characterize the manifold structure we want to analyze.

## Common Methodologies

1. **Activation Patching / Causal Tracing**: Used in IOI circuit, ACDC, and most circuit discovery work. Identifies which components contribute to specific behaviors.
2. **PCA/SVD of Activations**: Used in belief geometry work, representation analysis. Reveals geometric structure.
3. **Spectral Analysis**: Eigenvalue decomposition of attention patterns and transition matrices. Key method in Piotrowski et al.
4. **Prompt Perturbation**: Systematic variation of prompt formats, used in Sclar, Mizrahi, PromptBench.

## Standard Baselines

- Random prompt selection (no prediction of failure)
- Prompt ensembling (averaging over multiple prompts)
- Simple heuristics (prompt length, keyword presence)
- Logit-based confidence measures

## Evaluation Metrics

- **Accuracy variance** across prompt perturbations (Sclar 2023)
- **MSE to theoretical geometry** (Piotrowski 2025)
- **Circuit faithfulness** (fraction of model performance recovered by circuit)
- **Attention pattern stability** (cosine similarity across perturbations)
- **Curvature measures**: Ricci curvature, sectional curvature of attention manifold
- **Singular value spectrum** of attention matrices

## Datasets in the Literature

- **SST-2**: Sentiment analysis, most commonly used for prompt sensitivity (Sclar, Mizrahi)
- **GLUE/SuperGLUE tasks**: Standard NLP benchmarks for evaluating prompt effects
- **AG News**: Text classification, used in PromptBench
- **MMLU**: Multi-task benchmark, used in prompt fragility studies
- **Synthetic HMM data**: Used in belief geometry work (Mess3)
- **IOI dataset**: Indirect Object Identification prompts for circuit analysis

## Gaps and Opportunities

1. **No existing work directly connects attention circuit geometry to prompt failure prediction** — this is our core contribution
2. **Curvature analysis of attention manifolds is unexplored** — Piotrowski's work shows geometry exists but doesn't analyze curvature
3. **Spectral properties haven't been linked to prompt robustness** — eigenvalue structure is well-studied in isolation
4. **No predictive model exists** that uses geometric features to forecast prompt failures before deployment
5. **Bridge between prompt sensitivity (empirical) and circuit analysis (mechanistic) is missing**

## Recommendations for Our Experiment

- **Primary datasets**: SST-2 (most studied for prompt sensitivity), AG News (classification), MMLU subset (knowledge tasks)
- **Recommended baselines**: Random prediction, logit confidence, prompt similarity metrics
- **Recommended metrics**: Accuracy variance prediction (R²), prompt failure detection AUC, geometric curvature correlation with performance drop
- **Methodological approach**:
  1. Use TransformerLens to extract attention patterns for GPT-2 Small on various tasks
  2. Apply ACDC to identify task-specific circuits
  3. Compute geometric features (curvature, singular values, manifold properties) of attention patterns
  4. Vary prompts systematically (using Sclar/PromptBench methodology)
  5. Train a predictor mapping geometric features → prompt performance
  6. Evaluate whether geometric signatures predict failure modes
- **Key tools**: TransformerLens, ACDC, PromptBench, pyvene
