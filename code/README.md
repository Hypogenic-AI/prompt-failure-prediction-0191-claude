# Cloned Repositories

## Repo 1: TransformerLens
- **URL**: https://github.com/TransformerLensOrg/TransformerLens
- **Purpose**: Mechanistic interpretability library for transformer models. Provides tools to extract and analyze attention patterns, residual stream activations, and circuit components.
- **Location**: code/TransformerLens/
- **Key files**:
  - `transformer_lens/HookedTransformer.py` - Main model class with hooks for all components
  - `transformer_lens/utils.py` - Utility functions for analysis
  - `demos/` - Example notebooks for various analyses
- **Installation**: `pip install transformer-lens`
- **Notes**: Essential tool for this research. Enables:
  - Extracting attention patterns for any prompt
  - Computing activation patching / causal interventions
  - Analyzing OV and QK circuits
  - Loading GPT-2 Small and other models with interpretability hooks

## Repo 2: ACDC (Automatic Circuit Discovery)
- **URL**: https://github.com/ArthurConmy/Automatic-Circuit-Discovery
- **Purpose**: Automated discovery of computational circuits in transformers. Identifies the minimal subgraph of attention heads and MLPs that implement a specific behavior.
- **Location**: code/ACDC/
- **Key files**:
  - `acdc/` - Main ACDC algorithm implementation
  - `experiments/` - Pre-built experiments for IOI, greater-than, etc.
- **Notes**: Use to identify task-specific circuits whose geometry we want to analyze. Can discover circuits for prompt-sensitive tasks automatically.

## Repo 3: PromptBench
- **URL**: https://github.com/microsoft/promptbench
- **Purpose**: Comprehensive benchmark for evaluating LLM robustness to prompt perturbations. Provides systematic methods for generating prompt variations.
- **Location**: code/promptbench/
- **Key files**:
  - `promptbench/prompts/` - Prompt templates and perturbation methods
  - `promptbench/models/` - Model wrappers
  - `promptbench/metrics/` - Evaluation metrics
- **Installation**: `pip install promptbench`
- **Notes**: Provides the prompt perturbation methodology we need. Supports character-level, word-level, sentence-level, and semantic-level perturbations. Compatible with multiple models.

## Repo 4: pyvene
- **URL**: https://github.com/stanfordnlp/pyvene
- **Purpose**: Library for causal interventions on neural networks. Enables activation patching, interchange interventions, and other causal methods.
- **Location**: code/pyvene/
- **Key files**:
  - `pyvene/models/` - Intervention model wrappers
  - `tutorials/` - Usage examples
- **Installation**: `pip install pyvene`
- **Notes**: Useful for causal analysis of attention circuits. Can perform targeted interventions to test whether specific geometric features of attention are causally related to prompt failures.

## How These Tools Connect to Our Research

1. **TransformerLens** → Extract attention patterns and circuit activations for different prompts
2. **ACDC** → Identify which attention heads form task-specific circuits
3. **PromptBench** → Generate systematic prompt perturbations to test robustness
4. **pyvene** → Perform causal interventions to validate geometric features are causal

### Experimental Pipeline
```
Prompt → TransformerLens → Extract attention patterns
                         → ACDC → Identify circuit
                         → Compute geometric features (curvature, singular values)
Perturbed Prompt → TransformerLens → Extract attention patterns
                                   → Compare geometry with original
                                   → Measure performance change
                                   → Train predictor: geometry → failure
```
