# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT
committed to git due to size. Follow the download instructions below.

## Dataset 1: SST-2 (Stanford Sentiment Treebank v2)

### Overview
- **Source**: HuggingFace `glue/sst2`
- **Size**: 872 validation samples
- **Format**: HuggingFace Dataset
- **Task**: Binary sentiment classification
- **Splits**: validation (872)
- **License**: CC-BY-4.0

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("glue", "sst2", split="validation")
dataset.save_to_disk("datasets/sst2")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/sst2")
# Fields: sentence (str), label (int: 0=negative, 1=positive), idx (int)
```

### Notes
- Most commonly used dataset in prompt sensitivity studies (Sclar 2023, Mizrahi 2024)
- Performance can vary 0-76% accuracy depending on prompt format (Sclar 2023)
- Good for initial experiments due to simplicity and well-studied behavior

## Dataset 2: AG News

### Overview
- **Source**: HuggingFace `ag_news`
- **Size**: 1000 test samples (subset)
- **Format**: HuggingFace Dataset
- **Task**: 4-way topic classification (World, Sports, Business, Sci/Tech)
- **License**: Public domain

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("ag_news", split="test[:1000]")
dataset.save_to_disk("datasets/agnews")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/agnews")
# Fields: text (str), label (int: 0-3)
```

### Notes
- Used in PromptBench (Zhu 2023) as a text classification benchmark
- 4 classes provides more nuanced prompt sensitivity patterns than binary

## Dataset 3: MMLU (Abstract Algebra subset)

### Overview
- **Source**: HuggingFace `cais/mmlu` (abstract_algebra)
- **Size**: 100 test samples
- **Format**: HuggingFace Dataset
- **Task**: Multiple-choice knowledge questions
- **License**: MIT

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("cais/mmlu", "abstract_algebra", split="test")
dataset.save_to_disk("datasets/mmlu_sample")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/mmlu_sample")
# Fields: question (str), choices (list[str]), answer (int)
```

### Notes
- MMLU is widely used in prompt sensitivity studies
- Abstract algebra subset chosen as representative sample
- For full experiments, more MMLU subjects can be downloaded

## Additional Datasets (Not Downloaded - Available Online)

### PromptBench Datasets
- Available via the PromptBench library (code/promptbench)
- Includes: SST-2, CoLA, QQP, MNLI, QNLI, RTE, MRPC, WNLI, MMLU, SQuAD, IWSLT, UN Multi, Math
- Install: `pip install promptbench`

### Synthetic HMM Data (Mess3)
- Generated programmatically for belief geometry experiments
- Parameters: α (emission clarity), x (state persistence)
- See Piotrowski et al. 2025 for generation code

### IOI Dataset
- Indirect Object Identification sentences for circuit analysis
- Generated via templates: "When Mary and John went to the store, John gave a drink to"
- Available in TransformerLens library

## Usage Notes for Experiment Runner

1. **Primary experiment**: Use SST-2 with multiple prompt templates (following Sclar 2023 methodology)
2. **Validation**: Replicate on AG News and MMLU
3. **Circuit analysis**: Use IOI dataset (generated) and synthetic HMM data
4. **Prompt variations**: Use PromptBench library for systematic perturbations
