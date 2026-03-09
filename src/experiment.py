"""
Predicting Prompt Engineering Failure Modes via Attention Circuit Geometry

Main experiment script:
1. Load GPT-2 Small via TransformerLens
2. Create diverse prompt format variants for SST-2
3. Measure accuracy of each format
4. Extract geometric features from attention patterns
5. Correlate features with performance
6. Train failure prediction classifier
"""

import os
import json
import random
import warnings
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.linalg import svdvals
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ─── Reproducibility ───
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─── Configuration ───
CONFIG = {
    'seed': SEED,
    'model_name': 'gpt2',  # GPT-2 Small (well-studied circuits)
    'dataset': 'sst2',
    'n_examples': 200,  # Examples per prompt format (balance speed vs signal)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'timestamp': datetime.now().isoformat(),
}

RESULTS_DIR = Path('results')
FIGURES_DIR = Path('results/figures')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Config: {json.dumps(CONFIG, indent=2)}")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {CONFIG['device']}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ═══════════════════════════════════════════════════════════════
# STEP 1: Load Model
# ═══════════════════════════════════════════════════════════════

def load_model():
    """Load GPT-2 Small with TransformerLens for attention extraction."""
    import transformer_lens
    print(f"\nTransformerLens loaded")
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        CONFIG['model_name'],
        device=CONFIG['device'],
    )
    model.eval()
    print(f"Model loaded: {CONFIG['model_name']}")
    print(f"  Layers: {model.cfg.n_layers}, Heads: {model.cfg.n_heads}, d_model: {model.cfg.d_model}")
    return model


# ═══════════════════════════════════════════════════════════════
# STEP 2: Create Prompt Format Variants
# ═══════════════════════════════════════════════════════════════

def create_prompt_formats():
    """
    Create diverse prompt format variants for SST-2 sentiment classification.
    Inspired by Sclar et al. (2023) systematic prompt perturbation.

    Each format is a function: (sentence, label_text) -> prompt_string
    For inference, we use the prompt without the label.
    """
    formats = {}

    # Format 1: Simple direct
    formats['simple_direct'] = {
        'template': 'Review: {text}\nSentiment: ',
        'pos_label': 'positive',
        'neg_label': 'negative',
    }

    # Format 2: Question style
    formats['question'] = {
        'template': 'Is the following review positive or negative?\n{text}\nAnswer: ',
        'pos_label': 'positive',
        'neg_label': 'negative',
    }

    # Format 3: Classification style
    formats['classify'] = {
        'template': 'Classify the sentiment of this text as positive or negative.\nText: {text}\nSentiment: ',
        'pos_label': 'positive',
        'neg_label': 'negative',
    }

    # Format 4: Minimal
    formats['minimal'] = {
        'template': '{text}\nThis is ',
        'pos_label': 'good',
        'neg_label': 'bad',
    }

    # Format 5: Formal
    formats['formal'] = {
        'template': 'Please determine the sentiment expressed in the following review.\n\nReview: "{text}"\n\nThe sentiment is: ',
        'pos_label': 'positive',
        'neg_label': 'negative',
    }

    # Format 6: Caps instruction
    formats['caps_instruction'] = {
        'template': 'SENTIMENT ANALYSIS\nInput: {text}\nOutput: ',
        'pos_label': 'positive',
        'neg_label': 'negative',
    }

    # Format 7: Numbered
    formats['numbered'] = {
        'template': '1. Read the review.\n2. Determine sentiment.\n\nReview: {text}\nSentiment (positive/negative): ',
        'pos_label': 'positive',
        'neg_label': 'negative',
    }

    # Format 8: Dash separator
    formats['dash_sep'] = {
        'template': 'Review - {text}\n---\nSentiment - ',
        'pos_label': 'positive',
        'neg_label': 'negative',
    }

    # Format 9: JSON-like
    formats['json_like'] = {
        'template': '{{"review": "{text}", "sentiment": "',
        'pos_label': 'positive',
        'neg_label': 'negative',
    }

    # Format 10: Emoji context
    formats['emoji'] = {
        'template': 'Movie review: {text}\nRating: 👍 or 👎?\nAnswer: ',
        'pos_label': '👍',
        'neg_label': '👎',
    }

    # Format 11: Academic
    formats['academic'] = {
        'template': 'In a sentiment analysis task, the following text should be labeled as positive or negative.\n\nText: {text}\n\nLabel: ',
        'pos_label': 'positive',
        'neg_label': 'negative',
    }

    # Format 12: Chat style
    formats['chat'] = {
        'template': 'User: What is the sentiment of this review? {text}\nAssistant: The sentiment is ',
        'pos_label': 'positive',
        'neg_label': 'negative',
    }

    # Format 13: Pipe separator
    formats['pipe_sep'] = {
        'template': '{text} | Sentiment: ',
        'pos_label': 'positive',
        'neg_label': 'negative',
    }

    # Format 14: Parenthetical
    formats['parenthetical'] = {
        'template': 'Sentiment (positive or negative) of "{text}": ',
        'pos_label': 'positive',
        'neg_label': 'negative',
    }

    # Format 15: Star rating framing
    formats['star_rating'] = {
        'template': 'Review: {text}\nWould you give this a high or low rating? ',
        'pos_label': 'high',
        'neg_label': 'low',
    }

    # Format 16: Bare with newline
    formats['bare_newline'] = {
        'template': '{text}\n',
        'pos_label': 'great',
        'neg_label': 'terrible',
    }

    # Format 17: XML-like tags
    formats['xml_tags'] = {
        'template': '<review>{text}</review>\n<sentiment>',
        'pos_label': 'positive',
        'neg_label': 'negative',
    }

    # Format 18: Colon no space
    formats['colon_nospace'] = {
        'template': 'Review:{text}\nSentiment:',
        'pos_label': 'positive',
        'neg_label': 'negative',
    }

    # Format 19: Tab separated
    formats['tab_sep'] = {
        'template': 'Review\t{text}\nSentiment\t',
        'pos_label': 'positive',
        'neg_label': 'negative',
    }

    # Format 20: Reverse order
    formats['reverse_order'] = {
        'template': 'The sentiment is: positive or negative?\n{text}\nAnswer: ',
        'pos_label': 'positive',
        'neg_label': 'negative',
    }

    print(f"Created {len(formats)} prompt format variants")
    return formats


# ═══════════════════════════════════════════════════════════════
# STEP 3: Load Dataset
# ═══════════════════════════════════════════════════════════════

def load_sst2_data(n_examples):
    """Load SST-2 validation data."""
    from datasets import load_from_disk
    ds = load_from_disk('datasets/sst2')

    # Sample balanced subset
    texts = ds['sentence']
    labels = ds['label']

    # Get indices for each class
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    neg_idx = [i for i, l in enumerate(labels) if l == 0]

    n_per_class = min(n_examples // 2, len(pos_idx), len(neg_idx))
    random.shuffle(pos_idx)
    random.shuffle(neg_idx)

    selected = pos_idx[:n_per_class] + neg_idx[:n_per_class]
    random.shuffle(selected)

    data = [(texts[i], labels[i]) for i in selected]
    print(f"Loaded {len(data)} examples ({n_per_class} per class)")
    return data


# ═══════════════════════════════════════════════════════════════
# STEP 4: Measure Performance per Format
# ═══════════════════════════════════════════════════════════════

def measure_format_accuracy(model, data, formats):
    """
    For each prompt format, measure accuracy on SST-2 by comparing
    log-probabilities of positive vs negative label tokens.
    """
    results = {}

    for fmt_name, fmt_info in formats.items():
        template = fmt_info['template']
        pos_tok = fmt_info['pos_label']
        neg_tok = fmt_info['neg_label']

        # Get token IDs for labels
        pos_ids = model.tokenizer.encode(' ' + pos_tok, add_special_tokens=False)
        neg_ids = model.tokenizer.encode(' ' + neg_tok, add_special_tokens=False)
        # Also try without space
        pos_ids_ns = model.tokenizer.encode(pos_tok, add_special_tokens=False)
        neg_ids_ns = model.tokenizer.encode(neg_tok, add_special_tokens=False)

        correct = 0
        total = 0
        logit_diffs = []

        for text, label in data:
            prompt = template.format(text=text[:200])  # Truncate long texts

            try:
                tokens = model.to_tokens(prompt, prepend_bos=True)
                if tokens.shape[1] > 512:  # Skip overly long prompts
                    continue

                with torch.no_grad():
                    logits = model(tokens)

                # Get logits at the last position
                last_logits = logits[0, -1, :]

                # Compare positive vs negative label probability
                pos_logit = max(
                    last_logits[pos_ids[0]].item() if pos_ids else -float('inf'),
                    last_logits[pos_ids_ns[0]].item() if pos_ids_ns else -float('inf'),
                )
                neg_logit = max(
                    last_logits[neg_ids[0]].item() if neg_ids else -float('inf'),
                    last_logits[neg_ids_ns[0]].item() if neg_ids_ns else -float('inf'),
                )

                predicted = 1 if pos_logit > neg_logit else 0
                correct += (predicted == label)
                total += 1
                logit_diffs.append(pos_logit - neg_logit)

            except Exception as e:
                continue

        acc = correct / total if total > 0 else 0
        results[fmt_name] = {
            'accuracy': acc,
            'n_examples': total,
            'mean_logit_diff': np.mean(logit_diffs) if logit_diffs else 0,
            'std_logit_diff': np.std(logit_diffs) if logit_diffs else 0,
        }
        print(f"  {fmt_name:20s}: acc={acc:.3f} (n={total})")

    return results


# ═══════════════════════════════════════════════════════════════
# STEP 5: Extract Geometric Features from Attention Patterns
# ═══════════════════════════════════════════════════════════════

def compute_geometric_features(attention_patterns):
    """
    Compute geometric features from attention patterns for a single input.

    attention_patterns: tensor of shape (n_layers, n_heads, seq_len, seq_len)

    Returns dict of feature values.
    """
    n_layers, n_heads, seq_len, _ = attention_patterns.shape
    features = {}

    # 1. Attention entropy per head (averaged across layers)
    # High entropy = diffuse attention = potentially unstable
    entropies = []
    for l in range(n_layers):
        for h in range(n_heads):
            attn = attention_patterns[l, h]  # (seq_len, seq_len)
            # Compute entropy for each query position, average
            eps = 1e-10
            ent = -(attn * torch.log(attn + eps)).sum(dim=-1).mean().item()
            entropies.append(ent)

    features['mean_entropy'] = np.mean(entropies)
    features['std_entropy'] = np.std(entropies)
    features['max_entropy'] = np.max(entropies)

    # Layer-wise entropy (early, mid, late)
    layer_entropies = []
    for l in range(n_layers):
        layer_ent = []
        for h in range(n_heads):
            attn = attention_patterns[l, h]
            ent = -(attn * torch.log(attn + eps)).sum(dim=-1).mean().item()
            layer_ent.append(ent)
        layer_entropies.append(np.mean(layer_ent))

    features['early_layer_entropy'] = np.mean(layer_entropies[:4])
    features['mid_layer_entropy'] = np.mean(layer_entropies[4:8])
    features['late_layer_entropy'] = np.mean(layer_entropies[8:])
    features['entropy_layer_std'] = np.std(layer_entropies)

    # 2. Singular value spectrum features
    # High condition number or rapid decay = low effective rank = potentially fragile
    sv_features_per_layer = []
    for l in range(n_layers):
        for h in range(n_heads):
            attn = attention_patterns[l, h].cpu().numpy()
            svs = svdvals(attn)
            svs = svs / (svs[0] + 1e-10)  # Normalize

            sv_features_per_layer.append({
                'effective_rank': np.exp(-np.sum(svs * np.log(svs + 1e-10))),
                'condition_number': svs[0] / (svs[-1] + 1e-10),
                'top3_ratio': np.sum(svs[:3]) / (np.sum(svs) + 1e-10),
                'sv_entropy': -np.sum(svs * np.log(svs + 1e-10)),
            })

    features['mean_effective_rank'] = np.mean([s['effective_rank'] for s in sv_features_per_layer])
    features['std_effective_rank'] = np.std([s['effective_rank'] for s in sv_features_per_layer])
    features['mean_condition_number'] = np.mean([s['condition_number'] for s in sv_features_per_layer])
    features['mean_top3_sv_ratio'] = np.mean([s['top3_ratio'] for s in sv_features_per_layer])
    features['mean_sv_entropy'] = np.mean([s['sv_entropy'] for s in sv_features_per_layer])

    # 3. Attention variance features
    # High variance across positions = complex attention landscape = potential instability
    variances = []
    for l in range(n_layers):
        for h in range(n_heads):
            attn = attention_patterns[l, h]
            var = attn.var().item()
            variances.append(var)

    features['mean_attn_variance'] = np.mean(variances)
    features['std_attn_variance'] = np.std(variances)
    features['max_attn_variance'] = np.max(variances)

    # Layer-wise variance
    layer_vars = []
    for l in range(n_layers):
        layer_var = []
        for h in range(n_heads):
            layer_var.append(attention_patterns[l, h].var().item())
        layer_vars.append(np.mean(layer_var))

    features['mid_layer_variance'] = np.mean(layer_vars[4:8])
    features['variance_layer_std'] = np.std(layer_vars)

    # 4. Attention concentration (max attention weight per position)
    concentrations = []
    for l in range(n_layers):
        for h in range(n_heads):
            attn = attention_patterns[l, h]
            max_attn = attn.max(dim=-1)[0].mean().item()
            concentrations.append(max_attn)

    features['mean_concentration'] = np.mean(concentrations)
    features['std_concentration'] = np.std(concentrations)

    # 5. Inter-head similarity within layers (measures redundancy)
    inter_head_sims = []
    for l in range(n_layers):
        for h1 in range(n_heads):
            for h2 in range(h1 + 1, n_heads):
                a1 = attention_patterns[l, h1].flatten()
                a2 = attention_patterns[l, h2].flatten()
                sim = torch.nn.functional.cosine_similarity(a1.unsqueeze(0), a2.unsqueeze(0)).item()
                inter_head_sims.append(sim)

    features['mean_inter_head_sim'] = np.mean(inter_head_sims)
    features['std_inter_head_sim'] = np.std(inter_head_sims)

    # 6. Frobenius norm of attention matrices (overall magnitude)
    frob_norms = []
    for l in range(n_layers):
        for h in range(n_heads):
            fn = torch.norm(attention_patterns[l, h], p='fro').item()
            frob_norms.append(fn)

    features['mean_frob_norm'] = np.mean(frob_norms)
    features['std_frob_norm'] = np.std(frob_norms)

    return features


def extract_features_for_format(model, data, fmt_info, n_samples=50):
    """
    Extract geometric features for a prompt format, averaged over n_samples examples.
    Also returns per-example features for fine-grained analysis.
    """
    template = fmt_info['template']
    all_features = []

    sample_data = data[:n_samples]

    for text, label in sample_data:
        prompt = template.format(text=text[:200])

        try:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            if tokens.shape[1] > 512:
                continue

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)

            # Extract attention patterns: shape (n_layers, n_heads, seq_len, seq_len)
            attn_patterns = torch.stack([
                cache['pattern', l][0]  # Remove batch dim
                for l in range(model.cfg.n_layers)
            ])

            feats = compute_geometric_features(attn_patterns)
            all_features.append(feats)

        except Exception as e:
            continue

    if not all_features:
        return None

    # Average features across examples
    avg_features = {}
    for key in all_features[0].keys():
        vals = [f[key] for f in all_features]
        avg_features[key] = np.mean(vals)
        avg_features[f'{key}_std_across_examples'] = np.std(vals)

    return avg_features


# ═══════════════════════════════════════════════════════════════
# STEP 6: Baseline Predictors
# ═══════════════════════════════════════════════════════════════

def compute_baseline_features(model, data, fmt_info, n_samples=50):
    """Compute baseline features: logit confidence and embedding similarity."""
    template = fmt_info['template']
    confidences = []

    sample_data = data[:n_samples]

    for text, label in sample_data:
        prompt = template.format(text=text[:200])

        try:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            if tokens.shape[1] > 512:
                continue

            with torch.no_grad():
                logits = model(tokens)

            last_logits = logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=0)
            max_prob = probs.max().item()
            confidences.append(max_prob)

        except Exception:
            continue

    return {
        'mean_confidence': np.mean(confidences) if confidences else 0,
        'std_confidence': np.std(confidences) if confidences else 0,
    }


# ═══════════════════════════════════════════════════════════════
# STEP 7: Analysis and Visualization
# ═══════════════════════════════════════════════════════════════

def correlation_analysis(feature_df, accuracy_col='accuracy'):
    """Compute Spearman correlations between features and accuracy."""
    feature_cols = [c for c in feature_df.columns if c not in [
        'format', accuracy_col, 'n_examples', 'mean_logit_diff', 'std_logit_diff',
        'mean_confidence', 'std_confidence'
    ]]

    correlations = {}
    for col in feature_cols:
        if feature_df[col].std() < 1e-10:
            continue
        rho, p = stats.spearmanr(feature_df[col], feature_df[accuracy_col])
        correlations[col] = {'rho': rho, 'p_value': p}

    return correlations


def plot_top_correlations(correlations, feature_df, accuracy_col='accuracy'):
    """Plot top correlating features vs accuracy."""
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]['rho']), reverse=True)
    top_n = min(8, len(sorted_corr))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, (feat, info) in enumerate(sorted_corr[:top_n]):
        ax = axes[i]
        ax.scatter(feature_df[feat], feature_df[accuracy_col], alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'ρ={info["rho"]:.3f}, p={info["p_value"]:.3f}', fontsize=9)

        # Add trend line
        z = np.polyfit(feature_df[feat], feature_df[accuracy_col], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(feature_df[feat].min(), feature_df[feat].max(), 100)
        ax.plot(x_line, p_line(x_line), 'r--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'top_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/top_correlations.png")


def plot_accuracy_distribution(results):
    """Plot accuracy distribution across prompt formats."""
    accs = [r['accuracy'] for r in results.values()]
    names = list(results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart
    sorted_idx = np.argsort(accs)
    ax1.barh([names[i] for i in sorted_idx], [accs[i] for i in sorted_idx], color='steelblue')
    ax1.set_xlabel('Accuracy')
    ax1.set_title('SST-2 Accuracy by Prompt Format (GPT-2 Small)')
    ax1.axvline(np.mean(accs), color='red', linestyle='--', label=f'Mean={np.mean(accs):.3f}')
    ax1.legend()

    # Histogram
    ax2.hist(accs, bins=10, edgecolor='black', color='steelblue', alpha=0.7)
    ax2.set_xlabel('Accuracy')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Accuracy Distribution (std={np.std(accs):.3f})')
    ax2.axvline(np.mean(accs), color='red', linestyle='--')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'accuracy_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/accuracy_distribution.png")


def plot_classifier_results(y_true, y_pred_proba, classifier_name):
    """Plot ROC curve and confusion matrix."""
    from sklearn.metrics import roc_curve, auc

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, 'b-', lw=2, label=f'{classifier_name} (AUC={roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.500)')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve for Prompt Failure Prediction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Confusion matrix
    y_pred = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Stable', 'Failure'], yticklabels=['Stable', 'Failure'])
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title(f'Confusion Matrix ({classifier_name})')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'classifier_{classifier_name.lower().replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(feature_names, importances, title='Feature Importance'):
    """Plot feature importance from classifier."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_idx = np.argsort(importances)
    ax.barh([feature_names[i] for i in sorted_idx[-15:]],
            [importances[i] for i in sorted_idx[-15:]], color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/feature_importance.png")


def plot_layer_geometry(feature_df):
    """Plot layer-wise geometric properties."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Entropy by layer region
    for i, (col, label) in enumerate([
        ('early_layer_entropy', 'Early (L0-3)'),
        ('mid_layer_entropy', 'Mid (L4-7)'),
        ('late_layer_entropy', 'Late (L8-11)')
    ]):
        if col in feature_df.columns:
            axes[0].scatter(feature_df[col], feature_df['accuracy'],
                          label=label, alpha=0.7, s=50)
    axes[0].set_xlabel('Layer Entropy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Layer-wise Entropy vs Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Effective rank vs accuracy
    if 'mean_effective_rank' in feature_df.columns:
        axes[1].scatter(feature_df['mean_effective_rank'], feature_df['accuracy'],
                       color='green', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        axes[1].set_xlabel('Mean Effective Rank')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Effective Rank vs Accuracy')
        axes[1].grid(True, alpha=0.3)

    # Inter-head similarity vs accuracy
    if 'mean_inter_head_sim' in feature_df.columns:
        axes[2].scatter(feature_df['mean_inter_head_sim'], feature_df['accuracy'],
                       color='orange', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        axes[2].set_xlabel('Mean Inter-Head Similarity')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Inter-Head Similarity vs Accuracy')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'layer_geometry.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/layer_geometry.png")


# ═══════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════

def main():
    start_time = time.time()

    # --- Load model ---
    print("=" * 60)
    print("STEP 1: Loading model")
    print("=" * 60)
    model = load_model()

    # --- Create prompt formats ---
    print("\n" + "=" * 60)
    print("STEP 2: Creating prompt format variants")
    print("=" * 60)
    formats = create_prompt_formats()

    # --- Load data ---
    print("\n" + "=" * 60)
    print("STEP 3: Loading SST-2 data")
    print("=" * 60)
    data = load_sst2_data(CONFIG['n_examples'])

    # --- Measure accuracy per format ---
    print("\n" + "=" * 60)
    print("STEP 4: Measuring accuracy per prompt format")
    print("=" * 60)
    perf_results = measure_format_accuracy(model, data, formats)

    # Save intermediate results
    with open(RESULTS_DIR / 'performance_results.json', 'w') as f:
        json.dump(perf_results, f, indent=2)

    # Plot accuracy distribution
    plot_accuracy_distribution(perf_results)

    accs = [r['accuracy'] for r in perf_results.values()]
    print(f"\nAccuracy range: {min(accs):.3f} - {max(accs):.3f}")
    print(f"Accuracy spread: {max(accs) - min(accs):.3f}")
    print(f"Mean accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    # --- Extract geometric features ---
    print("\n" + "=" * 60)
    print("STEP 5: Extracting geometric features from attention patterns")
    print("=" * 60)

    n_feature_samples = min(50, len(data))
    all_geo_features = {}
    all_baseline_features = {}

    for fmt_name, fmt_info in formats.items():
        print(f"  Extracting features for: {fmt_name}")
        geo_feats = extract_features_for_format(model, data, fmt_info, n_samples=n_feature_samples)
        base_feats = compute_baseline_features(model, data, fmt_info, n_samples=n_feature_samples)

        if geo_feats is not None:
            all_geo_features[fmt_name] = geo_feats
            all_baseline_features[fmt_name] = base_feats

    # --- Build feature DataFrame ---
    print("\n" + "=" * 60)
    print("STEP 6: Correlation analysis")
    print("=" * 60)

    rows = []
    for fmt_name in all_geo_features:
        row = {'format': fmt_name}
        row.update(perf_results[fmt_name])
        row.update(all_geo_features[fmt_name])
        row.update(all_baseline_features[fmt_name])
        rows.append(row)

    feature_df = pd.DataFrame(rows)

    # Filter to only use mean features (not per-example std) for primary analysis
    primary_features = [c for c in feature_df.columns if not c.endswith('_std_across_examples')
                       and c not in ['format', 'accuracy', 'n_examples', 'mean_logit_diff',
                                    'std_logit_diff', 'mean_confidence', 'std_confidence']]

    # Correlation analysis
    correlations = correlation_analysis(feature_df)

    print("\nTop correlations with accuracy (by |ρ|):")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]['rho']), reverse=True)
    for feat, info in sorted_corr[:15]:
        sig = "***" if info['p_value'] < 0.001 else "**" if info['p_value'] < 0.01 else "*" if info['p_value'] < 0.05 else ""
        print(f"  {feat:40s}: ρ={info['rho']:+.3f}  p={info['p_value']:.4f} {sig}")

    # Plot correlations
    plot_top_correlations(correlations, feature_df)
    plot_layer_geometry(feature_df)

    # --- Failure prediction classifier ---
    print("\n" + "=" * 60)
    print("STEP 7: Training failure prediction classifier")
    print("=" * 60)

    # Define failure as below-median accuracy
    median_acc = feature_df['accuracy'].median()
    feature_df['is_failure'] = (feature_df['accuracy'] < median_acc).astype(int)

    print(f"Failure threshold (median accuracy): {median_acc:.3f}")
    print(f"Failures: {feature_df['is_failure'].sum()}, Stable: {(1 - feature_df['is_failure']).sum()}")

    # Prepare features
    geo_feature_cols = [c for c in feature_df.columns
                        if c not in ['format', 'accuracy', 'n_examples', 'mean_logit_diff',
                                    'std_logit_diff', 'mean_confidence', 'std_confidence',
                                    'is_failure']
                        and not c.endswith('_std_across_examples')]

    X_geo = feature_df[geo_feature_cols].values
    X_baseline_conf = feature_df[['mean_confidence']].values
    y = feature_df['is_failure'].values

    scaler_geo = StandardScaler()
    X_geo_scaled = scaler_geo.fit_transform(X_geo)

    scaler_base = StandardScaler()
    X_base_scaled = scaler_base.fit_transform(X_baseline_conf)

    # With only 20 formats, use leave-one-out cross-validation
    from sklearn.model_selection import LeaveOneOut

    loo = LeaveOneOut()

    classifiers = {
        'Geometric (Logistic Regression)': LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
        'Geometric (Random Forest)': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=SEED),
        'Baseline (Confidence)': LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
    }

    classifier_results = {}

    for clf_name, clf in classifiers.items():
        X = X_geo_scaled if 'Geometric' in clf_name else X_base_scaled

        y_pred_all = np.zeros(len(y))
        y_proba_all = np.zeros(len(y))

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf_copy = type(clf)(**clf.get_params())
            clf_copy.fit(X_train, y_train)

            y_pred_all[test_idx] = clf_copy.predict(X_test)
            if hasattr(clf_copy, 'predict_proba'):
                y_proba_all[test_idx] = clf_copy.predict_proba(X_test)[:, 1]
            else:
                y_proba_all[test_idx] = clf_copy.decision_function(X_test)

        acc = np.mean(y_pred_all == y)
        try:
            auc = roc_auc_score(y, y_proba_all)
        except:
            auc = 0.5

        prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred_all, average='binary', zero_division=0)

        classifier_results[clf_name] = {
            'accuracy': acc,
            'auc': auc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
        }

        print(f"\n{clf_name}:")
        print(f"  LOO Accuracy: {acc:.3f}")
        print(f"  AUC: {auc:.3f}")
        print(f"  Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

        # Plot ROC for geometric classifiers
        if 'Geometric' in clf_name:
            plot_classifier_results(y, y_proba_all, clf_name)

    # Random baseline
    random_auc = 0.5
    classifier_results['Random Baseline'] = {
        'accuracy': 0.5, 'auc': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5
    }

    # --- Feature importance from Random Forest ---
    print("\n" + "=" * 60)
    print("STEP 8: Feature importance analysis")
    print("=" * 60)

    rf_full = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=SEED)
    rf_full.fit(X_geo_scaled, y)
    importances = rf_full.feature_importances_

    print("\nTop 10 most important geometric features:")
    sorted_imp_idx = np.argsort(importances)[::-1]
    for i in sorted_imp_idx[:10]:
        print(f"  {geo_feature_cols[i]:40s}: importance={importances[i]:.4f}")

    plot_feature_importance(geo_feature_cols, importances)

    # --- Comparison plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    clf_names = list(classifier_results.keys())
    aucs = [classifier_results[n]['auc'] for n in clf_names]
    colors = ['steelblue', 'darkblue', 'orange', 'gray']
    ax.barh(clf_names, aucs, color=colors[:len(clf_names)])
    ax.set_xlabel('AUC-ROC')
    ax.set_title('Prompt Failure Prediction: Classifier Comparison')
    ax.axvline(0.5, color='red', linestyle='--', label='Random')
    ax.set_xlim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'classifier_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/classifier_comparison.png")

    # --- Save all results ---
    elapsed = time.time() - start_time

    final_results = {
        'config': CONFIG,
        'performance': perf_results,
        'correlations': {k: v for k, v in sorted_corr},
        'classifier_results': classifier_results,
        'feature_importance': {geo_feature_cols[i]: float(importances[i])
                              for i in sorted_imp_idx},
        'elapsed_seconds': elapsed,
        'accuracy_stats': {
            'mean': float(np.mean(accs)),
            'std': float(np.std(accs)),
            'min': float(min(accs)),
            'max': float(max(accs)),
            'spread': float(max(accs) - min(accs)),
            'median': float(np.median(accs)),
        },
    }

    with open(RESULTS_DIR / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    feature_df.to_csv(RESULTS_DIR / 'feature_data.csv', index=False)

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT COMPLETE in {elapsed:.1f} seconds")
    print(f"{'=' * 60}")
    print(f"Results saved to: results/")
    print(f"Figures saved to: results/figures/")

    return final_results


if __name__ == '__main__':
    results = main()
