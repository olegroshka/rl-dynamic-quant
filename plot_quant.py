#!/usr/bin/env python
"""
plot_quant.py

Usage:
    python plot_quant.py --name gpt2-250-gae-sig-ewa-rwd-v6

This script expects to find JSON data at: results/<name>/info_stats.json
It then creates multiple plots:
  1) "Raw Heatmap" of layer quant types by episode
  2) "Binned Heatmap" (majority vote per bin)
  3) "Distribution of Quant Types per Layer" (stacked bar)
  4) "Reward Components over Episodes" (line plots)
  5) "Layer Stats Distributions" (e.g., attention_entropy, weight_std, etc.)

All plots are saved in results/<name>/ with filenames including <name>.

Requirements:
  pip install matplotlib seaborn
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

PALETTE = "colorblind"#"pastel"


def load_data(json_path):
    """
    Load the top-level list from info_stats.json.
    Each element is typically one "episode" dict with keys:
      - "reward_components"
      - "layer_stats"/"layer_stats_0"/"layer_stats_1"/... or an "episode_summary"
    Returns the raw list of episodes as 'data'.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def extract_quant_matrix(data):
    """
    From each episode, read the 'episode_summary.layer_bits',
    which is typically a list of length == #layers.

    Returns:
       quant_matrix: shape (num_episodes, num_layers), each cell is a string, e.g. "fp16"/"int8"/"nf4"/"fp4"/"fp32"
    """
    num_episodes = len(data)
    # Suppose the first episode indicates the # of layers
    first_bits = data[0]["episode_summary"]["layer_bits"]
    num_layers = len(first_bits)

    quant_matrix = np.empty((num_episodes, num_layers), dtype=object)
    for i, episode in enumerate(data):
        layer_bits = episode["episode_summary"]["layer_bits"]
        for j, qtype in enumerate(layer_bits):
            quant_matrix[i, j] = qtype
    return quant_matrix

def bin_episodes(quant_matrix, bin_size=20):
    """
    Group consecutive episodes into bins. For each bin & layer,
    pick the most frequent quant type (majority).
    Returns a matrix of shape (num_bins, num_layers).
    """
    num_episodes, num_layers = quant_matrix.shape
    num_bins = (num_episodes + bin_size - 1) // bin_size

    binned = np.empty((num_bins, num_layers), dtype=object)
    for b in range(num_bins):
        start = b * bin_size
        end = min((b + 1) * bin_size, num_episodes)
        chunk = quant_matrix[start:end]
        for layer in range(num_layers):
            counts = Counter(chunk[:, layer])
            majority_type = max(counts, key=counts.get)
            binned[b, layer] = majority_type
    return binned

def plot_raw_heatmap(quant_matrix, out_path, experiment_name, palette="pastel"):
    """
    Raw heatmap (episodes x layers). Each cell is mapped to an integer for color.
    """
    sns.set(style="whitegrid")
    num_episodes, num_layers = quant_matrix.shape

    unique_types = sorted(set(quant_matrix.flatten()))
    quant_to_int = {qt: i for i, qt in enumerate(unique_types)}

    # Convert to integer codes
    int_mat = np.array([[quant_to_int[q] for q in row] for row in quant_matrix], dtype=int)
    # Pastel color scheme
    cmap = sns.color_palette(palette, n_colors=len(unique_types))

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        int_mat,
        cmap=cmap,
        vmin=0,
        vmax=len(unique_types)-1,
        cbar=True,
        xticklabels=[f"Layer {i}" for i in range(num_layers)],
        yticklabels=[f"Ep {i}" for i in range(num_episodes)]
    )
    # Build legend
    from matplotlib.patches import Patch
    patches = []
    for qt, idx in quant_to_int.items():
        patches.append(Patch(color=cmap[idx], label=qt))
    ax.legend(
        handles=patches,
        title="Quant Type",
        bbox_to_anchor=(1.25, 1),
        loc="upper left"
    )
    plt.title(f"Raw Quantization Heatmap - {experiment_name}\n(Episodes x Layers)")
    plt.xlabel("Layer Index")
    plt.ylabel("Episode Index")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved raw heatmap to: {out_path}")

def plot_binned_heatmap(binned_matrix, out_path, experiment_name, palette="pastel"):
    """
    Binned heatmap. shape (num_bins x num_layers).
    """
    sns.set(style="whitegrid")
    num_bins, num_layers = binned_matrix.shape

    unique_types = sorted(set(binned_matrix.flatten()))
    quant_to_int = {qt: i for i, qt in enumerate(unique_types)}

    int_mat = np.array([[quant_to_int[q] for q in row] for row in binned_matrix], dtype=int)
    cmap = sns.color_palette(palette, n_colors=len(unique_types))

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        int_mat,
        cmap=cmap,
        vmin=0,
        vmax=len(unique_types)-1,
        cbar=True,
        xticklabels=[f"Layer {i}" for i in range(num_layers)],
        yticklabels=[f"Bin {i}" for i in range(num_bins)]
    )
    from matplotlib.patches import Patch
    patches = []
    for qt, idx in quant_to_int.items():
        patches.append(Patch(color=cmap[idx], label=qt))
    ax.legend(
        handles=patches,
        title="Quant Type",
        bbox_to_anchor=(1.25, 1),
        loc="upper left"
    )
    plt.title(f"Binned Quantization Heatmap - {experiment_name}")
    plt.xlabel("Layer Index")
    plt.ylabel("Episode Bin")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved binned heatmap to: {out_path}")

def plot_layer_distribution_bar(quant_matrix, out_path, experiment_name, palette="pastel"):
    """
    Stacked bar chart of quant type frequencies per layer.
    """
    sns.set(style="whitegrid")
    num_episodes, num_layers = quant_matrix.shape

    # Count frequencies
    layer_type_counts = []
    all_qtypes = set()
    for layer in range(num_layers):
        counts = Counter(quant_matrix[:, layer])
        layer_type_counts.append(counts)
        all_qtypes.update(counts.keys())
    all_qtypes = sorted(list(all_qtypes))

    freq_mat = np.zeros((num_layers, len(all_qtypes)), dtype=float)
    for layer in range(num_layers):
        total = sum(layer_type_counts[layer].values())
        for i, qt in enumerate(all_qtypes):
            freq_mat[layer, i] = layer_type_counts[layer][qt] / total

    cmap = sns.color_palette(palette, n_colors=len(all_qtypes))
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(num_layers)

    for i, qt in enumerate(all_qtypes):
        plt.bar(
            range(num_layers),
            freq_mat[:, i],
            bottom=bottom,
            color=cmap[i],
            edgecolor='black',
            label=qt
        )
        bottom += freq_mat[:, i]

    plt.xticks(range(num_layers), [f"Layer {i}" for i in range(num_layers)], rotation=45)
    plt.ylim([0, 1])
    plt.ylabel("Fraction of episodes")
    plt.title(f"Distribution of Quant Types per Layer - {experiment_name}")
    plt.legend(title="Quant Type", bbox_to_anchor=(1.25, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved distribution bar chart to: {out_path}")


def plot_reward_components(data, out_path, experiment_name):
    """
    Plot line charts for each reward component (perf_reward, kl_reward, etc.) across episodes.
    Also handles the _ema versions if present.
    """
    sns.set(style="whitegrid")
    # We'll gather lists of each reward component by episode index
    # e.g. reward_components = data[i]["reward_components"]
    # Keys might include "perf_reward", "kl_reward", "entropy_reward", "memory_reward", "total_reward"
    # and "perf_reward_ema", ...
    all_keys = set()
    for ep in data:
        if "reward_components" in ep:
            all_keys.update(ep["reward_components"].keys())

    # We'll filter out keys that might be irrelevant if they appear rarely
    # but let's assume they're consistent. We'll just plot all keys we see.
    sorted_keys = sorted(list(all_keys))

    # Build dictionary: key -> list of values (one per episode)
    rc_values = {k: [] for k in sorted_keys}

    for ep in data:
        rwd = ep.get("reward_components", {})
        for k in sorted_keys:
            val = rwd.get(k, None)
            rc_values[k].append(val)

    # Plot each key on a line chart
    plt.figure(figsize=(10, 6))
    for k in sorted_keys:
        plt.plot(rc_values[k], label=k)
    plt.title(f"Reward Components - {experiment_name}")
    plt.xlabel("Episode Index")
    plt.ylabel("Reward Value")
    plt.legend(bbox_to_anchor=(1.25, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved reward components plot to: {out_path}")


def plot_layer_stats_distributions(data, out_dir, experiment_name):
    """
    Look at fields in the 'layer_stats_*' or 'layer_stats' dictionary for each layer.
    For each such field (e.g. "attention_entropy", "weight_mean", "weight_std", "gradient_norm"),
    we can produce a boxplot or distribution across episodes, grouped by layer.

    We'll produce one figure per metric.

    Example:
      "layer_stats_0": {
        "weight_mean": ...,
        "weight_std": ...,
        "gradient_norm": ...,
        "attention_entropy": ...,
        "layer_idx": 0,
        "current_quant_type": "nf4"
      }
    """
    # We'll accumulate metrics in a dict: metric_name -> {layer_idx: [values over episodes]}
    metrics_map = defaultdict(lambda: defaultdict(list))

    num_episodes = len(data)

    for ep_idx, ep in enumerate(data):
        # We might have "layer_stats" for one layer, or "layer_stats_0", "layer_stats_1", ...
        # or we might have multiple. We'll parse systematically.
        for key in ep.keys():
            if key.startswith("layer_stats"):
                # could be "layer_stats" or "layer_stats_0" etc.
                ls = ep[key]
                if not isinstance(ls, dict):
                    continue
                layer_idx = ls.get("layer_idx", None)
                if layer_idx is None:
                    # skip if no layer_idx
                    continue

                # Now gather each field
                for field, val in ls.items():
                    if field in ["layer_idx", "current_quant_type"]:
                        continue
                    # e.g. "weight_mean", "weight_std", "gradient_norm", "attention_entropy"
                    metrics_map[field][layer_idx].append(val)

    # Now for each metric in metrics_map, we have a dict of layer_idx -> list of values
    for metric_name, layer_dict in metrics_map.items():
        # We want to create a boxplot or stripplot across layers
        # We'll gather data in a form suitable for seaborn. Something like:
        #  data frame with columns: [ "layer", "value" ], then we do sns.boxplot(x="layer", y="value", data=...)
        import pandas as pd
        rows = []
        for layer_idx, vals in layer_dict.items():
            for v in vals:
                rows.append({"layer": layer_idx, "value": v})
        df = pd.DataFrame(rows)

        # Sort by layer
        df = df.sort_values("layer")

        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        sns.boxplot(
            x="layer",
            y="value",
            data=df,
            palette=PALETTE #"pastel"
        )
        plt.title(f"{metric_name} by Layer - {experiment_name}")
        plt.xlabel("Layer Index")
        plt.ylabel(metric_name)
        plt.tight_layout()
        out_path_metric = os.path.join(out_dir, f"{experiment_name}_layerstats_{metric_name}.png")
        plt.savefig(out_path_metric, dpi=200)
        plt.close()
        print(f"Saved layer stats boxplot for {metric_name} -> {out_path_metric}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True,
                        help="Experiment name, e.g. gpt2-250-gae-sig-ewa-rwd-v6. JSON is at results/<name>/info_stats.json")
    parser.add_argument("--bin_size", type=int, default=20,
                        help="Number of episodes per bin for the binned heatmap.")
    args = parser.parse_args()

    # Build path to JSON
    experiment_name = args.name
    base_dir = f"results/{experiment_name}"
    json_path = os.path.join(base_dir, "info_stats.json")

    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist.")
        return

    # Load data
    data = load_data(json_path)
    print(f"Loaded {len(data)} episodes from {json_path}")

    # Create output dir (same as base_dir)
    out_dir = base_dir

    # 1) Extract quantization matrix
    quant_matrix = extract_quant_matrix(data)

    # 2) Plot raw heatmap
    out_path_raw = os.path.join(out_dir, f"{experiment_name}_quant_heatmap_raw.png")
    plot_raw_heatmap(quant_matrix, out_path_raw, experiment_name, palette=PALETTE)

    # 3) Binned heatmap
    binned = bin_episodes(quant_matrix, args.bin_size)
    out_path_binned = os.path.join(out_dir, f"{experiment_name}_quant_heatmap_binned.png")
    plot_binned_heatmap(binned, out_path_binned, experiment_name, palette=PALETTE)

    # 4) Distribution bar chart
    out_path_dist = os.path.join(out_dir, f"{experiment_name}_quant_dist_bar.png")
    plot_layer_distribution_bar(quant_matrix, out_path_dist, experiment_name, palette=PALETTE)

    # 5) Reward components line plot
    out_path_rewards = os.path.join(out_dir, f"{experiment_name}_reward_components.png")
    plot_reward_components(data, out_path_rewards, experiment_name)

    # 6) Layer stats distributions
    #    e.g. attention_entropy, weight_mean, weight_std, gradient_norm
    plot_layer_stats_distributions(data, out_dir, experiment_name)

    print("All plots created successfully.")

if __name__ == "__main__":
    main()
