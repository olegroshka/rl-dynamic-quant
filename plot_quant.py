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
  4) "Reward Components in 2x2 subplots" (perf, kl, memory, total)
  5) "Layer Stats Distributions" (e.g., attention_entropy, weight_std, etc.)

All plots are saved in results/<name>/ with filenames including <name>.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

PALETTE = "colorblind"


def load_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def extract_quant_matrix(data):
    num_episodes = len(data)
    first_bits = data[0]["episode_summary"]["layer_bits"]
    num_layers = len(first_bits)

    quant_matrix = np.empty((num_episodes, num_layers), dtype=object)
    for i, episode in enumerate(data):
        layer_bits = episode["episode_summary"]["layer_bits"]
        for j, qtype in enumerate(layer_bits):
            quant_matrix[i, j] = qtype
    return quant_matrix


def bin_episodes(quant_matrix, bin_size=20):
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
    sns.set(style="whitegrid")
    num_episodes, num_layers = quant_matrix.shape

    unique_types = sorted(set(quant_matrix.flatten()))
    quant_to_int = {qt: i for i, qt in enumerate(unique_types)}

    int_mat = np.array([[quant_to_int[q] for q in row] for row in quant_matrix], dtype=int)
    cmap = sns.color_palette(palette, n_colors=len(unique_types))

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        int_mat,
        cmap=cmap,
        vmin=0,
        vmax=len(unique_types) - 1,
        cbar=True,
        xticklabels=[f"Layer {i}" for i in range(num_layers)],
        yticklabels=[f"Ep {i}" for i in range(num_episodes)]
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
    plt.title(f"Raw Quantization Heatmap - {experiment_name}\n(Episodes x Layers)")
    plt.xlabel("Layer Index")
    plt.ylabel("Episode Index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved raw heatmap to: {out_path}")


def plot_binned_heatmap(binned_matrix, out_path, experiment_name, palette="pastel"):
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
        vmax=len(unique_types) - 1,
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
    sns.set(style="whitegrid")
    num_episodes, num_layers = quant_matrix.shape

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
    Plot 5 main reward components (perf, kl, memory, entropy, total) + their EMA lines
    in a 3x2 grid of subplots (5 used, 1 left empty).
    Each subplot uses its own y-scale.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")

    # 1) Collect all possible reward keys
    all_keys = set()
    for ep in data:
        if "reward_components" in ep:
            all_keys.update(ep["reward_components"].keys())
    rc_values = {k: [] for k in all_keys}

    # 2) Populate arrays from data
    for ep in data:
        rwd = ep.get("reward_components", {})
        for k in all_keys:
            rc_values[k].append(rwd.get(k, None))

    # 3) Define our 5 main pairs: (raw_key, ema_key, descriptive_title)
    component_pairs = [
        ("perf_reward",    "perf_reward_ema",    "Performance Reward"),
        ("kl_reward",      "kl_reward_ema",      "KL Reward"),
        ("memory_reward",  "memory_reward_ema",  "Memory Reward"),
        ("entropy_reward", "entropy_reward_ema", "Entropy Reward"),
        ("total_reward",   "total_reward_ema",   "Total Reward")
    ]

    # 4) Create a 3x2 grid => 6 subplots, but we only need 5. We'll leave the 6th blank.
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 10), sharex=True)
    axs = axs.flatten()  # So we can index them easily

    for i, (main_k, ema_k, title_str) in enumerate(component_pairs):
        ax = axs[i]

        main_vals = rc_values.get(main_k, [])
        ema_vals  = rc_values.get(ema_k,  [])

        # Plot if we have data
        if main_vals:
            ax.plot(main_vals, label=main_k)
        if ema_vals:
            ax.plot(ema_vals, label=ema_k)

        ax.set_title(f"{title_str} — {experiment_name}")
        ax.set_ylabel("Reward Value")
        ax.legend(loc="best")

    # 5) If there's a 6th subplot we don't use, optionally label it "Unused" or hide it
    if len(axs) > 5:
        axs[5].axis("off")
        axs[5].set_title("")

    # 6) Common X label for bottom row subplots
    for ax in axs[4:]:
        ax.set_xlabel("Episode Index")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved 5-subplot reward plot to: {out_path}")


def plot_layer_stats_distributions(data, out_dir, experiment_name):
    metrics_map = defaultdict(lambda: defaultdict(list))

    for ep in data:
        for key in ep.keys():
            if key.startswith("layer_stats"):
                ls = ep[key]
                if not isinstance(ls, dict):
                    continue
                layer_idx = ls.get("layer_idx", None)
                if layer_idx is None:
                    continue

                for field, val in ls.items():
                    if field in ["layer_idx", "current_quant_type"]:
                        continue
                    metrics_map[field][layer_idx].append(val)

    for metric_name, layer_dict in metrics_map.items():
        import pandas as pd
        rows = []
        for layer_idx, vals in layer_dict.items():
            for v in vals:
                rows.append({"layer": layer_idx, "value": v})
        df = pd.DataFrame(rows)
        df = df.sort_values("layer")

        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        # Single color to avoid future palette/hue warnings
        sns.boxplot(
            x="layer",
            y="value",
            data=df,
            color="lightblue"
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

    experiment_name = args.name
    base_dir = f"results/{experiment_name}"
    json_path = os.path.join(base_dir, "info_stats.json")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist.")
        return

    data = load_data(json_path)
    print(f"Loaded {len(data)} episodes from {json_path}")

    out_dir = base_dir

    # 1) Extract layer quant matrix
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

    # 5) New 2x2 subplots for the four main reward components
    out_path_4sub = os.path.join(out_dir, f"{experiment_name}_reward_subplots.png")
    plot_reward_components(data, out_path_4sub, experiment_name)

    # 6) Layer stats distributions
    plot_layer_stats_distributions(data, out_dir, experiment_name)

    print("All plots created successfully.")


if __name__ == "__main__":
    main()
