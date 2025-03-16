import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
from collections import Counter
import pandas as pd


def load_data(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


# a replacement for plot_quantization_and_memory
def plot_layer_quantization(layer_bits, compression, ax1_label, ax2_label, title):
    """
    Plots the distribution of quantization types per layer over episodes
    and overlays compression values.

    Parameters:
    - layer_bits: List of lists (size: 250x12), each sublist contains quantization types per layer.
    - compression: List of 250 values representing compression per episode.
    - ax1_label: Label for the primary y-axis (quantization distribution).
    - ax2_label: Label for the secondary y-axis (compression).
    - title: Title of the plot.
    """
    episodes = list(range(len(layer_bits)))

    # Initialize dictionary to hold quantization counts per episode
    quant_types = ["fp32", "fp16", "int8", "fp4", "nf4"]
    quant_counts = {qt: [] for qt in quant_types}

    # Count quantization types for each episode
    for episode_layers in layer_bits:
        counts = Counter(episode_layers)
        total = len(episode_layers)

        # Normalize counts to percentage
        for qt in quant_types:
            quant_counts[qt].append(counts.get(qt, 0) / total * 100)

    # Convert to DataFrame for plotting
    df = pd.DataFrame({"episode": episodes, **quant_counts})

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Stack plot for quantization distributions
    ax1.stackplot(df["episode"], *[df[qt] for qt in quant_types],
                  labels=quant_types,
                  colors=["#ff9999", "#ffcc99", "#99ff99", "#66b3ff", "#c2c2f0"],
                  alpha=0.7)
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(ax1_label)
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0, len(episodes) - 1)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Create secondary Y-axis for compression
    ax2 = ax1.twinx()
    ax2.plot(episodes, [c * 100 for c in compression], 'r-', linewidth=2, label="Compression")
    ax2.set_ylabel(ax2_label, color='red')
    ax2.tick_params(axis='y', colors='red')

    # Set title and legend
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.show()


def plot_quantization_and_memory(data, ax=None):
    """
    Plot an area chart showing the distribution of quantization types over episodes
    and a line showing memory saved.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract episode summaries
    quant_types_by_episode = []
    memory_saved = []
    
    for episode in data:
        if "episode_summary" in episode and "layer_bits" in episode["episode_summary"]:
            quant_types_by_episode.append(episode["episode_summary"]["layer_bits"])
            memory_saved.append(episode["episode_summary"]["memory_saved"])
    
    # Create a DataFrame for the stacked area chart
    episodes = list(range(len(quant_types_by_episode)))
    quant_counts = {
        "fp32": [], "fp16": [], "int8": [], "fp4": [], "nf4": []
    }
    
    # Count each type per episode
    for episode_types in quant_types_by_episode:
        counts = Counter(episode_types)
        total = len(episode_types)
        
        # Populate counts, default to 0 if not present
        quant_counts["fp32"].append(counts.get("fp32", 0) / total * 100)
        quant_counts["fp16"].append(counts.get("fp16", 0) / total * 100)
        quant_counts["int8"].append(counts.get("int8", 0) / total * 100)
        quant_counts["fp4"].append(counts.get("fp4", 0) / total * 100)
        quant_counts["nf4"].append(counts.get("nf4", 0) / total * 100)
   
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'episode': episodes,
        'fp32': quant_counts["fp32"],
        'fp16': quant_counts["fp16"],
        'int8': quant_counts["int8"],
        'fp4': quant_counts["fp4"],
        'nf4': quant_counts["nf4"]
    })
    
    # Create stacked area chart
    ax.stackplot(df['episode'], 
                 df['fp32'], df['fp16'], df['int8'], df['fp4'], df['nf4'],
                 labels=['fp32', 'fp16', 'int8', 'fp4', 'nf4'],
                 colors=['#ff9999', '#ffcc99', '#99ff99', '#66b3ff', '#c2c2f0'],
                 alpha=0.7)
    
    # Create twin axis for memory saved
    ax2 = ax.twinx()
    ax2.plot(episodes, [m * 100 for m in memory_saved], 'r-', linewidth=2, label='Memory Saved')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Memory Saved (%)', color='red')
    ax2.tick_params(axis='y', colors='red')
    
    # Set labels and title for the primary axis
    ax.set_xlabel('Episode')
    ax.set_ylabel('Distribution of Quantization Types (%)')
    ax.set_title('Quantization Types Distribution and Memory Saved Over Episodes')
    ax.set_xlim(0, len(episodes) - 1)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Show legends for both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), ncol=6)
    
    return ax

def plot_three_scales(x, vals1, vals2, vals3, x_label, y_label1, y_label2, y_label3, title, save_path):
    """
    Plots three different sets of values on three Y-axes with different scales.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot first dataset (Left Y-axis)
    ax1.plot(x, vals1, 'b-', label=y_label1)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label1, color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create second Y-axis (Right)
    ax2 = ax1.twinx()
    ax2.plot(x, vals2, 'g-', label=y_label2)
    ax2.set_ylabel(y_label2, color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # Create third Y-axis (Second Right)
    ax3 = ax1.twinx()
    ax3.plot(x, vals3, 'r-', label=y_label3)
    ax3.set_ylabel(y_label3, color='r')
    ax3.tick_params(axis='y', labelcolor='r')

    # Offset the third Y-axis to prevent overlap
    ax3.spines["right"].set_position(("outward", 60))

    # Add legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax3.legend(loc="lower right")

    # Set title and save
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved plot: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot quantization and RL metrics from JSON data")
    parser.add_argument("json_file", type=str, help="Path to JSON file")
    # parser.add_argument("quant_plot", type=str, help="Output path for quantization plot")
    # parser.add_argument("ppo_plot", type=str, help="Output path for PPO loss plot")
    args = parser.parse_args()

    # Load data
    data = load_data(args.json_file)

    data = [
        ep['episode_summary']
        for ep in data
    ]

    reward, policy_loss, baseline_loss = zip(*[ 
        (pt['reward'], pt['policy_loss'], pt['baseline_loss'])
        for pt in data
    ])

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_quantization_and_memory(data, ax)
    plt.savefig(args.quant_plot, bbox_inches='tight')
    print(f"Saved plot: {args.quant_plot}")

    # Generate random data for PPO, Reward, Baseline Loss
    timesteps = np.linspace(0, 100, 100)
    vals1 = np.exp(-timesteps / 50) + np.random.normal(0, 0.05, size=100)  # PPO Loss
    vals2 = np.sin(timesteps / 10) * 50 + 200  # Reward
    vals3 = np.exp(-timesteps / 70) + np.random.normal(0, 0.02, size=100)  # Baseline Loss

    # Save PPO Loss Plot
    plot_three_scales(
        timesteps, vals1, vals2, vals3,
        x_label="Timesteps",
        y_label1="PPO Loss",
        y_label2="Reward",
        y_label3="Baseline Loss",
        title="PPO Loss, Reward, and Baseline Loss Over Time",
        save_path="losses_plot.png"
    )

if __name__ == "__main__":
    main()
