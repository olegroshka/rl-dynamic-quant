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
def plot_layer_quantization(layer_bits, memory_saved, x_label, y_label1, y_label2, title, save_path):
    """
    Plot an area chart showing the distribution of quantization types per layer over episodes
    and a line showing memory saved.

    Parameters:
    - layer_bits: List of lists (size: 250x12), each sublist contains quantization types per layer.
    - memory_saved: List of 250 values representing memory saved per episode.
    - x_label: Label for the x-axis.
    - y_label1: Label for the primary y-axis (quantization distribution).
    - y_label2: Label for the secondary y-axis (memory saved).
    - title: Title of the plot.
    - save_path: Path where to save the plot.
    """
    # Create figure and primary axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a DataFrame for the stacked area chart
    episodes = list(range(len(layer_bits)))
    # Get unique data types from flattened layer_bits
    unique_types = sorted(set([dtype for episode in layer_bits for dtype in episode]))
    quant_counts = {dtype: [] for dtype in unique_types}
    
    # Count each type per episode
    for episode_types in layer_bits:
        counts = Counter(episode_types)
        total = len(episode_types)
        
        # Populate counts for each type, default to 0 if not present
        for dtype in unique_types:
            quant_counts[dtype].append(counts.get(dtype, 0) / total * 100)
   
    # Create a DataFrame for easier plotting
    df_data = {'episode': episodes}
    df_data.update({dtype: quant_counts[dtype] for dtype in unique_types})
    df = pd.DataFrame(df_data)
    
    # Create color map for types
    colors = {
        'fp32': '#ff9999',
        'fp16': '#ffcc99',
        'int8': '#99ff99', 
        'fp4': '#66b3ff',
        'nf4': '#c2c2f0'
    }
    type_colors = [colors.get(dtype, f'#{hash(dtype) % 0xffffff:06x}') for dtype in unique_types]
    
    # Create stacked area chart
    ax.stackplot(df['episode'],
                 [df[dtype] for dtype in unique_types],
                 labels=unique_types,
                 colors=type_colors,
                 alpha=0.7)
    
    # Create twin axis for memory saved
    ax2 = ax.twinx()
    ax2.plot(episodes, [m * 100 for m in memory_saved], 'r-', linewidth=2, label='Memory Saved')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel(y_label2, color='red')
    ax2.tick_params(axis='y', colors='red')
    
    # Set labels and title for the primary axis
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label1)
    ax.set_title(title)
    ax.set_xlim(0, len(episodes) - 1)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Show legends for both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), ncol=6)
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved plot: {save_path}")

def plot_layer_distribution(layer_bits, x_label, y_label, title, save_path):
    """
    Plot a stacked bar chart showing distribution of quantization types across layers.

    Parameters:
    - layer_bits: List of lists, where each inner list contains quantization types for each layer
                 e.g. [[int8, fp16, fp32,...], [fp16, int8, fp32,...], ...]
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - title: Title of the plot.
    - save_path: Path where to save the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    num_layers = 12
    layers = range(num_layers)
    
    # Get unique data types from flattened layer_bits
    unique_types = sorted(set([dtype for episode in layer_bits for dtype in episode]))
    
    # Initialize counters for each layer and type
    layer_type_counts = {layer: {dtype: 0 for dtype in unique_types} 
                        for layer in layers}
    
    # Count occurrences of each type per layer
    for layer_types in layer_bits:  # Each element is a list of types for all layers
        for layer_idx, layer_type in enumerate(layer_types[:num_layers]):
            layer_type_counts[layer_idx][layer_type] += 1
    
    # Calculate percentages and create stacked bars
    bottoms = np.zeros(num_layers)
    
    # Fixed colors matching the other plots
    colors = {
        'fp32': '#ff9999',
        'fp16': '#ffcc99',
        'int8': '#99ff99', 
        'fp4': '#66b3ff',
        'nf4': '#c2c2f0'
    }
    type_colors = [colors.get(dtype, f'#{hash(dtype) % 0xffffff:06x}') for dtype in unique_types]
    
    # For each quantization type
    for qtype in unique_types:
        # Get percentages for this type across all layers
        percentages = []
        for layer in layers:
            total = sum(layer_type_counts[layer].values())
            count = layer_type_counts[layer][qtype]
            percentages.append(count / total * 100 if total > 0 else 0)
            
        # Create the bar segment
        ax.bar(layers, percentages, bottom=bottoms, label=qtype,
               color=colors.get(qtype, f'#{hash(qtype) % 0xffffff:06x}'),
               alpha=0.7)
        bottoms += percentages
    
    # Customize the plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
             ncol=5, title='Quantization Types')
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved plot: {save_path}")


def plot_three_scales(vals1, vals2, vals3, x_label, y_label1, y_label2, y_label3, title, save_path):
    """
    Plots three different sets of values on three Y-axes with different scales.
    """
    x = range(len(vals1))
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


def plot_single_layer_types(layer_types, x_label, y_label, title, save_path):
    """
    Plot a single instance of layer types with different markers for different quantization types.
    Same bit widths share the same y-value but have different markers.

    Parameters:
    - layer_types: List of strings, each string is a quantization type (e.g., ['fp32', 'int8', 'nf4', ...])
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - title: Title of the plot.
    - save_path: Path where to save the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define bit width classes and their corresponding y-values (equally spaced)
    bit_width_classes = {
        'fp32': 4,  # Class 4 (highest)
        'bf16': 3,  # Class 3
        'fp16': 3,  # Class 3 
        'int8': 2,  # Class 2
        'fp4': 1,   # Class 1 (lowest)
        'nf4': 1    # Class 1
    }
    
    # Define unique markers for each unique type
    markers = {
        'fp32': 'o',  # circle
        'bf16': 's',  # square
        'fp16': '^',  # triangle up
        'int8': 'D',  # diamond
        'fp4': 'v',   # triangle down
        'nf4': '<'    # triangle left
    }
    
    # Define colors for each type
    colors = {
        'fp32': '#ff9999',
        'bf16': '#ffcc99',
        'fp16': '#ffcc99',
        'int8': '#99ff99',
        'fp4': '#66b3ff',
        'nf4': '#c2c2f0'
    }
    
    # Plot each layer type
    layers = range(len(layer_types))
    plotted_types = set()  # Track which types we've already added to legend
    
    for layer_idx, qtype in enumerate(layer_types):
        y_val = bit_width_classes[qtype]
        # Only add to legend if we haven't seen this type before
        if qtype not in plotted_types:
            ax.scatter(layer_idx, y_val, 
                      marker=markers[qtype],
                      color=colors[qtype],
                      s=100,
                      label=qtype)
            plotted_types.add(qtype)
        else:
            ax.scatter(layer_idx, y_val,
                      marker=markers[qtype],
                      color=colors[qtype],
                      s=100)
    
    # Customize the plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.set_yticks([1, 2, 3, 4])  # Four equally spaced classes
    ax.set_yticklabels(['4-bit', '8-bit', '16-bit', '32-bit'])  # Label the classes
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
             ncol=3, title='Quantization Types')
    
    # Save the plot
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

    layer_bits, memory_saved = zip(*[ 
        (pt['layer_bits'], pt['memory_saved'])
        for pt in data
    ])

   # Save PPO Loss Plot
    plot_three_scales(
        reward, policy_loss, baseline_loss,
        x_label="Timesteps",
        y_label1="PPO Loss",
        y_label2="Reward",
        y_label3="Baseline Loss",
        title="PPO Loss, Reward, and Baseline Loss Over Time",
        save_path="report/losses_plot.png"
    )

    # Save Layer Quantization Plot
    plot_layer_quantization(
        layer_bits, memory_saved,
        x_label="Episode",
        y_label1="Distribution of Quantization Types (%)",
        y_label2="Memory Saved (%)", 
        title="Layer-wise Quantization Types Distribution and Memory Saved Over Episodes",
        save_path="report/layer_quantization_plot.png"
    )

    # Save Layer Distribution Plot
    plot_layer_distribution(
        layer_bits,
        x_label="Layer",
        y_label="Distribution of Quantization Types (%)",
        title="Quantization Type Distribution Across Layers",
        save_path="report/layer_distribution_plot.png"
    )

if __name__ == "__main__":
    main()
