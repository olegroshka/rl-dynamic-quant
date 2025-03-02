import argparse
import os
import re
import matplotlib.pyplot as plt
import pandas as pd


def parse_log_file(log_file):
    pattern = re.compile(
        r"INFO.*Episode (\d+), reward=([\d.-]+), final_loss=([\d.]+), "
        r"policy_loss=([\d.-]+), baseline_loss=([\d.-]+), qtypes=\[(.*?)\]"
    )

    episodes, rewards, final_losses, policy_losses, baseline_losses, quantization_types = [], [], [], [], [], []

    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                episodes.append(int(match.group(1)))
                rewards.append(float(match.group(2)))
                final_losses.append(float(match.group(3)))
                policy_losses.append(float(match.group(4)))
                baseline_losses.append(float(match.group(5)))
                bits_list = match.group(6).replace("'", "").split(", ")
                quantization_types.extend(bits_list)

    return pd.DataFrame({
        'episode': episodes,
        'reward': rewards,
        'final_loss': final_losses,
        'policy_loss': policy_losses,
        'baseline_loss': baseline_losses
    }), pd.Series(quantization_types).value_counts()


def plot_reward(df, experiment_name):
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['reward'], marker='o', label=experiment_name)
    plt.title('Total Reward Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join("results", experiment_name, f'{experiment_name}-reward.png'))
    plt.close()


def plot_losses(df, experiment_name):
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['final_loss'], label=f'{experiment_name} Final Loss', marker='o')
    plt.plot(df['episode'], df['policy_loss'], label=f'{experiment_name} Policy Loss', marker='x')
    plt.plot(df['episode'], df['baseline_loss'], label=f'{experiment_name} Baseline Loss', marker='.')
    plt.title('Loss Metrics Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join("results", experiment_name, f'{experiment_name}-losses.png'))
    plt.close()


def plot_quantization_distribution(qtypes_series, experiment_name):
    plt.figure(figsize=(8, 6))
    qtypes_series.plot(kind='bar', color='skyblue')
    plt.title(f'Quantization Type Distribution - {experiment_name}')
    plt.xlabel('Quantization Type')
    plt.ylabel('Count')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join("results", experiment_name, f'{experiment_name}-quantization_distribution.png'))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PPO RL Quantization Logs")
    parser.add_argument("--name", type=str, default="gpt-2-EAQ-500", help="Name for the experiment")
    args = parser.parse_args()

    log_file = os.path.join("results", args.name, 'ppo.log')

    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")

    data_df, quantization_distribution = parse_log_file(log_file)

    experiment_name = args.name
    plot_reward(data_df, experiment_name)
    plot_losses(data_df, experiment_name)
    plot_quantization_distribution(quantization_distribution, experiment_name)

    print(f"Plots saved for experiment: {experiment_name}")
