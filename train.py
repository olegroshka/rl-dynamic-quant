#!/usr/bin/env python
# train.py - Train models with RL-based quantization

import argparse
import json
import logging
import os

import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from environment import QuantizationEnv
from enhanced_env import EnhancedQuantizationEnv, LayerStats
from policy import PolicyNet
from ppo_trainer import ppo_train
from data_handler import DataHandler
from dataclasses import asdict

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def evaluate_model(model, val_loader, device):
    """Evaluate the model on the validation set"""
    logger.info("Evaluating model performance")

    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            batch_size = input_ids.size(0)

            total_loss += loss.item() * batch_size
            count += batch_size

    perplexity = torch.exp(torch.tensor(total_loss / count)).item()
    logger.info(f"Validation Loss: {total_loss / count:.4f}")
    logger.info(f"Validation Perplexity: {perplexity:.4f}")

    return {"loss": total_loss / count, "perplexity": perplexity}


def save_model_and_results(model,
                           tokenizer,
                           output_dir,
                           results,
                           chosen_bits_per_layer=None,
                           rl_algorithm=None,
                           hyperparams=None,
                           all_rewards=None,
                           info_stats=None):
    """Save the model, tokenizer, and training results"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save model
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save results
    results_data = {
        "evaluation_results": results,
        "chosen_bits_per_layer": chosen_bits_per_layer,
        "rl_algorithm": rl_algorithm,
        "hyperparameters": hyperparams
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results_data, f, indent=2)

    with open(os.path.join(output_dir, "all_rewards.json"), "w") as f:
        json.dump(all_rewards, f, indent=2)

    cleaned_info_stats = []

    for entry in info_stats:
        cleaned_entry = {}
        for k, v in entry.items():
            # If the value is LayerStats, convert to dict
            if isinstance(v, LayerStats):
                cleaned_entry[k] = asdict(v)
            else:
                cleaned_entry[k] = v
        cleaned_info_stats.append(cleaned_entry)

    # Now dump cleaned_info_stats
    with open(os.path.join(output_dir, "info_stats.json"), "w") as f:
        json.dump(cleaned_info_stats, f, indent=2)

    logger.info(f"Results saved to {output_dir}/results.json")


def main():
    parser = argparse.ArgumentParser(description="Train a model with RL-based quantization")
    parser.add_argument("--name", type=str, required=True, help="Name for the experiment")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name (default: gpt2)")
    parser.add_argument("--dataset", type=str, default="commonsense_qa", help="Dataset name (default: commonsense_qa)")
    parser.add_argument("--rl_algorithm", type=str, default="ppo", choices=["ppo"], help="RL algorithm (default: ppo)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training (default: 8)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of RL episodes (default: 10)")
    parser.add_argument("--state_dim", type=int, default=6, help="State dimension for policy network (default: 6)") # 5 for old env
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension for policy network (default: 32)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clipping parameter (default: 0.2)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for policy network (default: 1e-3)")
    parser.add_argument("--finetune_steps", type=int, default=5, help="Number of fine-tuning steps per layer (default: 5)")
    parser.add_argument("--bit_options", type=str, default="2,4,8,16", help="Comma-separated list of bit-width options (default: 2,4,8,16)")

    args = parser.parse_args()

    # Parse bit options
    bit_options = [int(b) for b in args.bit_options.split(',')]

    # Create output directory
    output_dir = os.path.join("results", args.name)
    if os.path.exists(output_dir):
        logger.warning(f"Output directory {output_dir} already exists. Files may be overwritten.")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(args.model)
    # Create a reference copy of the model
    reference_model = GPT2LMHeadModel.from_pretrained(args.model)

    # Move models to device
    model.to(device)
    reference_model.to(device)

    # Prepare dataset
    data_handler = DataHandler(dataset_name=args.dataset, batch_size=args.batch_size, max_length=128)
    train_loader, val_loader = data_handler.load_dataset()

    # Create RL environment
    # env = QuantizationEnv(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     reference_model=reference_model,
    #     dev=device,
    #     model_name=args.model,
    #     finetune_steps=args.finetune_steps
    # )

    env = EnhancedQuantizationEnv(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        reference_model=reference_model,
        finetune_steps=args.finetune_steps
    )

    # Create policy network
    num_actions = len(bit_options)
    policy = PolicyNet(state_dim=args.state_dim, hidden_dim=args.hidden_dim, num_actions=num_actions)

    # Collect hyperparameters for saving
    hyperparams = {
        "state_dim": args.state_dim,
        "hidden_dim": args.hidden_dim,
        "num_actions": num_actions,
        "bit_options": bit_options,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "lr": args.lr,
        "finetune_steps": args.finetune_steps,
        "episodes": args.episodes
    }

    # Train with the specified RL algorithm
    logger.info(f"Starting training with {args.rl_algorithm}, {args.episodes} episodes")
    if args.rl_algorithm == "ppo":
        final_info, all_rewards, info_stats = ppo_train(
            env=env,
            policy=policy,
            num_episodes=args.episodes,
            gamma=args.gamma,
            epsilon=args.epsilon,
            lr=args.lr,
            dev=device,
            bit_options=bit_options,
            results_path=output_dir
        )
    else:
        raise ValueError(f"Unsupported RL algorithm: {args.rl_algorithm}")

    # Get the chosen bits per layer
    summary = final_info["episode_summary"]
    chosen_bits_per_layer = summary.get("layer_bits", [])

    # Evaluate the final model
    evaluation_results = evaluate_model(model, val_loader, device)

    # Save model, tokenizer, and results
    save_model_and_results(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        results=evaluation_results,
        chosen_bits_per_layer=chosen_bits_per_layer,
        rl_algorithm=args.rl_algorithm,
        hyperparams=hyperparams,
        all_rewards=all_rewards,
        info_stats=info_stats
    )

    logger.info(f"Training complete. Model and results saved to {output_dir}")


if __name__ == "__main__":
    main()