# train_old.py

import argparse
import os
import torch
from environment import QuantizationEnv
from ppo_trainer_old import ppo_train
from data_handler import DataHandler
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--dataset", type=str, default="commonsense_qa", help="Dataset to use (commonsense_qa or openbookqa)")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load dataset using DataHandler
    data_handler = DataHandler(dataset_name=args.dataset, batch_size=8, max_length=128)
    train_loader, val_loader = data_handler.load_dataset()

    # Create environment
    env = QuantizationEnv(model, train_loader, val_loader, model, device)

    # Train the RL policy
    ppo_train(env, num_episodes=10)

    # Save the model and results
    os.makedirs(args.results_dir, exist_ok=True)
    model.save_pretrained(os.path.join(args.results_dir, args.experiment_name))
    print(f"Model and results saved to {args.results_dir}/{args.experiment_name}")

if __name__ == "__main__":
    main()