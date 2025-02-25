# eval.py

import argparse
import os
import torch
from transformers import GPT2LMHeadModel
from data_handler import DataHandler

def evaluate_perplexity(model, data_loader, dev="cuda"):
    """
    Compute average perplexity = exp( average cross-entropy ) over data_loader.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            batch_tokens = input_ids.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True, help="Name of the baseline model")
    parser.add_argument("--model", type=str, required=True, help="Name of the trained model")
    parser.add_argument("--dataset", type=str, default="commonsense_qa", help="Dataset to use (commonsense_qa or openbookqa)")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to load models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    baseline_model = GPT2LMHeadModel.from_pretrained(os.path.join(args.results_dir, args.baseline)).to(device)
    trained_model = GPT2LMHeadModel.from_pretrained(os.path.join(args.results_dir, args.model)).to(device)

    # Load dataset using DataHandler
    data_handler = DataHandler(dataset_name=args.dataset, batch_size=8, max_length=128)
    _, val_loader = data_handler.load_dataset()

    # Evaluate models
    baseline_loss = evaluate_perplexity(baseline_model, val_loader, device)
    trained_loss = evaluate_perplexity(trained_model, val_loader, device)

    print(f"Baseline Model Perplexity: {baseline_loss}")
    print(f"Trained Model Perplexity: {trained_loss}")

if __name__ == "__main__":
    main()