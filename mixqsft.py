#!/usr/bin/env python
# mixqsft.py

import argparse
import json
import logging
import os
import ast

import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from data_handler import DataHandler
from quantizer import Quantizer, MixedQuantizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="SFT with a mixed (layer-wise) quantization schema.")
    parser.add_argument("--name", type=str, required=True, help="Name for the experiment (subfolder in results/)")
    parser.add_argument("--model", type=str, default="gpt2", help="Base model name (default: gpt2)")
    parser.add_argument("--quant_schema", type=str, required=True, help="List of quantization types, e.g. ['fp16','nf4','int8',...]")
    parser.add_argument("--dataset", type=str, default="commonsense_qa", help="Dataset name (default: commonsense_qa)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    quant_schema = ast.literal_eval(args.quant_schema)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.to(device)
    model.train()

    logger.info(f"Applying per-layer schema: {quant_schema}")
    base_quantizer = Quantizer(
        compute_dtype=torch.float16,  # or torch.bfloat16, as desired
        compress_statistics=True,
        quant_storage=torch.uint8
    )
    mixed_quant = MixedQuantizer(quant_schema, base_quantizer)
    model = mixed_quant.quantize_model(model, layer_attribute="transformer.h")

    data_handler = DataHandler(dataset_name=args.dataset, batch_size=args.batch_size, max_length=128)
    train_loader, val_loader = data_handler.load_dataset()

    logger.info(f"Fine-tuning model with mixed quantization for {args.epochs} epochs.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        steps = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in progress_bar:
            model.train()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if torch.isnan(loss):
                logger.warning(f"NaN loss at step {steps}, skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1
            global_step += 1
            progress_bar.set_postfix({"loss": f"{(epoch_loss/steps):.4f}"})

        avg_loss = epoch_loss / steps if steps > 0 else float('inf')
        logger.info(f"[Epoch {epoch+1}] avg loss = {avg_loss:.4f}")

    logger.info("Evaluating final perplexity on validation set (optional quick check).")
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            batch_tokens = input_ids.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    if total_tokens == 0:
        perplexity = float("inf")
    else:
        avg_val_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()

    logger.info(f"Validation perplexity: {perplexity:.4f}")

    output_dir = os.path.join("results", args.name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving final model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    results_dict = {
        "model_name": args.model,
        "quant_schema": quant_schema,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "val_perplexity": perplexity
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results_dict, f, indent=2)

    logger.info("Mixed quantization SFT complete. Results saved.")

if __name__ == "__main__":
    main()
