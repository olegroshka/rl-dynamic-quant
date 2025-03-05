import argparse
import json
import logging
import os

import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from data_handler import DataHandler

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ModelCreator:
    def __init__(self, model_name="gpt2", bit_width=8, device=None):
        self.model_name = model_name
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        # model_8bit = AutoModelForCausalLM.from_pretrained(
        #     "gpt2",
        #     load_in_4bit=bit_width == 4,  # This triggers 4-bit weights
        #     load_in_8bit=bit_width == 8,  # This triggers 8-bit weights
        #     device_map="auto",  # Places modules automatically
        # )

        self.model.to(self.device)

    def fine_tune(self, train_loader, epochs=3, lr=5e-5, max_grad_norm=1.0):
        """Fine-tune the model on the dataset"""
        logger.info(f"Fine-tuning {self.model_name} for {epochs} epochs")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            steps = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                if torch.isnan(loss):
                    logger.warning(f"Encountered NaN loss at step {steps}. Skipping batch.")
                    continue  # skip batches that produce nan

                loss.backward()

                # Gradient clipping to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()

                epoch_loss += loss.item()
                steps += 1
                progress_bar.set_postfix({"loss": epoch_loss / steps})

            avg_loss = epoch_loss / steps if steps > 0 else float('inf')
            logger.info(f"Epoch {epoch + 1}/{epochs} - Average loss: {avg_loss:.4f}")

    def evaluate(self, val_loader):
        """Evaluate the model on the validation set"""
        logger.info("Evaluating model performance")

        self.model.eval()
        total_loss = 0.0
        total_valid_tokens = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)  # use pre-computed partial labels

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                valid_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * valid_tokens
                total_valid_tokens += valid_tokens

        if total_valid_tokens == 0:
            logger.error("No valid tokens to evaluate. Did something go wrong with labels?")
            return {"loss": float('inf'), "perplexity": float('inf')}

        avg_loss = total_loss / total_valid_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        logger.info(f"Validation Loss: {avg_loss:.4f}")
        logger.info(f"Validation Perplexity: {perplexity:.4f}")

        return {"loss": avg_loss, "perplexity": perplexity}

    def save_model(self, output_dir):
        """Save the model and tokenizer"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Create a reference model.")
    parser.add_argument("--name", type=str, required=True, help="Name for the baseline experiment")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name (default: gpt2)")
    parser.add_argument("--bit_width", type=int, default=8, help="Quantization bit width (default: 8)")
    parser.add_argument("--dataset", type=str, default="commonsense_qa", help="Dataset name (default: commonsense_qa)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training (default: 8)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs (default: 3)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate (default: 5e-5)")

    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.join("results", args.name)
    if os.path.exists(output_dir):
        logger.warning(f"Output directory {output_dir} already exists. Files may be overwritten.")

    # Create and configure the model
    creator = ModelCreator(model_name=args.model, bit_width=args.bit_width)

    # Load dataset
    data_handler = DataHandler(dataset_name=args.dataset, batch_size=args.batch_size, max_length=128)
    train_loader, val_loader = data_handler.load_dataset()

    # Fine-tune the model
    creator.fine_tune(train_loader, epochs=args.epochs, lr=args.learning_rate)

    # Evaluate before quantization
    logger.info("Evaluating")
    eval_results = creator.evaluate(val_loader)

    # Save model and results
    creator.save_model(output_dir)

    results = {
        "model_name": args.model,
        "bit_width": args.bit_width,
        "dataset": args.dataset,
        "eval_results": eval_results
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Baseline creation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
