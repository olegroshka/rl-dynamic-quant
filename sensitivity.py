# sensitivity.py

import argparse
import json
import torch
from transformers import GPT2LMHeadModel
from data_handler import DataHandler

class SensitivityRecorder:
    def __init__(self, model, output_file):
        self.model = model
        self.output_file = output_file
        self.sensitivities = {}

    def record_sensitivities(self, train_loader, num_steps=100):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for step in range(num_steps):
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = input_ids.clone()

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()

                # Record sensitivities (gradient norms)
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        sensitivity = param.grad.norm().item()
                        if name not in self.sensitivities:
                            self.sensitivities[name] = []
                        self.sensitivities[name].append(sensitivity)

                optimizer.step()

        # Save sensitivities to JSON
        with open(self.output_file, "w") as f:
            json.dump(self.sensitivities, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=100, help="Number of fine-tuning steps")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file for sensitivities")
    parser.add_argument("--dataset", type=str, default="commonsense_qa", help="Dataset to use (commonsense_qa or openbookqa)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # Load dataset using DataHandler
    data_handler = DataHandler(dataset_name=args.dataset, batch_size=8, max_length=128)
    train_loader, _ = data_handler.load_dataset()

    # Record sensitivities
    recorder = SensitivityRecorder(model, args.output_file)
    recorder.record_sensitivities(train_loader, args.num_steps)

if __name__ == "__main__":
    main()