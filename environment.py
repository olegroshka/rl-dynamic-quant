import torch
import torch.nn as nn
import numpy as np

from quant_utils import apply_quantization_to_layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QuantizationEnv:
    """
    RL Environment that:
      - Holds a GPT-2-like model
      - Dynamically quantizes layer by layer
      - Fine-tunes for a small number of steps to measure performance
      - Returns a reward that balances performance and bit usage
    """

    def __init__(self, model, train_loader, val_loader, reference_model, dev=device, model_name="gpt2", finetune_steps=5, reward_scaling=0.01, lr=5e-5):
        self.device = dev
        self.model_name = model_name
        self.finetune_steps = finetune_steps
        self.reward_scaling = reward_scaling

        # The main model we'll quantize
        self.model = model.to(dev)

        # A reference model for baseline loss comparisons
        self.reference_model = reference_model.to(dev)
        self.lr=lr

        self.train_loader = train_loader
        self.val_loader = val_loader

        # GPT-2 blocks: a list of transformer layers
        self.layers = list(self.model.transformer.h)
        self.num_layers = len(self.layers)
        self.layer_index = 0

        self.criterion = nn.CrossEntropyLoss()

        # This will store the chosen bit per layer for the entire episode
        self.chosen_bits_for_layer = [-1] * self.num_layers

        # Evaluate baseline loss using the reference model
        self.baseline_loss = self.evaluate_loss(self.reference_model)

    def reset(self):
        """
        Resets the environment for a new RL episode:
          - reload model weights from reference
          - reset layer index
          - recalculate baseline loss
        """
        # Reload from reference
        self.model.load_state_dict(self.reference_model.state_dict())
        self.layer_index = 0

        # Evaluate loss after reset if you prefer
        self.baseline_loss = self.evaluate_loss(self.model)

        init_state = np.array([self.baseline_loss, 0, self.layer_index, 0, 0], dtype=np.float32)
        # Reset chosen bits for new episode
        self.chosen_bits_for_layer = [-1] * self.num_layers

        return init_state

    def step(self, action_bits):
        # 1. Quantize
        apply_quantization_to_layer(self.layers[self.layer_index], action_bits)

        # 2. Fine-tune
        self.fine_tune(steps=self.finetune_steps)

        # 3. Evaluate
        new_loss = self.evaluate_loss(self.model)
        delta_loss = new_loss - self.baseline_loss
        reward = -(delta_loss + self.reward_scaling * action_bits)
        self.baseline_loss = new_loss

        # 4. Record bits for the current layer
        self.chosen_bits_for_layer[self.layer_index] = action_bits

        # 5. Move to next layer
        self.layer_index += 1
        done = (self.layer_index >= self.num_layers)

        # Build next_state from the layer we just quantized
        cur_layer = self.layers[self.layer_index - 1]
        with torch.no_grad():
            w = cur_layer.attn.c_attn.weight
            w_mean = w.mean().item()
            w_std = w.std().item()

        next_state = np.array([new_loss, action_bits, self.layer_index, w_mean, w_std], dtype=np.float32)

        info = {}
        if done:
            info["chosen_bits_for_layer"] = self.chosen_bits_for_layer[:]

        return next_state, reward, done, info

    def fine_tune(self, steps=5):
        """
        Minimal fine-tuning loop on training data loader.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.model.train()
        step_count = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = input_ids.clone()

            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            step_count += 1
            if step_count >= steps:
                break

    def evaluate_loss(self, model):
        """
        Evaluate cross-entropy loss on the validation set.
        """
        model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = input_ids.clone()

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                batch_size = input_ids.size(0)

                total_loss += loss.item() * batch_size
                count += batch_size

        return total_loss / count if count > 0 else 0.0
