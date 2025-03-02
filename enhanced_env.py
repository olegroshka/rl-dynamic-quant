from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quantizer import Quantizer, Q_TYPE_SIZE


@dataclass
class LayerStats:
    """Track per-layer statistics for better state representation"""
    weight_mean: float
    weight_std: float
    gradient_norm: float
    attention_entropy: float  # measure of pattern diversity
    layer_idx: int
    current_quant_type: str

class EnhancedQuantizationEnv:
    """
    Enhanced RL Environment that:
      - Holds a GPT-2-like model
      - Dynamically quantizes layer by layer
      - Fine-tunes adaptively for a small number of steps
      - Tracks extended metrics (attention entropy, gradient norm, etc.)
      - Uses a more comprehensive reward function
      - Implements curriculum learning for bit choices
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        reference_model: nn.Module,
        quantizer: Quantizer,
        device: torch.device = None,
        curriculum_schedule: Optional[Dict] = None,
        reward_weights: Optional[Dict] = None,
        finetune_steps: int = 5,
        lr: float = 5e-5
    ):
        """
        Args:
            model: GPT-2-like model to be quantized/fine-tuned
            train_loader: training dataloader
            val_loader: validation dataloader
            reference_model: same architecture; used for baseline metrics
            device: torch device
            reward_weights: weighting for performance, memory, and stability components
            finetune_steps: number of steps for adaptive fine-tuning each layer
            lr: learning rate for fine-tuning
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.reference_model = reference_model.to(self.device)
        self.train_loader = train_loader
        self.quantizer = quantizer
        self.val_loader = val_loader
        self.finetune_steps = finetune_steps
        self.lr = lr

        # GPT-2 blocks: a list of transformer layers
        self.layers = list(self.model.transformer.h)
        self.num_layers = len(self.layers)
        self.layer_idx = 0  # which layer we're working on now

        # Enhanced layer statistics
        self.layer_stats = [LayerStats(0, 0, 0, 0, i, "fp32") for i in range(self.num_layers)]

        # Reward weights - todo calibrate these
        self.reward_weights = reward_weights or {
            'performance': 1.0, #1.0,
            'memory': 1.0, #0.3,
            'entropy': 1.0,
            'kl': 1.0
        }

        # Track global step for curriculum
        self.total_steps = 0

        # Compute and store baseline metrics (loss, memory usage, perplexity, etc.)
        self.baseline_metrics = self._compute_baseline_metrics()

    def _compute_baseline_metrics(self) -> Dict:
        """Compute metrics on the reference model (or unmodified model)."""
        with torch.no_grad():
            val_loss = self.evaluate_loss(self.reference_model)
            val_perplexity = float(torch.exp(torch.tensor(val_loss))) if val_loss < 20 else float('inf')
            memory_usage = sum(p.numel() * p.element_size() for p in self.reference_model.parameters())

        return {
            'loss': val_loss,
            'perplexity': val_perplexity,
            'memory': memory_usage,
            # pre-compute layer stats on reference model
            'layer_stats': self._get_layer_statistics(self.reference_model)
        }

    def _get_layer_statistics(self, model: nn.Module) -> List[LayerStats]:
        # 1) Do one forward pass (with attentions)
        entropies = self._compute_attention_entropy_all_layers(model)

        stats = []
        for i, layer in enumerate(model.transformer.h):
            with torch.no_grad():
                w = layer.attn.c_attn.weight
                w_mean = w.mean().item()
                w_std = w.std().item()
                grad_norm = w.grad.norm().item() if w.grad is not None else 0.0

            stats.append(
                LayerStats(
                    weight_mean=w_mean,
                    weight_std=w_std,
                    gradient_norm=grad_norm,
                    attention_entropy=entropies[i],
                    layer_idx=i,
                    current_quant_type=self.layer_stats[i].current_quant_type,
                )
            )
        return stats

    def _compute_attention_entropy_all_layers(self, model: nn.Module) -> List[float]:
        batch = next(iter(self.train_loader))
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )  # This returns attentions for all layers

        # outputs.attentions is a tuple of length num_layers
        # each element is [batch_size, num_heads, seq_len, seq_len]
        all_entropies = []
        for layer_idx, attn_weights in enumerate(outputs.attentions):
            # shape [batch_size, num_heads, seq_len, seq_len]
            entropy = -(attn_weights * (attn_weights + 1e-10).log()).sum(-1).mean()
            all_entropies.append(entropy.item())

        return all_entropies


    def reset(self):
        """
        Reset environment for a new episode:
        - Copy reference model weights into our main model
        - Reset layer index
        - Recompute baseline metrics if desired
        - Return initial state
        """
        self.model.load_state_dict(self.reference_model.state_dict())
        self.layer_idx = 0

        # Re-init layer_stats so that current_bits = -1 again
        self.layer_stats = [LayerStats(0, 0, 0, 0, i, -1) for i in range(self.num_layers)]

        # Could recompute the baseline on the fresh model if you want:
        # self.baseline_metrics = self._compute_baseline_metrics()

        # Build an initial state. We can re-use _build_state_representation,
        # but layer_idx=0 may not have interesting stats yet.
        # We'll just do a dummy evaluation or let everything be zero.
        init_state = self._build_state_representation()
        return init_state

    def step(self, action_quant_type: str) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step function that:
          1. Applies quantization to the current layer
          2. Fine-tunes the model for self.finetune_steps
          3. Evaluates new loss and computes reward
          4. Updates layer stats
          5. Builds next state
          6. Moves to next layer or ends episode
        """
        # 1. Quantize the current layer
        current_layer = self.layers[self.layer_idx]
        self._quantize_layer(current_layer, action_quant_type)

        # 2. Fine-tune
        self._adaptive_fine_tune()

        # 3. Evaluate new loss
        new_loss = self.evaluate_loss(self.model)
        # Compute reward
        reward, reward_info = self.compute_reward(
            new_loss,
            action_quant_type,
            self.layer_idx
        )

        # 4. Update layer stats for the layer we just quantized
        updated_layer_stats = self._get_layer_statistics(self.model)[self.layer_idx]
        self.layer_stats[self.layer_idx] = updated_layer_stats
        # Set the current_bits to the chosen action
        self.layer_stats[self.layer_idx].current_quant_type = action_quant_type

        # 5. Build next state
        next_state = self._build_state_representation()

        # 6. Move to next layer
        self.layer_idx += 1
        self.total_steps += 1  # global step for curriculum

        done = (self.layer_idx >= self.num_layers)

        info = {
            'reward_components': reward_info,
            'layer_stats': self.layer_stats[self.layer_idx - 1],
        }
        for i, layer_stat in enumerate(self.layer_stats):
            info[f'layer_stats_{i}'] = layer_stat

        if done:
            info['episode_summary'] = self._generate_episode_summary()

        return next_state, reward, done, info


    def _quantize_layer(self, layer: nn.Module, quant_type: str):
        for name, child in layer.named_children():
            if isinstance(child, nn.Linear):
                setattr(layer, name, self.quantizer.quantize_linear_layer(child, quant_type))
            else:
                self._quantize_layer(child, quant_type)


    def _adaptive_fine_tune(self):
        """
        Fine-tune the model for self.finetune_steps, using a simple training loop.
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
            if step_count >= self.finetune_steps:
                break

    def evaluate_loss(self, model: nn.Module) -> float:
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

    def compute_reward_old(self, new_loss: float, chosen_bits: int, layer_idx: int) -> Tuple[float, Dict]:
        baseline_loss = self.baseline_metrics['loss']

        # 1) Performance: simpler difference (baseline - new_loss)
        perf_diff = baseline_loss - new_loss
        #perf_diff = max(min(perf_diff, 2.0), -2.0)

        # Multiply by weight
        perf_reward = perf_diff * self.reward_weights['performance']

        # 2) Memory reward
        memory_factor = (16 - chosen_bits) / 16.0
        memory_reward = memory_factor * self.reward_weights['memory']

        # 3) Stability reward
        layer_stat = self.layer_stats[layer_idx]
        stability_factor = 1.0
        if layer_stat.attention_entropy > 0.5:
            stability_factor = 0.5
        stability_reward = stability_factor * self.reward_weights['stability']

        total_reward = perf_reward + memory_reward + stability_reward
        reward_info = {
            'performance': perf_reward,
            'memory': memory_reward,
            'stability': stability_reward,
            'total': total_reward
        }
        return total_reward, reward_info


    def compute_reward(self,
                       quant_loss: float,
                       quant_type: str,
                       layer_idx: int) -> Tuple[float, Dict]:
        """
        A new reward function that combines:
          1) Reference-vs-Quant Loss Delta
          2) KL Divergence to Reference
          3) Attention Entropy Preservation
          4) Memory (bit) savings

        We assume self.reward_weights dict has keys like:
          { 'performance': float, 'kl': float, 'entropy': float, 'memory': float }

        Returns:
            total_reward (float): The scalar reward
            reward_info (dict): Breakdown of individual reward components
        """

        # ------------------------------------------------------------------------
        # 1) Compare new quant model's loss to the reference model's loss
        #    "baseline_loss" was computed once at environment init.
        # ------------------------------------------------------------------------
        ref_loss = self.baseline_metrics["loss"]
        perf_diff = ref_loss - quant_loss
        perf_reward = perf_diff * self.reward_weights.get('performance', 1.0)

        # ------------------------------------------------------------------------
        # 2) KL Divergence from reference model
        #    We'll do a forward pass on some batch to get p (quant) and q (ref)
        #    If you haven't implemented a method for it, see _compute_kl_div()
        # ------------------------------------------------------------------------
        kl_value = self._compute_kl_divergence(
            quant_model=self.model,
            ref_model=self.reference_model
        )
        kl_reward = -kl_value * self.reward_weights.get('kl', 1.0)
        # Negative sign because we want to minimize KL => larger KL => negative reward

        # ------------------------------------------------------------------------
        # 3) Attention Entropy Preservation on the current layer
        #    We'll compare quant vs. reference attention on the current layer
        # ------------------------------------------------------------------------
        # Retrieve the attention entropy for *all* layers from each model
        ref_entropies = self._compute_attention_entropy_all_layers(self.reference_model)
        quant_entropies = self._compute_attention_entropy_all_layers(self.model)

        # Compare only the current layer's entropies for local layer feedback
        ref_layer_entropy = ref_entropies[layer_idx]
        quant_layer_entropy = quant_entropies[layer_idx]

        entropy_diff = quant_layer_entropy - ref_layer_entropy
        # If quant_layer_entropy < ref_layer_entropy, it's a negative difference => penalty
        # If quant_layer_entropy >= ref_layer_entropy, we want to reward that
        entropy_reward = entropy_diff * self.reward_weights.get('entropy', 1.0)

        # ------------------------------------------------------------------------
        # 4) Bits-based memory reward
        #    Fewer bits => more reward (like your existing memory approach)
        # ------------------------------------------------------------------------
        memory_factor = (16 - Q_TYPE_SIZE[quant_type]) / 16.0
        memory_reward = memory_factor * self.reward_weights.get('memory', 1.0)

        # ------------------------------------------------------------------------
        # Combine all components
        # ------------------------------------------------------------------------
        total_reward = perf_reward + kl_reward + entropy_reward + memory_reward

        reward_info = {
            'perf_reward': perf_reward,
            'kl_reward': kl_reward,
            'entropy_reward': entropy_reward,
            'memory_reward': memory_reward,
            'total_reward': total_reward
        }

        return total_reward, reward_info

    def _compute_kl_divergence(self, quant_model: nn.Module, ref_model: nn.Module) -> float:
        """
        Compute the average KL( p || q ) over one batch from self.train_loader or val_loader,
        where p = quant_model logits distribution, q = reference_model logits distribution.

        Returns:
            kl (float): the scalar average KL divergence
        """
        quant_model.eval()
        ref_model.eval()

        # sample one batch for quickness for now (todo: try can average over more).
        batch = next(iter(self.val_loader))
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
            # Get logits from both models
            quant_outputs = quant_model(input_ids, attention_mask=attention_mask)
            ref_outputs = ref_model(input_ids, attention_mask=attention_mask)

        # quant_logits, ref_logits: [batch_size, seq_len, vocab_size]
        quant_logits = quant_outputs.logits
        ref_logits = ref_outputs.logits

        # Convert logits -> probabilities
        #   p = softmax(quant_logits), q = softmax(ref_logits)
        #   small epsilon to avoid log(0)
        p = torch.nn.functional.softmax(quant_logits, dim=-1)
        q = torch.nn.functional.softmax(ref_logits, dim=-1)

        # KL(p||q) = sum( p * log(p/q) )
        # We'll do an elementwise approach: p * (log p - log q)
        kl_elementwise = p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))
        # Sum over vocab dimension => shape [batch_size, seq_len]
        kl_per_token = kl_elementwise.sum(dim=-1)
        # Then average over batch * sequence
        kl_avg = kl_per_token.mean().item()

        return kl_avg


    def compute_reward_prev(
        self,
        new_loss: float,
        chosen_bits: int,
        layer_idx: int
    ) -> Tuple[float, Dict]:
        """
        Enhanced reward computation considering multiple factors:
            - Performance impact (loss delta from baseline)
            - Memory efficiency (fewer bits -> higher reward)
            - Stability (penalize aggressive quant if layer has high attention entropy)
        """
        # 1) Performance: difference from baseline
        loss_delta = new_loss - self.baseline_metrics['loss']
        perf_reward = -loss_delta * self.reward_weights['performance']

        # 2) Memory: scale reward by how many bits we saved
        memory_factor = (16 - chosen_bits) / 16.0
        memory_reward = memory_factor * self.reward_weights['memory']

        # 3) Stability: if the layer has large attention entropy, penalize big changes
        #    For example, if entropy > 0.5, we reduce the stability reward
        layer_stat = self.layer_stats[layer_idx]
        stability_factor = 1.0
        if layer_stat.attention_entropy > 0.5:
            stability_factor = 0.5  # reduce reward for aggressive quantization
        stability_reward = stability_factor * self.reward_weights['stability']

        total_reward = perf_reward + memory_reward + stability_reward
        reward_info = {
            'performance': perf_reward,
            'memory': memory_reward,
            'stability': stability_reward,
            'total': total_reward
        }

        return total_reward, reward_info

    def _build_state_representation(self) -> np.ndarray:
        """
        Create a feature vector for the agent based on the current layer stats,
        plus any other relevant normalized signals.
        """
        # If we haven't quantized the current layer yet (happens at reset),
        # we might want to pull stats from the reference model or from self.layer_stats anyway.
        if self.layer_idx < self.num_layers:
            current_stats = self.layer_stats[self.layer_idx]
        else:
            # If we're past the last layer (done), just replicate something stable
            current_stats = self.layer_stats[-1]

        # Example 6D feature vector:
        #   [weight_mean, weight_std, gradient_norm, attention_entropy,
        #    normalized_layer_idx, normalized_current_bits]
        # Feel free to add more or remove some depending on your experiments.
        qtype_size = Q_TYPE_SIZE.get(current_stats.current_quant_type, 32)  # fallback to 32 if unknown
        state = np.array([
            current_stats.weight_mean,
            current_stats.weight_std,
            current_stats.gradient_norm,
            current_stats.attention_entropy,
            self.layer_idx / self.num_layers,  # fraction of the way through layers
            (qtype_size if qtype_size > 0 else 0) / 16.0,
        ], dtype=np.float32)

        return state

    def _generate_episode_summary(self) -> Dict:
        """
        Generate comprehensive summary at the end of an episode.
        Could include final loss, memory usage, bits used per layer, etc.
        """
        final_loss = self.evaluate_loss(self.model)
        memory_saved = self._compute_memory_savings()
        layer_bits = [stat.current_quant_type for stat in self.layer_stats]
        attention_entropies = [stat.attention_entropy for stat in self.layer_stats]

        return {
            'final_loss': final_loss,
            'memory_saved': memory_saved,
            'layer_bits': layer_bits,
            'attention_entropies': attention_entropies,
        }

    def _compute_memory_savings(self) -> float:
        """
        Rough calculation of memory saved vs. the 16-bit or 32-bit baseline.
        For a simplistic example, assume the reference is 16-bit for all layers.
        """
        baseline_bits = 16
        # sum param sizes * chosen_bits and compare
        total_params = 0
        quantized_params = 0
        for i, layer in enumerate(self.layers):
            w = layer.attn.c_attn.weight
            num_params = w.numel()
            chosen_bits = Q_TYPE_SIZE.get(self.layer_stats[i].current_quant_type, baseline_bits)
            quantized_params += num_params * chosen_bits
            total_params += num_params * baseline_bits

        saved = (total_params - quantized_params) / total_params
        return saved
