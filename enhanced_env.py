import copy
import logging
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import bitsandbytes as bnb
import numpy as np
import torch
import torch.nn as nn
from bitsandbytes.functional import dequantize_4bit
from torch.utils.data import DataLoader
from transformers import Conv1D

from quantizer import Quantizer, Q_TYPE_SIZE

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class LayerStats:
    """Track per-layer statistics for better state representation"""
    weight_mean: float
    weight_std: float
    gradient_norm: float
    attention_entropy: float  # measure of pattern diversity
    layer_idx: int
    current_quant_type: str


def perplexity_sigmoid_reward(ref_ppl: float, quant_ppl: float, alpha: float = 5.0) -> float:
    """
    Returns a smooth, bounded reward in (0, 1) using a sigmoid function.

    Args:
        ref_ppl: Perplexity of the reference model
        quant_ppl: Perplexity of the quantized model
        alpha: scaling factor that controls how sharply the reward saturates.
               Larger alpha => more "binary" (quickly saturates near 0 or 1).

    Returns:
        A scalar in (0, 1), where values > 0.5 mean quant is better than reference,
        and values < 0.5 mean quant is worse, with a smooth transition.
    """
    # 1) Delta perplexity: positive if quant is better (lower perplexity)
    delta_ppl = (ref_ppl - quant_ppl)

    # 2) Sigmoid transform: 0.5 => no difference, > 0.5 => quant is better
    #    If alpha * delta_ppl = 0 => reward=0.5
    #    If alpha * delta_ppl >> 0 => reward approaches 1.0
    #    If alpha * delta_ppl << 0 => reward approaches 0.0
    #    We use a small Tensor conversion because sigmoid is a PyTorch op.
    delta_tensor = torch.tensor(alpha * delta_ppl, dtype=torch.float32)
    reward = torch.sigmoid(delta_tensor).item()  # to Python float

    return reward


def perplexity(val_loss):
    return float(torch.exp(torch.tensor(val_loss))) if val_loss < 20 else float('inf')


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
        reward_weights: Optional[Dict] = None,
        finetune_steps: int = 5,
        lr: float = 5e-5,
        ema_alpha: float = 0.9  # EMA alpha hyperparam
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
        self.ref_model_for_episode = None
        self.train_loader = train_loader
        self.finetune_iter = iter(self.train_loader)
        self.quantizer = quantizer
        self.val_loader = val_loader
        self.finetune_steps = finetune_steps
        self.lr = lr

        # ### EMA ###
        # We'll store separate EMAs for each reward component so you can inspect them if desired.
        self.ema_alpha = ema_alpha
        self.ema_initialized = False  # Will become True after the first reward is computed

        self.smoothed_perf_reward = 0.0
        self.smoothed_kl_reward = 0.0
        self.smoothed_entropy_reward = 0.0
        self.smoothed_memory_reward = 0.0
        self.smoothed_total_reward = 0.0

        # Track the current/previous losses and perplexities to build better states
        self.current_loss = 0.0
        self.current_quant_ppl = 0.0
        self.last_quant_loss = None  # For measuring the change vs. previous layer
        self.last_quant_ppl = None
        self.loss_diff = 0.0
        self.quant_ppl_diff = 0.0

        # GPT-2 blocks: a list of transformer layers
        self.layers = list(self.model.transformer.h)
        self.num_layers = len(self.layers)
        self.layer_idx = 0  # which layer we're working on now

        # Precompute total parameter count across the entire model
        self.total_params = sum(p.numel() for p in self.model.parameters())

        # For each layer, compute how many parameters it has.
        # Then compute ratio = layer_params / total_params
        self.layer_param_counts = [
            sum(p.numel() for p in layer.parameters()) for layer in self.layers
        ]
        self.layer_param_ratios = [
            layer_count / self.total_params for layer_count in self.layer_param_counts
        ]

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
            val_perplexity = perplexity(val_loss)
            memory_usage = sum(p.numel() * p.element_size() for p in self.reference_model.parameters())

        return {
            'loss': val_loss,
            'perplexity': val_perplexity,
            'memory': memory_usage,
            # pre-compute layer stats on reference model
            'layer_stats': self._get_layer_statistics(self.reference_model)
        }

    def _get_layer_statistics(self, model: nn.Module) -> List[LayerStats]:
        """
        Computes per-layer statistics, including:
          - weight mean/std
          - gradient norm
          - attention entropy
        by performing a forward pass for attention entropies, then iterating
        over each layer's c_attn weights. If a layer is in 4-bit form (bnb.nn.Params4bit),
        we first dequantize to get real float values for mean/std.
        """
        # 1) Do one forward pass with output_attentions=True to compute attention entropies
        entropies = self._compute_attention_entropy_all_layers(model)

        stats = []
        for i, layer in enumerate(model.transformer.h):
            with torch.no_grad():
                w = layer.attn.c_attn.weight

                # Decide whether to dequantize or use regular .mean()/.std()
                if isinstance(w, bnb.nn.Params4bit) and w.quant_state is not None:
                    # Dequantize from 4-bit to real float
                    w_dequant = dequantize_4bit(w.data, w.quant_state)
                    w_mean = w_dequant.mean().item()
                    w_std = w_dequant.std().item()
                else:
                    # Normal float16/bfloat16/float32 or int8
                    # At least cast to float to avoid "Byte" or other issues
                    w_mean = w.float().mean().item()
                    w_std = w.float().std().item()

                # For gradient norm, we assume w.grad is floating. If no grad, set 0
                grad_norm = w.grad.norm().item() if (w.grad is not None) else 0.0

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
        Start new episode:
          - Copy reference model => self.ref_model_for_episode
          - Copy reference model => self.model
          - We'll do 1 layer step at a time, up to num_layers
        """
        self.ref_model_for_episode = copy.deepcopy(self.reference_model).to(self.device)
        self.model = copy.deepcopy(self.reference_model).to(self.device)

        # Evaluate the cross-entropy of this fresh copy
        #self.prev_loss = self.evaluate_loss(self.model)

        # Refresh the 'self.layers' to point at the new model's blocks
        self.layers = list(self.model.transformer.h)
        self.num_layers = len(self.layers)
        self.layer_idx = 0

        # Reset stats and loss tracking
        self.layer_stats = [LayerStats(0, 0, 0, 0, i, "fp32") for i in range(self.num_layers)]
        self.current_loss = 0.0
        self.current_quant_ppl = 0.0
        self.last_quant_loss = None
        self.loss_diff = 0.0
        self.quant_ppl_diff = 0.0

        return self._build_state_representation()


    def _get_finetune_batches(self, num_steps: int):
        """Get exactly `num_steps` mini-batches from train_loader,
        rewinding to the start if we run out."""
        batches = []
        for _ in range(num_steps):
            try:
                batch = next(self.finetune_iter)
            except StopIteration:
                # If we exhaust the loader, start over
                self.finetune_iter = iter(self.train_loader)
                batch = next(self.finetune_iter)
            batches.append(batch)
        return batches


    def step(self, action_quant_type: str) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step function that:
          1. Fetch the same data for both reference and quantized model
          2  Fine-tune the episode reference model
          3. Applies quantization to the current layer
          4. Fine-tunes the quantized model for self.finetune_steps
          5. Evaluates new loss and computes reward
          6. Updates layer stats
          7. Builds next state
          8. Moves to next layer or ends episode
        """
        # 1 Fetch the same data for both reference and quantized model
        finetune_batches = self._get_finetune_batches(self.finetune_steps)

        # 2 Fine-tune the episode reference model for a few steps
        self._adaptive_fine_tune(self.ref_model_for_episode, finetune_batches)

        # 3 Quantize the current layer
        current_layer = self.layers[self.layer_idx]
        self._quantize_layer(current_layer, action_quant_type)

        # 4 Fine-tune quantized model for the same few steps
        self._adaptive_fine_tune(self.model, finetune_batches)

        # 5 Evaluate new loss
        ref_loss = self.evaluate_loss(self.ref_model_for_episode)
        quant_loss = self.evaluate_loss(self.model)

        # Update stored environment-level loss stats
        self._update_env_loss_stats(quant_loss, ref_loss)

        # Compute reward
        reward, reward_info = self.compute_reward(
            ref_loss,
            quant_loss,
            action_quant_type,
            self.layer_idx
        )

        # 6 Update layer stats for the layer we just quantized
        updated_layer_stats = self._get_layer_statistics(self.model)[self.layer_idx]
        self.layer_stats[self.layer_idx] = updated_layer_stats
        # Set the current_bits to the chosen action
        self.layer_stats[self.layer_idx].current_quant_type = action_quant_type

        # 7 Build next state
        next_state = self._build_state_representation()

        # 8 Move to next layer
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

    def _update_env_loss_stats(self, quant_loss, ref_loss):
        ref_ppl = perplexity(ref_loss)
        quant_ppl = perplexity(quant_loss)
        if self.last_quant_loss is None:
            self.loss_diff = 0.0
        else:
            self.loss_diff = quant_loss - self.last_quant_loss

        if self.last_quant_ppl is None:
            self.quant_ppl_diff = 0.0
        else:
            self.quant_ppl_diff = quant_ppl - self.last_quant_ppl

        self.last_quant_loss = quant_loss
        self.last_quant_ppl = quant_ppl
        self.current_loss = quant_loss
        self.loss_diff = ref_loss - quant_loss
        self.current_quant_ppl = quant_ppl
        self.quant_ppl_diff = ref_ppl - quant_ppl

    def debug_fp16_all_params(self, model: nn.Module):
        """
        Prints any parameters (including unnamed) that are physically stored in float16
        and have requires_grad=True.
        """
        # 1) Collect named parameters into a dict {param_obj: name}
        named_map = {}
        for name, param in model.named_parameters():
            named_map[param] = name

        # 2) Iterate over *all* parameters from model.parameters()
        #    Some might not appear in model.named_parameters() if not registered properly.
        all_params = list(model.parameters())

        found_any = False
        for i, param in enumerate(all_params):
            # See if param has a known name
            param_name = named_map.get(param, f"<unnamed_param_{i}>")

            # Check for float16 + requires_grad
            if param.dtype == torch.float16 and param.requires_grad:
                logger.info(f"WARNING: {param_name} is float16 + requires_grad=True!")
                found_any = True

        if not found_any:
            logger.info("No leftover float16 trainable params found among all parameters.")


    def _quantize_layer(self, layer: nn.Module, quant_type: str):
        for name, child in layer.named_children():
            # skip if it is already quantized
            if isinstance(child, (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)):
                continue
            if isinstance(child, (nn.Linear, Conv1D)):
                #logger.info(f"Quantizing layer {layer}, name {name}, with type {quant_type}")
                new_mod = self.quantizer.quantize_linear_layer(child, quant_type)
                setattr(layer, name, new_mod)
            else:
                self._quantize_layer(child, quant_type)


    def _adaptive_fine_tune(self, model: nn.Module, batch_list: List[Dict[str, torch.Tensor]]):
        # 1) Make sure all non-bitsandbytes parameters are physically float32
        #    and bitsandbytes params are requires_grad=False
        self.freeze_quantised_params(model)

        #self.debug_fp16_all_params(self.model)

        # 2) Build optimizer only over trainable (float32) params
        optim_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(optim_params, lr=self.lr)

        scaler = torch.amp.GradScaler('cuda')
        self.model.train()

        for batch in batch_list:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = input_ids.clone()

            optimizer.zero_grad()

            # Mixed precision forward/backward
            with torch.amp.autocast('cuda', dtype=torch.float32):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


    def freeze_quantised_params(self, model: nn.Module):
        for param in model.parameters():
            if isinstance(param, (bnb.nn.Params4bit, bnb.nn.Int8Params)):
                param.requires_grad = False
            else:
                # force float32 if you suspect some param is still half
                param.data = param.data.float()
                param.requires_grad = True


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

    def compute_reward(self,
                       ref_loss: float,
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
        # Loss based
        # 1) Compare new quant model's loss to the reference model's loss
        #    "baseline_loss" was computed once at environment init.
        # ------------------------------------------------------------------------
        #ref_loss = self.baseline_metrics["loss"]
        # perf_diff = ref_loss - quant_loss
        # perf_reward = perf_diff * self.reward_weights.get('performance', 1.0)

        # 1) Convert losses to perplexities
        ref_ppl = perplexity(ref_loss) #math.exp(ref_loss) if ref_loss < 20 else float('inf')
        quant_ppl = perplexity(quant_loss) #math.exp(quant_loss) if quant_loss < 20 else float('inf')

        # 2) Performance reward using perplexity
        # Option A: difference in perplexities
        #   (the sign is negative if quant ppl is higher => negative reward)
        # perf_diff_ppl = ref_ppl - quant_ppl
        # perf_reward = perf_diff_ppl * self.reward_weights.get('performance', 1.0)

        # TODO: try Option B: ratio-based term
        #   ratio < 1 => quant is better => positive reward
        #   ratio > 1 => quant is worse => negative reward
        # ratio = quant_ppl / (ref_ppl + 1e-8)
        # perf_reward = (1.0 - ratio) * self.reward_weights.get('performance', 1.0)

        # Option C: Sigmoid-based reward
        perf_reward_raw = perplexity_sigmoid_reward(ref_ppl, quant_ppl, alpha=1.0)
        perf_reward = perf_reward_raw * self.reward_weights.get('performance', 1.0)

        # ------------------------------------------------------------------------
        # 2) KL Divergence from reference model
        #    We'll do a forward pass on some batch to get p (quant) and q (ref)
        #    If you haven't implemented a method for it, see _compute_kl_div()
        # ------------------------------------------------------------------------
        kl_value = self._compute_kl_divergence(
            quant_model=self.model,
            ref_model=self.ref_model_for_episode
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

        # 4) Memory reward with layer weighting
        #    memory_factor is how many bits we save relative to a 16-bit baseline
        #    then we multiply by this layer's fraction of total parameters
        memory_factor = (16 - Q_TYPE_SIZE[quant_type]) / 16.0
        layer_ratio = self.layer_param_ratios[layer_idx]  # precomputed fraction of total params in this layer
        memory_reward = memory_factor * layer_ratio * self.reward_weights.get('memory', 1.0)

        # ------------------------------------------------------------------------
        # Combine all components
        # ------------------------------------------------------------------------
        total_reward = perf_reward + kl_reward + entropy_reward + memory_reward

        # (2) ### EMA LOGIC ###
        if not self.ema_initialized:
            # Initialize the EMA buffers on the first call
            self.smoothed_perf_reward = perf_reward
            self.smoothed_kl_reward = kl_reward
            self.smoothed_entropy_reward = entropy_reward
            self.smoothed_memory_reward = memory_reward
            self.smoothed_total_reward = total_reward
            self.ema_initialized = True
        else:
            alpha = self.ema_alpha

            self.smoothed_perf_reward = alpha * self.smoothed_perf_reward + (1 - alpha) * perf_reward
            self.smoothed_kl_reward = alpha * self.smoothed_kl_reward + (1 - alpha) * kl_reward
            self.smoothed_entropy_reward = alpha * self.smoothed_entropy_reward + (1 - alpha) * entropy_reward
            self.smoothed_memory_reward = alpha * self.smoothed_memory_reward + (1 - alpha) * memory_reward
            self.smoothed_total_reward = alpha * self.smoothed_total_reward + (1 - alpha) * total_reward

        # (3) Use the smoothed total reward for the RL update
        final_reward = self.smoothed_total_reward

        reward_info = {
            'perf_reward': perf_reward,
            'kl_reward': kl_reward,
            'entropy_reward': entropy_reward,
            'memory_reward': memory_reward,
            'total_reward': total_reward,
            # EMA versions
            'perf_reward_ema': self.smoothed_perf_reward,
            'kl_reward_ema': self.smoothed_kl_reward,
            'entropy_reward_ema': self.smoothed_entropy_reward,
            'memory_reward_ema': self.smoothed_memory_reward,
            'total_reward_ema': self.smoothed_total_reward
        }

        logger.info(f"Layer {layer_idx}: Reward {reward_info}")

        return final_reward, reward_info # we use EMAs for the RL update here
        #return total_reward, reward_info

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

        # Example state-size-D feature vector:
        #   [weight_mean, weight_std, gradient_norm, attention_entropy,
        #    normalized_layer_idx, normalized_current_bits...]
        qtype_size = Q_TYPE_SIZE.get(current_stats.current_quant_type, 32)  # fallback to 32 if unknown

        # Quant type of previous layer
        if self.layer_idx > 0:
            prev_qtype = self.layer_stats[self.layer_idx - 1].current_quant_type
            prev_qtype_size = Q_TYPE_SIZE.get(prev_qtype, 32)
        else:
            # If no previous layer, default to 16 (fp16) or something neutral
            prev_qtype_size = 16

        state = np.array([
            current_stats.weight_mean,
            current_stats.weight_std,
            current_stats.gradient_norm,
            current_stats.attention_entropy,
            self.layer_idx / self.num_layers,  # fraction of the way through layers
            (qtype_size if qtype_size > 0 else 0) / 16.0,
            self.current_quant_ppl,  # perplexity of current partial model
            self.quant_ppl_diff,  # change in perplexity from previous step
            self.loss_diff,  # change in loss from previous step
            prev_qtype_size / 16.0,  # previous layer's quant type
            # EMA versions of rewards
            self.smoothed_total_reward,
            self.smoothed_perf_reward,
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
