
# RL-Based Dynamic Quantization

**Table of Contents**  
1. [Overview](#overview)  
2. [Key Ideas](#key-ideas)  
3. [Repository Structure](#repository-structure)  
4. [Installation & Requirements](#installation--requirements)  
5. [Usage](#usage)  
6. [Project Workflow](#project-workflow)  
7. [Experiments & Results](#experiments--results)  
8. [Future Work](#future-work)  
9. [License](#license)

## Overview

This repository demonstrates an **RL-based (Reinforcement Learning) approach** to quantizing a GPT-2 model. Instead of using a uniform bit-width (e.g., 8-bit or 4-bit) across all layers, we dynamically choose different bit-widths per layer (or per group of layers) to balance **model accuracy** and **memory footprint**.

We leverage:
- **PyTorch** for neural network operations,  
- **Hugging Face Transformers** for loading and fine-tuning GPT-2,  
- A **custom RL environment** that applies layer-by-layer quantization during fine-tuning,  
- A **PPO-based** RL loop (Proximal Policy Optimization) to find the optimal quantization policy.

By the end, we obtain a strategy for **minimizing validation loss** (or perplexity) while penalizing high-bit usage.

## Key Ideas

1. **Dynamic Quantization**  
   - The algorithm can pick different bit-widths (2, 4, 8, 16) for each layer.  
   - This deviates from typical static or uniform quantization, potentially reducing memory usage without sacrificing too much accuracy.

2. **Reinforcement Learning**  
   - We treat layer-wise bit-width selection as an RL action.  
   - A reward function penalizes an increase in validation loss and also the size of chosen bits.

3. **Fine-Tuning**  
   - After each quantization decision, we run a few fine-tuning steps on a domain-specific dataset (e.g., CommonsenseQA).  
   - This partial fine-tuning helps the model adapt to quantization-induced distortions.

4. **Comparison**  
   - We compare our RL-based quantized model to baselines like **bitsandbytes 8-bit** or standard PyTorch dynamic quantization (where applicable).  
   - We measure perplexity or accuracy on validation sets (e.g., CommonsenseQA, OpenBookQA).


## Repository Structure

.
├── environment.py         # Defines the custom RL environment (QuantizationEnv)
├── policy.py              # A simple feed-forward policy network (PolicyNet)
├── ppo_trainer.py         # Implements a simplified PPO training loop
├── quant_utils.py         # Utility functions for naive min-max quantization
├── main.py                # Entry point for data loading, environment setup, PPO training
├── requirements.txt       # Python dependencies
└── README.md              # This file

## Installation & Requirements

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/rl-dynamic-quant.git
   cd rl-dynamic-quant
   ```

2. **Create and Activate an Environment**

   ```bash
   conda create -n rlq_env python=3.9
   conda activate rlq_env
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Key packages:
   - `torch` (preferably with CUDA support)
   - `transformers`
   - `datasets`
   - `numpy`
   - (Optional) `bitsandbytes` if comparing with 8-bit or 4-bit baselines.

4. **GPU Environment (Recommended)**  
   Make sure you have a supported GPU (e.g., NVIDIA 4070) and CUDA toolkit installed for efficient training.

---

## Usage

1. **Train the RL Quantization Policy**

   Open `main.py` and modify any hyperparameters or dataset references if needed. Then run:

   ```bash
   python main.py
   ```

   This script:
   - Loads GPT-2 and tokenizer,  
   - Loads and tokenizes the dataset(s),  
   - Creates the `QuantizationEnv`,  
   - Initializes the policy and runs PPO training,  
   - Prints reward, layer choices, and final losses.

2. **Monitor Outputs**

   Check console logs for each episode, including total reward, final policy loss, and possibly the chosen bit-width for each layer.  

3. **Evaluate**

   After training, the script can compute perplexities on your chosen validation sets. Logs or prints will show how the RL-quantized model performs versus baselines.

---

## Project Workflow

1. **Data Loading**  
   - We use Hugging Face `datasets` for reasoning tasks like CommonsenseQA or OpenBookQA.  

2. **Model & Environment**  
   - Load a pretrained GPT-2 in FP32.  
   - Wrap it with our `QuantizationEnv`, which provides `env.step(action_bits)` to quantize layers on demand.

3. **RL Loop (PPO)**  
   - The policy chooses a bit-width (2, 4, 8, or 16) for each layer.  
   - After quantization, we do a mini fine-tuning step, measure validation loss, and compute the reward.  
   - PPO updates the policy for better future decisions.

4. **Comparison**  
   - We may compare to standard PyTorch dynamic quantization or bitsandbytes-based 4-bit/8-bit baselines.  
   - We measure perplexity (or accuracy) on validation sets.

5. **Results**  
   - We log final perplexities or scores, showing whether the RL-based approach improved the accuracy–size tradeoff.

---

## Experiments & Results


| Model/Method               | CommonsenseQA Val PPL | OpenBookQA Val PPL |
|----------------------------|-----------------------|--------------------|
| **FP32 GPT-2 (Baseline)**  | 2382.924              | 3326.866           |
| **8-bit Bitsandbytes**     | 2379.504              | 3330.174           |
| **RL Fine-Quantized**      | 244.747               | 404.201            |

- The RL-based quantization strategy shows a significant reduction in perplexity compared to the 8-bit baseline. 
- But this evaluation is very preliminary; also note we evaluate on the same dataset used for fine-tuning.  
 
---

## Future Work

- **QLoRA Integration**: Merge the RL bit-width selection with a QLoRA approach to keep the base model 4-bit while learning LoRA adapters.  
- **Better State Space**: Incorporate activation statistics into the environment state.  
- **Complex Reward Functions**: Combine perplexity, memory usage, and downstream QA accuracy into a more comprehensive reward.  
- **Scaling to Larger Models**: Try GPT-2-medium or GPT-Neo families with model parallel or more advanced memory-saving strategies.

---

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it, but please give credit to the authors.

---

**Questions or Feedback?**  
Open an issue on this repository or contact `Oleg Roshka` at `oleg.roshka@proton.me`.
```
