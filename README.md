Below is a **complete, ready-to-use `README.md`** that is consistent with the details shown in your poster. It summarizes DynaQuant’s motivation, core approach, results, and future directions. You can adapt the command-line examples and file references to match your actual code structure.

---

# DynaQuant: Dynamic Quantization of LLMs via Reinforcement Learning

**Course**: CS234 (Winter 2025) Final Project  
**GitHub Repository**: [rl-dynamic-quant](https://github.com/olegroshka/rl-dynamic-quant)  

## Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Installation & Requirements](#installation--requirements)
4. [Usage & Workflow](#usage--workflow)
5. [Results](#results)
6. [Future Work](#future-work)
7. [References](#references)

---

## Introduction

Large Language Models (LLMs) are highly effective but extremely resource-intensive. **Quantization**—reducing floating-point precision—helps reduce model size, speed up inference, and lower memory usage. However, *uniform* quantization often can be suboptimal because not all layers are equally sensitive to lower precision.

**DynaQuant** introduces a *dynamic*, **per-layer** quantization strategy powered by **Reinforcement Learning (RL)**. Specifically, we train a **PPO** agent to select an optimal quantization type for each layer (e.g., nf4, fp4, int8, bf16, fp16, etc.) during a short fine-tuning process. This agent aims to **balance model accuracy** (measured by perplexity or task accuracy) **with memory savings**.

---

## Key Features

1. **Adaptive Layer-by-Layer Quantization**  
   - The RL agent chooses among multiple bit-widths per layer, rather than applying a uniform quant scheme.

2. **Short Post-Quantization Fine-Tuning**  
   - After each layer is quantized, the model is fine-tuned briefly so that subsequent layers can adapt to earlier choices.

3. **Multi-Objective Reward**  
   - Balances perplexity improvement, memory savings, distribution alignment (KL), and attention entropy preservation.

4. **Scalable to Various Model Sizes**  
   - Implemented and tested on GPT-2 variants; designed to scale to larger LLMs.

5. **Open-Source Implementation**  
   - Uses widely available libraries such as **bitsandbytes** for quantization, plus **PyTorch** for training.

---

## Installation & Requirements

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/olegroshka/rl-dynamic-quant.git
   cd rl-dynamic-quant
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   - Typical dependencies include `torch`, `bitsandbytes`, `numpy`, etc.

---

## Usage & Workflow

Below is a high-level workflow. Adapt the file names/commands to your actual code organization.

1. **Train the PPO Agent**  
   - The agent interacts with a custom RL environment in which **each step** corresponds to quantizing a single layer of the LLM, followed by short fine-tuning to measure updated performance.
   - Example (placeholder) command:
     ```bash
     python train.py \
       --model gpt2 \
       --dataset commonsenseqa \
       --num-episodes 100 \
       --max-layers 12 \
       --quant-types "nf4,fp4,int8,fp16" \
       --reward-weights "6.0,5.0,0.8,4.0"
     ```
   - The script might fine-tune the model after each layer’s quantization, compare it against a reference copy, then compute the reward.

2. **Evaluate Quantized Model**  
   - After training, the learned policy yields a *mixed-precision schedule*. You can apply that schedule to a fresh LLM and do a final fine-tuning pass to confirm performance.
   - Evaluate perplexity, accuracy, and memory usage on test sets like **BoolQ** or **PIQA**:
     ```bash
     python eval.py \
       --model gpt2 \
       --dataset piqa \
       --quant_schema "['int8', 'int8', 'nf4', 'nf4', 'nf4', 'nf4', 'fp4', 'nf4', 'fp4', 'nf4', 'fp4', 'fp4']"
     ```
3. **Compare to Uniform Quantization**  
   - For comparison, you can uniformly quantize all layers (e.g., all int8, all nf4) and measure perplexity/accuracy in the same manner.

---

## Results

Below is a condensed table from our experiments on **GPT-2 Small** evaluated on **PIQA**. The baseline is full-precision (FP32). We also tested a uniform `nf4` approach and our **DynaQuant** RL-based mixed-precision method.

| **Method**          | **PPL**  | **ΔPPL(\%)** | **Acc(\%)** | **ΔAcc(pp)** | **Mem (MB)** | **ΔMem(\%)** |
|---------------------|---------:|-------------:|------------:|-------------:|-------------:|-------------:|
| **Baseline (FP32)** | **25.31** | ---          | **60.72**   | ---          | **574.22**   | ---          |
| **Baseline (FP16)** | 25.31     | +0.01\%      | 60.72       | +0.00        | 422.69       | -26.38\%     |
| **Uniform NF4**     | 22.66     | -10.47\%     | 60.77       | +0.05        | 300.95       | -47.60\%     |
| **DynaQuant** (mixed) | **21.44** | **-15.29\%** | **61.53**   | **+0.82**    | 464.29       | -19.14\%     |

- **Observations**:
  - **DynaQuant** obtains the best overall accuracy and perplexity trade-off while still reducing memory usage relative to FP32.
  - **Uniform NF4** offers substantial memory gains but with higher perplexity than DynaQuant.
  - **FP16** is a simple baseline, maintaining the same perplexity but with moderate memory savings.

---

## Future Work

1. **Scaling to Larger Models**  
   - Test on bigger GPT-2 variants or other popular LLMs (e.g., Qwen, phi-2), ensuring that the RL approach scales efficiently.

2. **Skip Quantization Actions**  
   - Allow the agent to “skip” a layer if it’s too sensitive, improving stability.

3. **Enhance the Policy Network**  
   - Explore a transformer-based policy that uses more context on prior quantization decisions.

4. **Different Reward Weighting Schemes**  
   - Investigate how changes in reward hyperparameters (e.g., putting more emphasis on memory or perplexity) affect final outcomes.

5. **Faster Fine-Tuning Strategies**  
   - Fine-tuning after each layer is powerful but expensive. Optimizing or reducing the cost of these short fine-tuning steps would be beneficial.

---

## References

- **LLM Quantization**: 
  - GPTQ: [Frantar & Alistarh, 2022]  
  - QLoRA: [Dettmers et al., 2023]  
  - bitsandbytes / 8-bit/4-bit: [Dettmers et al., 2022]
- **RL for Architecture/Quant Search**:  
  - NASNet: [Zoph & Le, 2016]  
  - APQ: [Wang et al., 2020]  
- **Mixed-Precision**:  
  - HAWQ: [Dong et al., 2019]

*(References in biblatex style are in the poster’s bibliography.)*

---

**Thank you for checking out DynaQuant!** For more details, visit the [GitHub repository](https://github.com/olegroshka/rl-dynamic-quant) or contact me at [oros@stanford.edu](mailto:oros@stanford.edu). Feedback and contributions are welcome!