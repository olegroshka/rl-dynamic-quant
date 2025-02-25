import torch
from torch.utils.data import DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorWithPadding
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from environment import QuantizationEnv
from policy import PolicyNet
from ppo_trainer_old import ppo_train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################
# 1) TOKENIZE FUNCTIONS & DATA PREP
###############################################

def tokenize_commonsenseqa_function(examples, tokenizer, max_length=128):
    """
    Tokenizer for CommonsenseQA with batched=True.
    Each example has:
        examples["question"] (string),
        examples["choices"] is a dict like {"label": [...], "text": [...]}
    We combine question + all possible choices into a single string.
    """
    text_list = []
    for question, choices_dict in zip(examples["question"], examples["choices"]):
        merged_choice_text = " ".join(choices_dict["text"])
        combined_str = question + " " + merged_choice_text
        text_list.append(combined_str)

    return tokenizer(
        text_list,
        truncation=True,
        max_length=max_length,
        padding=False  # We'll rely on DataCollatorWithPadding
    )

def tokenize_openbookqa_function(examples, tokenizer, max_length=128):
    """
    Tokenizer for OpenBookQA (main) with batched=True.
    The dataset has fields like:
      - "question_stem",
      - "choices" (dict with "text", "label"),
      - "fact1",
      - "answerKey"
    We'll combine question_stem + choices["text"] into one string.
    """
    text_list = []
    for question_stem, choices_dict in zip(examples["question_stem"], examples["choices"]):
        # choices_dict is typically {"text": [...], "label": [...]}
        merged_choice_text = " ".join(choices_dict["text"])
        combined_str = question_stem + " " + merged_choice_text
        text_list.append(combined_str)

    return tokenizer(
        text_list,
        truncation=True,
        max_length=max_length,
        padding=False
    )

def evaluate_perplexity(model, data_loader, dev="cpu"):
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

# Standard PyTorch dynamic quantization for a baseline
def quantize_model_pytorch_dynamic(model_fp32):
    import torch.quantization as tq
    cpu_model = model_fp32.to(device)
    cpu_model.eval()
    # Typically GPT2 uses Conv1D, which isn't always directly handled by quantize_dynamic.
    # We'll do a best-effort approach on nn.Linear:
    quantized_model = tq.quantize_dynamic(
        cpu_model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model


###############################################
# 2) MAIN
###############################################

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Initialize tokenizer & model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model_fp32 = GPT2LMHeadModel.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model_fp32.resize_token_embeddings(len(tokenizer))

    # 2. Load two datasets:
    #    a) CommonsenseQA (train & validation)
    #    b) OpenBookQA (train & validation) for perplexity checks
    cqa_dataset = load_dataset("commonsense_qa")
    obqa_dataset = load_dataset("openbookqa", "main")  # 'main' config typically

    cqa_train_data = cqa_dataset["train"]
    cqa_val_data   = cqa_dataset["validation"]

    obqa_train_data = obqa_dataset["train"]
    obqa_val_data   = obqa_dataset["validation"]

    # 3. Tokenize each dataset
    def tokenize_cqa(examples):
        return tokenize_commonsenseqa_function(examples, tokenizer, max_length=128)

    def tokenize_obqa(examples):
        return tokenize_openbookqa_function(examples, tokenizer, max_length=128)

    # CommonsenseQA
    cqa_train_dataset = cqa_train_data.map(tokenize_cqa, batched=True)
    cqa_val_dataset   = cqa_val_data.map(tokenize_cqa,   batched=True)

    # OpenBookQA
    obqa_train_dataset = obqa_train_data.map(tokenize_obqa, batched=True)
    obqa_val_dataset   = obqa_val_data.map(tokenize_obqa,   batched=True)

    # 4. Keep only input_ids, attention_mask
    keep_cols = ["input_ids", "attention_mask"]

    def filter_cols(ds):
        return ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

    cqa_train_dataset = filter_cols(cqa_train_dataset)
    cqa_val_dataset   = filter_cols(cqa_val_dataset)
    obqa_train_dataset = filter_cols(obqa_train_dataset)
    obqa_val_dataset   = filter_cols(obqa_val_dataset)

    # 5. DataCollators
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # 6. DataLoaders
    cqa_train_loader = DataLoader(cqa_train_dataset, batch_size=4, shuffle=True,  collate_fn=data_collator)
    cqa_val_loader   = DataLoader(cqa_val_dataset,   batch_size=4, shuffle=False, collate_fn=data_collator)

    obqa_train_loader = DataLoader(obqa_train_dataset, batch_size=4, shuffle=False, collate_fn=data_collator)
    obqa_val_loader   = DataLoader(obqa_val_dataset,   batch_size=4, shuffle=False, collate_fn=data_collator)

    # 7. Reference model (FP32 baseline)
    reference_model = GPT2LMHeadModel.from_pretrained("gpt2")
    reference_model.resize_token_embeddings(len(tokenizer))
    reference_model.load_state_dict(model_fp32.state_dict())

    # 8. Build environment for RL-based quantization
    env = QuantizationEnv(
        model=model_fp32,
        train_loader=cqa_train_loader,
        val_loader=cqa_val_loader,
        reference_model=reference_model,
        dev=device,
        model_name="gpt2"
    )
    policy = PolicyNet(state_dim=5, hidden_dim=32, num_actions=4).to(device)

    # 9. Train with PPO
    ppo_train(env, policy, num_episodes=10, gamma=0.99, epsilon=0.2, lr=1e-3, bit_options=[4, 8, 8, 16], ppo_updates_per_iteration=3, dev=device)

    # Now, env.model (== model_fp32) has gone through RL-based layer-by-layer quantization.

    # 10. Compare perplexities on both CommonsenseQA & OpenBookQA

    # (A) Baseline FP32
    reference_model.to(device)
    p_ref_cqa  = evaluate_perplexity(reference_model, cqa_val_loader, dev=device)
    p_ref_obqa = evaluate_perplexity(reference_model, obqa_val_loader, dev=device)

    # (B) Standard PyTorch dynamic quantization from reference_model
    #quantized_base = quantize_model_pytorch_dynamic(reference_model)
    # If you want to evaluate on GPU, you might move it, though dynamic quant is often CPU-based:
    # quantized_base.to(device)

    #p_stdq_cqa  = evaluate_perplexity(quantized_base, cqa_val_loader, dev="cpu")
    #p_stdq_obqa = evaluate_perplexity(quantized_base, obqa_val_loader, dev="cpu")
    # Bitsandbytes 8-bit
    model_8bit = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        load_in_8bit=True,  # This triggers 8-bit weights
        device_map="auto",  # Places modules automatically
    )
    model_8bit.resize_token_embeddings(len(tokenizer))
    p_bitsandbytes_cqa = evaluate_perplexity(model_8bit, cqa_val_loader, dev=device)
    p_bitsandbytes_obqa = evaluate_perplexity(model_8bit, obqa_val_loader, dev=device)

    # (C) RL Fine-Quantized model
    env.model.to(device)
    p_rlq_cqa  = evaluate_perplexity(env.model, cqa_val_loader, dev=device)
    p_rlq_obqa = evaluate_perplexity(env.model, obqa_val_loader, dev=device)

    # Print results
    print("\n=== Perplexity Comparison ===")
    print("CommonsenseQA Val Set:")
    print(f"  * Baseline FP32:        {p_ref_cqa:.3f}")
    print(f"  * Bitsandbytes 8bit:    {p_bitsandbytes_cqa:.3f}")
    print(f"  * RL Fine-Quantized:    {p_rlq_cqa:.3f}")

    print("\nOpenBookQA Val Set:")
    print(f"  * Baseline FP32:        {p_ref_obqa:.3f}")
    print(f"  * Bitsandbytes 8bit:    {p_bitsandbytes_obqa:.3f}")
    print(f"  * RL Fine-Quantized:    {p_rlq_obqa:.3f}")


if __name__ == "__main__":
    main()
