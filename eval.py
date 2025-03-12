#!/usr/bin/env python
# eval.py

import argparse
import ast
import logging
import time
import torch

MAX_LENGTH_MAP = {
        "commonsense_qa": 128,
        "boolq": 256,
        "openbookqa": 256,
        "piqa": 128
}

N_LABELS = 100

USE_ENTIRE_SEQUENCE_FOR_PPL = True

AUTOCAST_TYPE = torch.float32

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM
from data_handler import DataHandler
from datasets import load_dataset

from quantizer import MixedQuantizer, Quantizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def evaluate_perplexity(model, data_loader, device="cuda"):
    """
    Compute average perplexity = exp( average cross-entropy ).
    Uses the tokenized version of the dataset from DataHandler
    (with "input_ids" and "labels" etc.)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Some DataHandler code sets 'labels' to only cover the answer portion
            # For perplexity, we can either use that or the entire input_ids as labels
            # If 'labels' is in batch, we use it:
            if USE_ENTIRE_SEQUENCE_FOR_PPL:
                labels = input_ids  # measure perplexity on full sequence
            else:
                labels = batch["labels"] if "labels" in batch else input_ids

            with torch.amp.autocast('cuda', dtype=AUTOCAST_TYPE):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            batch_tokens = (labels != -N_LABELS).sum().item()  # count real tokens
            if batch_tokens == 0:
                # If the entire prompt is masked, fallback to number of tokens in input_ids
                batch_tokens = input_ids.numel()

            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

def evaluate_mc_accuracy(model, tokenizer, raw_val_data, dataset_name, device="cuda"):
    """
    Multiple-choice accuracy for:
      - CommonsenseQA
      - OpenBookQA
      - BoolQ (treated as 2-choice: Yes/No)
      - PIQA (2-choice: sol1, sol2)

    We'll compute a negative cross-entropy (log-prob) for each choice,
    then pick whichever choice has the highest log-prob (lowest XEnt).
    """
    model.eval()
    correct = 0
    total = 0

    with (torch.no_grad()):
        for ex in raw_val_data:
            # 1. Gather the question/prompt and the list of choices
            if dataset_name == "commonsense_qa":
                question = ex["question"]
                # e.g. ex["choices"]["text"] -> ["the sun", "a fish", ...]
                choices_texts = ex["choices"]["text"]
                # e.g. ex["answerKey"] -> "A"
                answer_key = ex["answerKey"]
                gold_index = ord(answer_key) - ord("A")

                # We'll build prompt as: "Question: {q}\nAnswer: {choice}"

            elif dataset_name == "openbookqa":
                question = ex["question_stem"]
                choices_texts = ex["choices"]["text"]
                answer_key = ex["answerKey"]
                gold_index = ord(answer_key) - ord("A")

                # Prompt: "Question: {question}\nAnswer: {choice}"

            elif dataset_name == "boolq":
                # BoolQ has: passage, question, answer (bool)
                passage = ex["passage"]
                question = ex["question"]
                answer_bool = ex["answer"]  # True/False

                # We'll do a 2-choice scenario: 0="No", 1="Yes"
                choices_texts = ["No", "Yes"]
                gold_index = 1 if answer_bool else 0

                # We'll build prompt:
                # "Passage: {passage}\nQuestion: {question}\nAnswer: {choice}"

            elif dataset_name == "piqa":
                # PIQA has: goal, sol1, sol2, label
                goal = ex["goal"]
                sol1 = ex["sol1"]
                sol2 = ex["sol2"]
                label = ex["label"]   # 0 or 1
                gold_index = label    # whichever is correct
                choices_texts = [sol1, sol2]

                # We'll build prompt:
                # "Goal: {goal}\nSolution: {choice}"

            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            # 2. For each choice, compute cross-entropy
            choice_logprobs = []
            for i, choice_text in enumerate(choices_texts):
                # Build the prompt differently per dataset
                if dataset_name == "commonsense_qa":
                    prompt_text = f"Question: {question}\nAnswer: {choice_text}"
                elif dataset_name == "openbookqa":
                    prompt_text = f"Question: {question}\nAnswer: {choice_text}"
                elif dataset_name == "boolq":
                    prompt_text = (
                        f"Passage: {passage}\n"
                        f"Question: {question}\n"
                        f"Answer: {choice_text}"
                    )
                elif dataset_name == "piqa":
                    prompt_text = f"Goal: {goal}\nSolution: {choice_text}"
                else:
                    raise ValueError(f"Unsupported dataset: {dataset_name}")

                # Tokenize
                tokenized = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_LENGTH_MAP[dataset_name],
                    padding="max_length"
                ).to(device)
                input_ids = tokenized["input_ids"]
                attention_mask = tokenized["attention_mask"]

                # Compute average cross-entropy
                with torch.amp.autocast('cuda', dtype=AUTOCAST_TYPE):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)

                # outputs.loss is average CE across all tokens in prompt_text
                # Lower loss => higher probability
                avg_ce = outputs.loss.item()
                choice_logprobs.append(-avg_ce)  # negative => logprob

            # 3. Pick the best choice
            pred_index = max(range(len(choice_logprobs)), key=lambda i: choice_logprobs[i])
            if pred_index == gold_index:
                correct += 1
            total += 1

    if total == 0:
        return 0.0
    return correct / total


def evaluate_mc_accuracy_old(model, tokenizer, raw_val_data, dataset_name, device="cuda"):
    """
    Multiple-choice accuracy for CommonsenseQA or OpenBookQA.
    We re-load the *raw* validation data so we can see the original question,
    choice texts, and the correct answer key (e.g. 'A', 'B', 'C', etc.).

    We'll compute a negative cross-entropy (log-prob) for each choice,
    then pick whichever choice has the highest log-prob.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for ex in raw_val_data:
            # 1. Grab question text
            if dataset_name == "commonsense_qa":
                question = ex["question"]
            elif dataset_name == "openbookqa":
                question = ex["question_stem"]
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            # 2. Grab list of choices and correct answer index
            #    For both datasets, we have ex["choices"]["text"] and ex["answerKey"].
            choices_texts = ex["choices"]["text"]    # e.g. ["the sun", "a fish", ...]
            answer_key = ex["answerKey"]            # e.g. "A", "B", ...
            gold_index = ord(answer_key) - ord("A")  # Convert to 0-based

            # 3. Score each choice by computing cross-entropy on "Question: {Q}\nAnswer: {choice}"
            choice_logprobs = []
            for choice_text in choices_texts:
                prompt_text = f"Question: {question}\nAnswer: {choice_text}"
                tokenized = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_LENGTH_MAP[dataset_name],
                    padding="max_length"
                ).to(device)
                input_ids = tokenized["input_ids"]
                attention_mask = tokenized["attention_mask"]

                with torch.amp.autocast('cuda', dtype=AUTOCAST_TYPE):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)

                # outputs.loss is average cross-entropy across all tokens
                # Lower loss => higher probability
                choice_logprobs.append(-outputs.loss.item())

            # 4. Pick the best choice (max log-prob is min loss)
            pred_index = max(range(len(choice_logprobs)), key=lambda i: choice_logprobs[i])

            if pred_index == gold_index:
                correct += 1
            total += 1

    if total == 0:
        return 0.0
    return correct / total

def measure_inference_throughput(model, tokenizer, device="cuda", seq_len=32, n_trials=10, batch_size=8):
    """
    Quick test: measure forward-pass tokens/s and peak memory usage.
    Adjust the seq_len, n_trials, and batch_size as needed.
    """
    model.eval()

    # Create random input to measure forward-pass speed
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Clear old stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_trials):
            with torch.amp.autocast('cuda', dtype=AUTOCAST_TYPE):
                _ = model(input_ids, attention_mask=attention_mask)
    end_time = time.time()

    elapsed = end_time - start_time
    total_tokens = batch_size * seq_len * n_trials
    tokens_per_sec = total_tokens / elapsed

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # in MB
    return tokens_per_sec, peak_mem

def main():
    parser = argparse.ArgumentParser(description="Evaluation script for GPT2-based models.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory containing the fine-tuned model (e.g., results/dynamic_qlora_exp).")
    parser.add_argument("--quant_schema", type=str, required=True, help="List of quantization types, e.g. ['fp16','nf4','int8',...] or 4 or 8 for uniform quantization.")
    parser.add_argument("--dataset", type=str, default="boolq", help="Which dataset to evaluate on: 'boolq' or 'boolq'")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for perplexity evaluation DataLoader.")
    parser.add_argument("--use_lm_chunking", default=True, action="store_true", help="If set, chunk the dataset for LM-style perplexity.")
    parser.add_argument("--block_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quant_schema = ast.literal_eval(args.quant_schema)

    if( type(quant_schema) == int ):
        bit_width = quant_schema
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            load_in_4bit=bit_width == 4,  # This triggers 4-bit weights
            load_in_8bit=bit_width == 8,  # This triggers 8-bit weights
            device_map="auto",  # Places modules automatically
        )
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model_dir)
        #model.load_state_dict(torch.load(args.model_dir))
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

    model.to(device)

    # 1. Load the fine-tuned model + tokenizer from disk
    #model = GPT2LMHeadModel.from_pretrained(args.model_dir).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Create a DataHandler to get the tokenized val_loader for perplexity
    data_handler = DataHandler(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        max_length=128,
        use_lm_chunking=args.use_lm_chunking,
        block_size=args.block_size
    )
    _, val_loader = data_handler.load_dataset()

    # 3. Evaluate perplexity on the tokenized data
    ppl = evaluate_perplexity(model, val_loader, device)
    print(f"[{args.dataset}] Validation Perplexity: {ppl:.8f}")

    # 4. Multiple-choice accuracy: need original raw validation data
    raw_dataset = load_dataset(args.dataset)
    raw_val_data = raw_dataset["validation"]
    mc_accuracy = evaluate_mc_accuracy(model, tokenizer, raw_val_data, args.dataset, device)
    print(f"[{args.dataset}] Multiple-choice Accuracy: {mc_accuracy * 100:.8f}%")

    # 5. (Optional) Measure throughput / memory usage
    tps, mem_mb = measure_inference_throughput(model, tokenizer, device, seq_len=32, n_trials=10, batch_size=args.batch_size)
    print(f"Inference speed: {tps:.8f} tokens/s at batch_size={args.batch_size}, seq_len=32")
    print(f"Peak GPU memory used: {mem_mb:.8f} MB")

if __name__ == "__main__":
    main()

# Example usage:
#  python eval.py --model_dir results/gpt-2-baseline --quant_schema "['int8', 'int8', 'nf4', 'nf4', 'nf4', 'nf4', 'fp4', 'nf4', 'fp4', 'nf4', 'fp4', 'fp4']"