#!/usr/bin/env python
"""
batch_eval.py

Runs eval.py for multiple datasets and quantization schemas in a systematic way,
with repeated runs for statistical significance.

Example usage:
  python batch_eval.py
"""

import subprocess


def main():
    # 1) the list of datasets
    datasets = [
        #"commonsense_qa",
        "boolq",
        #"openbookqa",
        "piqa"
    ]

    # 2) the list of quantization schemas
    quant_schemas = [
        "['fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16']",
        "['int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8']",
        "['nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4', 'nf4']",
        "['fp16', 'int8', 'fp16', 'nf4', 'fp16', 'int8', 'fp16', 'int8', 'fp4', 'nf4', 'int8', 'fp4']",
        "['fp4', 'int8', 'fp4', 'nf4', 'fp16', 'fp4', 'fp4', 'fp4', 'nf4', 'nf4', 'fp16', 'nf4']",
    ]

    model_dir = "results/gpt-2-baseline"

    # 3) number of repeated runs per combination
    num_runs = 3


    for dataset in datasets:
        for schema in quant_schemas:
            for run_idx in range(num_runs):
                # Build the command to run
                cmd = [
                    "python", "eval.py",
                    "--model_dir", model_dir,
                    "--dataset", dataset,
                    "--quant_schema", schema,
                    "--batch_size", "8",
                    "--use_lm_chunking",  # or remove if you don't want chunking
                    "--block_size", "16",  # or change to 32, etc. if desired
                ]

                print("============================================================")
                print(f"[Run {run_idx + 1}/{num_runs}] eval for dataset={dataset}, schema={schema}")
                print("============================================================")
                # Invoke the command
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
