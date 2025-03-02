from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from transformers import GPT2Tokenizer, default_data_collator
from torch.utils.data import DataLoader


class DataHandler:
    def __init__(self, dataset_name="commonsense_qa", batch_size=8, max_length=128):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # Ensure pad_token is set for GPT2
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_dataset(self):
        try:
            dataset = load_dataset(self.dataset_name)
        except DatasetNotFoundError:
            raise ValueError(f"Unsupported dataset or dataset not found: {self.dataset_name}")

        # Expect train & validation splits
        if "train" not in dataset or "validation" not in dataset:
            raise ValueError(f"Dataset {self.dataset_name} is missing train or validation splits.")

        train_data = dataset["train"]
        val_data = dataset["validation"]

        # Choose appropriate tokenize function
        if self.dataset_name == "commonsense_qa":
            tokenize_fn = self._tokenize_commonsenseqa
        elif self.dataset_name == "openbookqa":
            tokenize_fn = self._tokenize_openbookqa
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        # Map the tokenize function
        train_dataset = train_data.map(
            tokenize_fn,
            remove_columns=train_data.column_names,
            batched=True,
        )
        val_dataset = val_data.map(
            tokenize_fn,
            remove_columns=val_data.column_names,
            batched=True,
        )

        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=default_data_collator
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=default_data_collator
        )

        return train_loader, val_loader

    def _tokenize_commonsenseqa(self, examples):
        input_texts = []
        for q, choices, answer_key in zip(examples["question"], examples["choices"], examples["answerKey"]):
            # Convert choices into a label->text dict
            choices_text = {label: text for label, text in zip(choices["label"], choices["text"])}
            answer = choices_text[answer_key]

            prompt = f"Question: {q}\nAnswer:"
            completion = f" {answer} {self.tokenizer.eos_token}"
            full_text = prompt + completion

            # Tokenize prompt alone
            tokenized_prompt = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length
            )
            # Tokenize prompt+answer with padding
            tokenized_full = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )

            # Build label_ids so that prompt tokens are -100, while answer tokens are real IDs
            num_prompt_tokens = len(tokenized_prompt["input_ids"])
            full_ids = tokenized_full["input_ids"]

            label_ids = (
                [-100] * num_prompt_tokens
                + full_ids[num_prompt_tokens:]
            )
            # Truncate to max_length
            label_ids = label_ids[:self.max_length]

            input_texts.append({
                "input_ids": tokenized_full["input_ids"],
                "attention_mask": tokenized_full["attention_mask"],
                "labels": label_ids
            })

        # Convert list of dicts -> dict of lists
        batch = {k: [dic[k] for dic in input_texts] for k in input_texts[0]}
        return batch

    def _tokenize_openbookqa(self, examples):
        input_texts = []
        for q, choices, answer_key in zip(examples["question_stem"], examples["choices"], examples["answerKey"]):
            # Convert choices into a label->text dict
            choices_text = {label: text for label, text in zip(choices["label"], choices["text"])}
            answer = choices_text[answer_key]

            prompt = f"Question: {q}\nAnswer:"
            completion = f" {answer} {self.tokenizer.eos_token}"
            full_text = prompt + completion

            # Tokenize prompt alone
            tokenized_prompt = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length
            )
            # Tokenize prompt+answer with padding
            tokenized_full = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )

            # Build label_ids so that prompt tokens are -100, while answer tokens are real IDs
            num_prompt_tokens = len(tokenized_prompt["input_ids"])
            full_ids = tokenized_full["input_ids"]

            label_ids = (
                [-100] * num_prompt_tokens
                + full_ids[num_prompt_tokens:]
            )
            label_ids = label_ids[:self.max_length]

            input_texts.append({
                "input_ids": tokenized_full["input_ids"],
                "attention_mask": tokenized_full["attention_mask"],
                "labels": label_ids
            })

        batch = {k: [dic[k] for dic in input_texts] for k in input_texts[0]}
        return batch
