# data_handler.py

from datasets import load_dataset
from transformers import GPT2Tokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader


class DataHandler:
    """
    Handles dataset loading, tokenization, and DataLoader creation for CQA and OBQA.
    """

    def __init__(self, dataset_name="commonsense_qa", batch_size=8, max_length=128):
        """
        Initialize the DataHandler.

        :param dataset_name: Name of the dataset to load ("commonsense_qa" or "openbookqa").
        :param batch_size: Batch size for DataLoader.
        :param max_length: Maximum sequence length for tokenization.
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")

    def load_dataset(self):
        """
        Load and tokenize the dataset.

        :return: Tuple of (train_loader, val_loader).
        """
        # Load the dataset
        dataset = load_dataset(self.dataset_name)
        train_data = dataset["train"]
        val_data = dataset["validation"]

        # Tokenize the datasets
        if self.dataset_name == "commonsense_qa":
            train_dataset = train_data.map(self._tokenize_commonsenseqa, batched=True)
            val_dataset = val_data.map(self._tokenize_commonsenseqa, batched=True)
        elif self.dataset_name == "openbookqa":
            train_dataset = train_data.map(self._tokenize_openbookqa, batched=True)
            val_dataset = val_data.map(self._tokenize_openbookqa, batched=True)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        # Filter columns to keep only input_ids and attention_mask
        def filter_cols(ds):
            return ds.remove_columns([c for c in ds.column_names if c not in ["input_ids", "attention_mask"]])

        train_dataset = filter_cols(train_dataset)
        val_dataset = filter_cols(val_dataset)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  collate_fn=self.data_collator)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.data_collator)

        return train_loader, val_loader

    def _tokenize_commonsenseqa(self, examples):
        """
        Tokenizer for CommonsenseQA with batched=True.
        """
        text_list = []
        for question, choices_dict in zip(examples["question"], examples["choices"]):
            merged_choice_text = " ".join(choices_dict["text"])
            combined_str = question + " " + merged_choice_text
            text_list.append(combined_str)

        return self.tokenizer(
            text_list,
            truncation=True,
            max_length=self.max_length,
            padding=False  # Rely on DataCollatorWithPadding
        )

    def _tokenize_openbookqa(self, examples):
        """
        Tokenizer for OpenBookQA with batched=True.
        """
        text_list = []
        for question_stem, choices_dict in zip(examples["question_stem"], examples["choices"]):
            merged_choice_text = " ".join(choices_dict["text"])
            combined_str = question_stem + " " + merged_choice_text
            text_list.append(combined_str)

        return self.tokenizer(
            text_list,
            truncation=True,
            max_length=self.max_length,
            padding=False
        )