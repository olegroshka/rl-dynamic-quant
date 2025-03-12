from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from transformers import GPT2Tokenizer, default_data_collator
from torch.utils.data import DataLoader


class DataHandler:
    def __init__(self,
                 dataset_name="commonsense_qa",
                 batch_size=8,
                 max_length=128,
                 use_lm_chunking=False,
                 block_size=128):
        """
        Args:
            dataset_name: 'commonsense_qa' or 'openbookqa'
            batch_size: ...
            max_length: typical max sequence length for QA encoding
            use_lm_chunking: if True, we create a "block dataset" for LM-style perplexity
            block_size: length of each LM chunk if use_lm_chunking=True
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_lm_chunking = use_lm_chunking
        self.block_size = block_size

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

        if not self.use_lm_chunking:
            # ====== ORIGINAL QA TOKENIZATION (for MC accuracy & short-answer PPL) ====== #
            if not self.use_lm_chunking:
                if self.dataset_name == "commonsense_qa":
                    tokenize_fn = self._tokenize_commonsenseqa
                elif self.dataset_name == "openbookqa":
                    tokenize_fn = self._tokenize_openbookqa
                elif self.dataset_name == "boolq":
                    tokenize_fn = self._tokenize_boolq
                else:
                    raise ValueError(f"Unsupported dataset: {self.dataset_name}")

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

        else:
            # ====== LM-STYLE CHUNKING FOR PERPLEXITY ====== #
            train_dataset = train_data.map(
                self._prep_lm_text,
                remove_columns=train_data.column_names,
                batched=True,
            )
            val_dataset = val_data.map(
                self._prep_lm_text,
                remove_columns=val_data.column_names,
                batched=True,
            )
            # Now group into block-size chunks
            train_dataset = train_dataset.map(
                self._group_texts,
                batched=True,
            )
            val_dataset = val_dataset.map(
                self._group_texts,
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
        """
        Original QA tokenization: For multiple-choice scoring,
        the question tokens get -100 as labels, and only the
        short answer portion is predicted.
        """
        input_texts = []
        for q, choices, answer_key in zip(examples["question"],
                                          examples["choices"],
                                          examples["answerKey"]):
            choices_text = {label: text for label, text in zip(choices["label"], choices["text"])}
            answer = choices_text[answer_key]

            prompt = f"Question: {q}\nAnswer:"
            completion = f" {answer} {self.tokenizer.eos_token}"
            full_text = prompt + completion

            tokenized_prompt = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length
            )
            tokenized_full = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )

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

    def _tokenize_openbookqa(self, examples):
        """
        Same logic for openbookqa, with the question text in "question_stem".
        """
        input_texts = []
        for q, choices, answer_key in zip(examples["question_stem"],
                                          examples["choices"],
                                          examples["answerKey"]):
            choices_text = {label: text for label, text in zip(choices["label"], choices["text"])}
            answer = choices_text[answer_key]

            prompt = f"Question: {q}\nAnswer:"
            completion = f" {answer} {self.tokenizer.eos_token}"
            full_text = prompt + completion

            tokenized_prompt = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length
            )
            tokenized_full = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )

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

    def _tokenize_boolq(self, examples):
        """
        For BoolQ, each example has:
           "passage": string
           "question": string
           "answer": bool (True/False)
        We'll treat this as a multiple-choice with two choices: "Yes", "No".
        The gold index is 1 if answer=True, else 0.

        We'll do the same style as we do for the others:
        mask out the prompt tokens, and only the final answer gets predicted.
        """
        input_texts = []

        for passage, question, ans in zip(examples["passage"], examples["question"], examples["answer"]):
            # We'll create two "virtual" choices:
            #  label A -> "No", label B -> "Yes"
            # gold_index = 1 if ans==True, else 0
            # But if you're only measuring perplexity, we'll just keep the correct choice tokens as labels.

            # For perplexity or single correct choice, let's do the same style:
            # prompt = "Passage: {passage}\nQuestion: {question}\nAnswer:"
            # completion = " Yes" or " No"

            prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
            completion = f" {'Yes' if ans else 'No'} {self.tokenizer.eos_token}"
            full_text = prompt + completion

            # Tokenize the prompt alone:
            tokenized_prompt = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length
            )
            # Tokenize full text (prompt+answer)
            tokenized_full = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )

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

    def _tokenize_piqa(self, examples):
        """
        piqa has:
          "goal" (string),
          "sol1" (string),
          "sol2" (string),
          "label" (0 or 1) -> which sol is correct
        We'll structure it as a standard multiple-choice:
           prompt = f"Goal: {goal}\nSolution: "
           choice A -> sol1
           choice B -> sol2
        Then mask out the prompt, only solution tokens are labeled.
        """
        input_texts = []
        for goal, sol1, sol2, label in zip(examples["goal"],
                                           examples["sol1"],
                                           examples["sol2"],
                                           examples["label"]):
            # We'll only store the correct choice for perplexity if that's your approach
            # If you want to do forced decoding for both to compare, that's in evaluate_mc_accuracy.

            prompt = f"Goal: {goal}\nSolution:"
            chosen_sol = sol1 if label == 0 else sol2
            completion = f" {chosen_sol} {self.tokenizer.eos_token}"
            full_text = prompt + completion

            tokenized_prompt = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length
            )
            tokenized_full = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )

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


    # --------------------------- LM-CHUNKING --------------------------- #
    def _prep_lm_text(self, examples):
        """
        For each row, build a single text string (question + correct answer).
        Return a tokenized dict with keys like "input_ids", "attention_mask".
        We'll group them into blocks later.
        """
        texts = []
        if self.dataset_name == "commonsense_qa":
            for q, choices, answer_key in zip(examples["question"],
                                              examples["choices"],
                                              examples["answerKey"]):
                choices_text = {label: text for label, text in zip(choices["label"], choices["text"])}
                answer = choices_text[answer_key]
                text = f"Question: {q}\nAnswer: {answer}\n"
                texts.append(text)

        elif self.dataset_name == "openbookqa":
            for q, choices, answer_key in zip(examples["question_stem"],
                                              examples["choices"],
                                              examples["answerKey"]):
                choices_text = {label: text for label, text in zip(choices["label"], choices["text"])}
                answer = choices_text[answer_key]
                text = f"Question: {q}\nAnswer: {answer}\n"
                texts.append(text)

        elif self.dataset_name == "boolq":
            # Each example has: "passage", "question", "answer" (bool)
            for passage, question, ans in zip(examples["passage"],
                                              examples["question"],
                                              examples["answer"]):
                # We'll just convert bool to "Yes"/"No"
                answer_text = "Yes" if ans else "No"
                text = f"Passage: {passage}\nQuestion: {question}\nAnswer: {answer_text}\n"
                texts.append(text)

        elif self.dataset_name == "piqa":
            # Each example has: "goal", "sol1", "sol2", "label" (0 or 1)
            for goal, sol1, sol2, label in zip(examples["goal"],
                                               examples["sol1"],
                                               examples["sol2"],
                                               examples["label"]):
                # We'll just include the correct solution
                correct_sol = sol1 if label == 0 else sol2
                text = f"Goal: {goal}\nAnswer: {correct_sol}\n"
                texts.append(text)

        else:
            raise ValueError(f"Unsupported dataset in _prep_lm_text: {self.dataset_name}")

        # Now we tokenize all texts in this batch
        tokenized = self.tokenizer(
            texts,
            return_special_tokens_mask=True,
            truncation=True,
            max_length=self.max_length,
            # no padding, because we will eventually concat
        )
        return tokenized

    def _group_texts(self, examples):
        """
        Standard "group_texts" method:
        1) Concat all input_ids into a single list.
        2) Chunk into blocks of size self.block_size.
        3) 'labels' = same as 'input_ids' for LM training/perplexity.
        """
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size

        result = {}
        for k in concatenated.keys():
            result[k] = [
                concatenated[k][i: i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
        # For LM, labels = input_ids
        result["labels"] = [chunk[:] for chunk in result["input_ids"]]
        return result
