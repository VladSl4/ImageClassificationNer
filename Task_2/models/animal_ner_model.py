import os
import torch
from typing import List
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    pipeline
)
from datasets import load_dataset
from interfaces.ner_interface import NerInterface


class AnimalNerModel(NerInterface):
    def __init__(self, model_dir: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        self.model_name = 'distilbert-base-uncased'

        self.id2label = {0: 'O', 1: 'B-ANIMAL'}
        self.label2id = {'O': 0, 'B-ANIMAL': 1}

        if self.model_dir and os.path.exists(self.model_dir):
            self._load_model()
        else:
            self.tokenizer = None
            self.model = None
            self.ner_pipeline = None

    def _load_model(self):
        print(f"Loading NER model from {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)

        self.ner_pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )

    def train(self, dataset_path: str, **kwargs):
        num_epochs = kwargs.get('num_epochs', 3)
        batch_size = kwargs.get('batch_size', 16)
        learning_rate = kwargs.get('learning_rate', 2e-5)
        save_path = kwargs.get('save_dir') or kwargs.get('save_path') or 'weights/best_ner_model'

        print(f"Loading base model {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
        ).to(self.device)

        raw_datasets = load_dataset("json", data_files={"train": dataset_path})

        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
                examples["tokens"],
                truncation=True,
                is_split_into_words=True
            )

            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)

        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir='./results',
            eval_strategy='no',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            save_strategy='no',
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            data_collator=data_collator,
            processing_class=self.tokenizer
        )

        print("Training transformer")
        trainer.train()

        print(f"Saving model and tokenizer to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def predict(self, text: str, **kwargs) -> List[str]:
        if not self.ner_pipeline:
            self.ner_pipeline = pipeline(
                "token-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )

        ner_results = self.ner_pipeline(text)

        animals = []
        for entity in ner_results:
            if entity['entity_group'] == 'ANIMAL':
                word = entity['word'].strip()
                animals.append(word)

        return list(set(animals))