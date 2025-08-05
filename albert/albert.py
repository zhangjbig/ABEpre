import torch
import numpy as np
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from torch.utils.data import Dataset, DataLoader
from transformers import AlbertConfig, AlbertForMaskedLM, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
import sentencepiece as spm

# Load SentencePiece tokenizer
tokenizer = spm.SentencePieceProcessor(model_file='./custom_tokenizer/custom_tokenizer.model')


class AminoAcidDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.load(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.data[idx], dtype=torch.long),
            "attention_mask": torch.tensor((self.data[idx] != 0).astype(int), dtype=torch.long)
        }

# Load the data
train_dataset = AminoAcidDataset('train_data.npy')
val_dataset = AminoAcidDataset('val_data.npy')

# Create ALBERT configuration object
config = AlbertConfig(
    vocab_size=tokenizer.get_piece_size(),
    hidden_size=128,
    num_attention_heads=2,
    num_hidden_layers=2,
    intermediate_size=512,
)

# Create ALBERT model object
model = AlbertForMaskedLM(config=config)


# Custom data collator
class CustomDataCollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, examples):
        batch = {}
        batch["input_ids"] = torch.stack([item["input_ids"] for item in examples])
        batch["attention_mask"] = torch.stack([item["attention_mask"] for item in examples])

        # Here, you should implement the logic to mask tokens for MLM using the SentencePiece tokenizer.
        # For simplicity, we assume the input data is already masked.
        batch["labels"] = batch["input_ids"].clone()

        return batch

    def mask_tokens(self, inputs):
        """ Prepare masked tokens inputs/labels for masked language modeling:
        15% MASK, 85% original. """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability mlm_probability)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.piece_to_id('[MASK]')

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels



data_collator = CustomDataCollator(tokenizer, mlm_probability=0.15)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./albert-custom",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

# Create a trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # If you have validation data
)

# Start training
trainer.train()

# Save only the model (since the tokenizer is saved separately with SentencePiece)
model.save_pretrained('./albert-custom')