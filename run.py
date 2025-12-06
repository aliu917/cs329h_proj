"""
Main training script for DPO fine-tuning.

This script initializes models, loads datasets, and orchestrates the DPO fine-tuning
process with optional IPS weighting. Supports training on CPU, CUDA, or Apple MPS.
"""

import argparse
import sys

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
import wandb
import os

from custom_dpo import finetune
from dpo_dataset import DpoJsonlDataset

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

default_config = {
    "model_name": "google/gemma-2-2b-it",
    "batch_size": 4,
    "learning_rate": 2e-6,
    "num_epochs": 3,
    "beta": 1,
    "max_seq_len": 512,
    "ips": False,
    "device": device,
}

model_name = default_config["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HUGGINGFACE_TOKEN"))
train_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    token=os.environ.get("HUGGINGFACE_TOKEN"),
    low_cpu_mem_usage=True
)
lora_config = LoraConfig()
train_model = get_peft_model(train_model, lora_config)
train_model.train()

ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    token=os.environ.get("HUGGINGFACE_TOKEN"),
    low_cpu_mem_usage=True
)
ref_model.eval()

def dpo_collate(batch, tokenizer, add_generation_prompt=False):
    """
    Collate function for batching DPO training samples.

    Applies chat templates to prompts and responses, tokenizes them, and pads
    to create uniform batch tensors.

    Args:
        batch (list): List of dictionaries containing prompt, chosen_text, and rejected_text.
        tokenizer: HuggingFace tokenizer with chat template support.
        add_generation_prompt (bool): Whether to add generation prompt. Defaults to False.

    Returns:
        dict: Dictionary with keys:
            - chosen: Padded tensor batch for preferred responses
            - rejected: Padded tensor batch for non-preferred responses
    """
    chosen_texts = [
        [{"role": "user", "content": item["prompt"]},
         {"role": "assistant", "content": item["chosen_text"]}]
        for item in batch
    ]
    rejected_texts = [
        [{"role": "user", "content": item["prompt"]},
         {"role": "assistant", "content": item["rejected_text"]}]
        for item in batch
    ]

    chosen_encoding_ids = tokenizer.apply_chat_template(
        chosen_texts, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    rejected_encoding_ids = tokenizer.apply_chat_template(
        rejected_texts, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    chosen_encodings = [{"input_ids": ids} for ids in chosen_encoding_ids]
    rejected_encodings = [{"input_ids": ids} for ids in rejected_encoding_ids]

    chosen_batch = tokenizer.pad(chosen_encodings, padding=True, return_tensors="pt")
    rejected_batch = tokenizer.pad(rejected_encodings, padding=True, return_tensors="pt")
    # metadata = [item["metadata"] for item in batch]

    return {
        "chosen": chosen_batch,
        "rejected": rejected_batch,
    }

def train(path, wandb):
    """
    Run the DPO training loop.

    Loads the dataset, creates a dataloader, and runs fine-tuning with the specified
    configuration from wandb.

    Args:
        path (str): Path to the JSONL dataset file.
        wandb: Weights & Biases run object containing configuration.
    """
    dataset = DpoJsonlDataset(path, tokenizer=tokenizer, max_length=wandb.config.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=0, collate_fn=lambda batch: dpo_collate(batch, tokenizer))
    optimizer = torch.optim.Adam(train_model.parameters(), lr=wandb.config.learning_rate)
    finetune(wandb, optimizer, train_model, ref_model, dataloader, wandb.config.beta, wandb.config.num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    wandb.init(
        project="dpo-finetune",
        name="experiment-1",  # optional: give a descriptive run name
        config=default_config
    )
    if wandb.config.ips:
        path = "data/sampled_dpo_dataset.jsonl"
    else:
        path = "data/all_dpo_dataset.jsonl"
    train(path, wandb)


