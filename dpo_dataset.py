"""Dataset class for loading DPO (Direct Preference Optimization) training data."""

import json
from torch.utils.data import Dataset, DataLoader


class DpoJsonlDataset(Dataset):
    """
    PyTorch Dataset for DPO training from JSONL files.

    Loads preference pairs (chosen/rejected responses) for Direct Preference Optimization.
    Each sample contains a prompt with both a preferred (chosen) and non-preferred (rejected)
    response, along with metadata about sampling probabilities.

    Args:
        path (str): Path to the JSONL file containing DPO data.
        tokenizer (PreTrainedTokenizer, optional): HuggingFace tokenizer for encoding text.
            If None, returns raw text. Defaults to None.
        max_length (int, optional): Maximum sequence length for tokenization. Defaults to 1024.

    Attributes:
        data (list): List of dictionaries containing all loaded samples.
        tokenizer: The tokenizer instance.
        max_length (int): Maximum sequence length.
    """

    def __init__(self, path, tokenizer=None, max_length=1024):
        """Initialize the DPO dataset by loading data from JSONL file."""
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load all rows from JSONL into memory
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single training sample.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing:
                - prompt (str): The user prompt/question
                - chosen_text (str): The preferred response
                - rejected_text (str): The non-preferred response
                - sample_frac (float): IPS sampling probability for this user
        """
        row = self.data[idx]

        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        sample_frac = row.get("sample_frac")
        preferred_chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen}
        ]
        nonpreferred_chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected}
        ]
        metadata = {
            "user_id": row.get("user_id"),
            "user_llm_usage": row.get("user_llm_usage"),
            "prompt": prompt,
            "rejected": row.get("rejected"),
            "chosen": row.get("chosen"),
        }

        # If no tokenizer, return raw text
        if self.tokenizer is None:
            return {
                "chosen": chosen,
                "rejected": rejected,
                "sample_frac": sample_frac,
            }

        # return {
        #     "chosen": self.tokenizer.apply_chat_template(preferred_chat, tokenize=True, add_generation_prompt=False),
        #     "rejected": self.tokenizer.apply_chat_template(nonpreferred_chat, tokenize=True, add_generation_prompt=False),
        #     "sample_frac": sample_frac,
        # }
        return {
            "prompt": prompt,
            "chosen_text": chosen,
            "rejected_text": rejected,
            "sample_frac": sample_frac,
        }