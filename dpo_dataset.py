import json
from torch.utils.data import Dataset, DataLoader


class DpoJsonlDataset(Dataset):
    def __init__(self, path, tokenizer=None, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load all rows from JSONL into memory
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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