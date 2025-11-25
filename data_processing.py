import json
from collections import defaultdict
import random

import pandas as pd
from datasets import load_dataset

ALL_OUTPUT_PATH = "data/all_dpo_dataset.jsonl"
SAMPLED_OUTPUT_PATH = "data/sampled_dpo_dataset.jsonl"
LLM_USAGE_MAPPING = {
    "Every day": 1,
    "Every week": 0.8,
    "More than once a month": 0.6,
    "Once per month": 0.4,
    "Less than one a year": 0.2,
}


def write_data(user_id, user_llm_usage, sample_frac, prompt, label_ouptuts, f):
    for chosen in label_ouptuts["chosen"]:
        for rejected in label_ouptuts["rejected"]:
            json.dump(
                {
                    "user_id": user_id,
                    "user_llm_usage": user_llm_usage,
                    "sample_frac": sample_frac,
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

def create_train_data():
    dataset = load_dataset(
        "json",
        data_files="hf://datasets/HannahRoseKirk/prism-alignment/utterances.jsonl",
        split="train"
    )
    user_data = load_dataset(
        "json",
        data_files="hf://datasets/HannahRoseKirk/prism-alignment/survey.jsonl",
        split="train"
    ).to_pandas()

    data = defaultdict(lambda: defaultdict(lambda: {"chosen": [], "rejected": []}))
    for row in dataset:
        user_id = row["user_id"]
        prompt = row.get("user_prompt")
        response = row.get("model_response")
        is_chosen = row.get("if_chosen")

        if prompt and response:
            if is_chosen:
                data[user_id][prompt]["chosen"].append(response)
            else:
                data[user_id][prompt]["rejected"].append(response)

    with open(ALL_OUTPUT_PATH, "w", encoding="utf-8") as f1, \
            open(SAMPLED_OUTPUT_PATH, "w", encoding="utf-8") as f2:
        for user_id in data.keys():
            user_survey_data = user_data.loc[user_data["user_id"] == user_id]
            user_lm_freq = user_survey_data["lm_frequency_use"].iloc[0]
            sample_frac = LLM_USAGE_MAPPING[user_lm_freq] if user_lm_freq in LLM_USAGE_MAPPING else 0
            for prompt, label_outputs in data[user_id].items():
                write_data(user_id, user_lm_freq, sample_frac, prompt, label_outputs, f1)
                if random.random() < sample_frac:
                    write_data(user_id, user_lm_freq, sample_frac, prompt, label_outputs, f2)


def filter_user_data(input_jsonl_path: str, max_user_id: int = 200) -> pd.DataFrame:
    df = pd.read_json(input_jsonl_path, lines=True)
    print(f"Original number of rows: {len(df)}")
    initial_distribution = df['user_llm_usage'].value_counts(normalize=True) * 100
    print(initial_distribution.round(2).to_string())

    df['numeric_id'] = df['user_id'].str.replace('user', '', regex=False).astype(int)
    filtered_df = df[df['numeric_id'] < max_user_id].copy()
    filtered_df = filtered_df.drop(columns=['numeric_id'])

    print(f"Filtered number of rows (keeping user IDs < {max_user_id}): {len(filtered_df)}")
    filtered_distribution = filtered_df['user_llm_usage'].value_counts(normalize=True) * 100
    print(filtered_distribution.round(2).to_string())

    return filtered_df


def run_filter(input_path, output_path, max_user_id: int = 200):
    filtered_df = filter_user_data(input_path, max_user_id)
    filtered_df.to_json(output_path, orient='records', lines=True)


def create_val_set(input_path, val_count=100):
    df = pd.read_json(input_path, lines=True)
    indices = random.sample(range(200, len(df)), val_count)

    val_df = df.iloc[indices]
    val_path = input_path.split(".")[0] + f"_val_{val_count}.jsonl"
    val_df.to_json(val_path, orient='records', lines=True)

    val_distr = val_df['user_llm_usage'].value_counts(normalize=True) * 100
    print(val_distr.round(2).to_string())


def count_df(input_path):
    df = pd.read_json(input_path, lines=True)
    print(f"Original number of rows: {len(df)}")

    inv_sum = (1 / df['values']).sum()
    print("sum of IPS:", inv_sum)



if __name__ == '__main__':
    # create_train_data()
    # filter_user_data("data/all_dpo_dataset.jsonl")
    # run_filter("data/sampled_dpo_dataset.jsonl", "data/sampled_dpo_dataset_200.jsonl")
    create_val_set("data/all_dpo_dataset.jsonl")
