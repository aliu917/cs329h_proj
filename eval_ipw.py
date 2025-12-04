import json
import os

import pandas as pd
from util import seed
from sentence_transformers import SentenceTransformer, util

def load_df():
    fformat = lambda x: f"out/gen_result{x}_fixtemplate.jsonl"
    all_df = pd.read_json(fformat("all"), lines=True)
    ips_df = pd.read_json(fformat("ips"), lines=True)
    noips_df = pd.read_json(fformat("noips"), lines=True)

    all_df = all_df.rename(columns={'model_response': 'model_response_all'})
    ips_df = ips_df.rename(columns={'model_response': 'model_response_ips'})
    noips_df = noips_df.rename(columns={'model_response': 'model_response_noips'})

    JOIN_KEYS = ['user_id', 'prompt']
    merged_df = all_df
    merged_df = pd.merge(merged_df, ips_df, on=JOIN_KEYS, how='inner')
    final_df = pd.merge(merged_df, noips_df, on=JOIN_KEYS, how='inner')

    return final_df


def calc_bertsim(model, s1, s2):
    embeddings = model.encode([s1, s2], convert_to_tensor=True)
    similarity_score = util.cos_sim(embeddings[0], embeddings[1])
    return similarity_score.item()


def run_eval_similarity():
    final_df = load_df()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    overall_acc = 0
    ips_scores = []
    noips_scores = []

    for index, row in final_df.iterrows():
        ips_sim_score = calc_bertsim(model, row["model_response_all"], row["model_response_ips"])
        noips_sim_score = calc_bertsim(model, row["model_response_all"], row["model_response_noips"])

        ips_scores.append(ips_sim_score)
        noips_scores.append(noips_sim_score)
        if ips_sim_score > noips_sim_score:
            overall_acc += 1

    final_df['ips_score'] = ips_scores
    final_df['noips_score'] = noips_scores
    accuracy = overall_acc / len(final_df)

    output_dir = "out/logs/run1"
    os.makedirs(output_dir, exist_ok=True)
    df_output_path = os.path.join(output_dir, "similarity_results.csv")
    final_df.to_csv(df_output_path, index=False)

    results_output_path = os.path.join(output_dir, "result.txt")
    with open(results_output_path, 'w') as f:
        f.write(f"{accuracy:.6f}\n")

    return overall_acc / len(final_df)


if __name__ == '__main__':
    seed()
    result = run_eval_similarity()
    print(result)
