import json
import os
import openai
import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from util import seed
from sentence_transformers import SentenceTransformer, util

def load_survey_data():
    """
    Load user survey data including LLM usage frequency.

    Returns:
        pd.DataFrame: Survey data with user_id and lm_frequency_use columns.
    """
    dataset = load_dataset(
        "json",
        data_files="hf://datasets/HannahRoseKirk/prism-alignment/survey.jsonl",
        split="train"
    )
    return dataset.to_pandas()


def load_df(survey_df):
    """
    Load and merge model outputs with user metadata.

    Args:
        survey_df (pd.DataFrame): Survey data with user metadata.

    Returns:
        pd.DataFrame: Merged dataframe with model responses and user LLM frequency.
    """
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

    # Add user LLM frequency metadata
    final_df = pd.merge(final_df, survey_df[['user_id', 'lm_frequency_use']],
                        on='user_id', how='left')

    return final_df


def calculate_metrics_by_group(df):
    """
    Calculate overall and per-group accuracy and score statistics.

    Args:
        df (pd.DataFrame): Dataframe with ips_score, noips_score, and lm_frequency_use columns.

    Returns:
        dict: Dictionary with 'overall' and per-group metrics including:
              - accuracy: fraction where ips_score > noips_score
              - correct_count: number of correct predictions
              - total_count: total number of non-tied samples
              - avg_ips_score: mean IPS score
              - avg_noips_score: mean noIPS score
              - total_samples: total number of samples (including ties)
    """
    metrics = {}

    # Calculate overall metrics
    df_no_ties = df[df['ips_score'] != df['noips_score']]
    correct_count = (df_no_ties['ips_score'] > df_no_ties['noips_score']).sum()
    total_count = len(df_no_ties)

    metrics['overall'] = {
        'accuracy': correct_count / total_count if total_count > 0 else 0.0,
        'correct_count': int(correct_count),
        'total_count': int(total_count),
        'avg_ips_score': float(df['ips_score'].mean()),
        'avg_noips_score': float(df['noips_score'].mean()),
        'total_samples': len(df)
    }

    # Calculate per-group metrics
    if 'lm_frequency_use' in df.columns:
        for group_name, group_df in df.groupby('lm_frequency_use'):
            group_no_ties = group_df[group_df['ips_score'] != group_df['noips_score']]
            group_correct = (group_no_ties['ips_score'] > group_no_ties['noips_score']).sum()
            group_total = len(group_no_ties)

            metrics[group_name] = {
                'accuracy': group_correct / group_total if group_total > 0 else 0.0,
                'correct_count': int(group_correct),
                'total_count': int(group_total),
                'avg_ips_score': float(group_df['ips_score'].mean()),
                'avg_noips_score': float(group_df['noips_score'].mean()),
                'total_samples': len(group_df)
            }

    return metrics


def save_grouped_results(df, metrics, output_dir, prefix):
    """
    Save results including grouped metrics to files.

    Args:
        df (pd.DataFrame): Full results dataframe.
        metrics (dict): Metrics dictionary from calculate_metrics_by_group.
        output_dir (str): Directory to save results.
        prefix (str): Prefix for output files (e.g., 'similarity' or 'llmjudge').
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save full results CSV
    df_output_path = os.path.join(output_dir, f"{prefix}_results.csv")
    df.to_csv(df_output_path, index=False)

    # Save overall result
    overall = metrics['overall']
    results_output_path = os.path.join(output_dir, f"{prefix}_result.txt")
    with open(results_output_path, 'w') as f:
        f.write(f"Overall Accuracy: {overall['accuracy']:.6f} "
                f"({overall['correct_count']}/{overall['total_count']})\n")
        f.write(f"Avg IPS Score: {overall['avg_ips_score']:.6f}\n")
        f.write(f"Avg NoIPS Score: {overall['avg_noips_score']:.6f}\n")
        f.write(f"Total Samples: {overall['total_samples']}\n")

    # Save grouped breakdown
    grouped_output_path = os.path.join(output_dir, f"{prefix}_grouped_results.txt")
    with open(grouped_output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Accuracy: {overall['accuracy']:.6f} "
                f"({overall['correct_count']}/{overall['total_count']})\n")
        f.write(f"Avg IPS Score: {overall['avg_ips_score']:.6f}\n")
        f.write(f"Avg NoIPS Score: {overall['avg_noips_score']:.6f}\n")
        f.write(f"Total Samples: {overall['total_samples']}\n\n")

        # Write per-group metrics
        f.write("=" * 80 + "\n")
        f.write("BREAKDOWN BY USER LLM FREQUENCY\n")
        f.write("=" * 80 + "\n\n")

        # Sort groups for consistent ordering
        group_order = ["Every day", "Every week", "More than once a month",
                       "Once per month", "Less than one a year"]
        for group_name in group_order:
            if group_name in metrics:
                group = metrics[group_name]
                f.write(f"Group: {group_name}\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Accuracy: {group['accuracy']:.6f} "
                        f"({group['correct_count']}/{group['total_count']})\n")
                f.write(f"  Avg IPS Score: {group['avg_ips_score']:.6f}\n")
                f.write(f"  Avg NoIPS Score: {group['avg_noips_score']:.6f}\n")
                f.write(f"  Total Samples: {group['total_samples']}\n\n")


def visualize_grouped_metrics(metrics, output_dir, prefix, metric_name="Score"):
    """
    Create side-by-side bar chart comparing IPS vs noIPS scores by user frequency group.

    Args:
        metrics (dict): Metrics dictionary from calculate_metrics_by_group.
        output_dir (str): Directory to save the chart.
        prefix (str): Prefix for output file (e.g., 'similarity' or 'llmjudge').
        metric_name (str): Label for the metric being visualized. Defaults to "Score".
    """
    # Define group ordering from highest to lowest frequency
    group_order = ["Every day", "Every week", "More than once a month",
                   "Once per month", "Less than one a year"]

    # Extract data for groups that exist in metrics
    groups = []
    ips_scores = []
    noips_scores = []

    for group_name in group_order:
        if group_name in metrics:
            groups.append(group_name)
            ips_scores.append(metrics[group_name]['avg_ips_score'])
            noips_scores.append(metrics[group_name]['avg_noips_score'])

    if not groups:
        print(f"Warning: No group data available for visualization")
        return

    # Set up the bar chart
    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, ips_scores, width, label='IPS Model', color='#2E86AB')
    bars2 = ax.bar(x + width/2, noips_scores, width, label='No IPS Model', color='#A23B72')

    # Customize the chart
    ax.set_xlabel('User LLM Frequency', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Average {metric_name}', fontsize=12, fontweight='bold')

    # Create title based on evaluation method
    if 'similarity' in prefix.lower():
        title = 'BERT Similarity Scores by User LLM Frequency'
    elif 'llm' in prefix.lower() or 'judge' in prefix.lower():
        title = 'LLM Judge Scores by User LLM Frequency'
    else:
        title = f'{metric_name} by User LLM Frequency'

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)

    autolabel(bars1)
    autolabel(bars2)

    # Add overall scores as horizontal reference lines
    if 'overall' in metrics:
        overall_ips = metrics['overall']['avg_ips_score']
        overall_noips = metrics['overall']['avg_noips_score']
        ax.axhline(y=overall_ips, color='#2E86AB', linestyle=':', alpha=0.5,
                   label=f'Overall IPS: {overall_ips:.3f}')
        ax.axhline(y=overall_noips, color='#A23B72', linestyle=':', alpha=0.5,
                   label=f'Overall No IPS: {overall_noips:.3f}')
        ax.legend(fontsize=10, loc='best')

    plt.tight_layout()

    # Save the chart
    chart_path = os.path.join(output_dir, f"{prefix}_grouped_chart.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved visualization: {prefix}_grouped_chart.png")
    plt.close()


def calc_bertsim(model, s1, s2):
    embeddings = model.encode([s1, s2], convert_to_tensor=True)
    similarity_score = util.cos_sim(embeddings[0], embeddings[1])
    return similarity_score.item()


def run_eval_ipw_similarity(survey_df, run_name="run1"):
    """
    Evaluate models using BERT similarity with overall and grouped metrics.

    Args:
        survey_df (pd.DataFrame): User survey data with LLM frequency.
        run_name (str): Name for this evaluation run.

    Returns:
        float: Overall accuracy.
    """
    final_df = load_df(survey_df)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    ips_scores = []
    noips_scores = []

    print("Computing BERT similarities...")
    for index, row in final_df.iterrows():
        ips_sim_score = calc_bertsim(model, row["model_response_all"], row["model_response_ips"])
        noips_sim_score = calc_bertsim(model, row["model_response_all"], row["model_response_noips"])

        ips_scores.append(ips_sim_score)
        noips_scores.append(noips_sim_score)

    final_df['ips_score'] = ips_scores
    final_df['noips_score'] = noips_scores

    # Calculate overall and grouped metrics
    metrics = calculate_metrics_by_group(final_df)

    # Save results
    output_dir = "out/results/" + run_name
    save_grouped_results(final_df, metrics, output_dir, "similarity")

    # Generate visualization
    visualize_grouped_metrics(metrics, output_dir, "similarity", metric_name="BERT Similarity")

    return metrics['overall']['accuracy']


class GPTQuery:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.context = [] # Later if we want to do multiturn conversations

    def query(self, prompt, max_tokens=1000, use_context=False):
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages= [*self.context, {"role": "user", "content": prompt}] if use_context \
                                        else [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
                seed=1,
                n=1,
            )
            return response.choices[0].message.content
        except Exception as e:
            print("Error querying GPT-4:", e)
            return None


def calc_llm_score(llm_judge, utterances_df, response_row):
    """
    Score model responses using LLM-as-a-judge based on user's historical scores.

    Args:
        llm_judge (GPTQuery): GPT query object for scoring.
        response_row (pd.Series): Row containing user_id, prompt, and model responses.

    Returns:
        tuple: (ips_score, noips_score) as floats between 0 and 1.
    """
    # Extract user_id and prompt from the row
    user_id = response_row['user_id']
    prompt = response_row['prompt']

    # Find all examples from the same user and prompt with their scores
    user_examples = utterances_df[
        (utterances_df['user_id'] == user_id) &
        (utterances_df['user_prompt'] == prompt)
    ]

    # Build context from user's previous scores
    examples_text = ""
    if len(user_examples) > 0:
        examples_text = "User's previous scores for this same prompt:\n\n"

        # Show up to 5 examples with their scores
        for idx, (_, ex) in enumerate(user_examples.head(5).iterrows(), 1):
            score = ex.get('score', 'N/A')
            response_text = ex['model_response']
            examples_text += f"Example {idx}:\nResponse: \"{response_text}\"\nUser's Score: {score}\n"

    use_examples_text = ""
    if examples_text:
        use_examples_text = "Use the user's previous scores as calibration to understand their preferences and scoring standards."

    # Create scoring prompt template (1-100 scale to match PRISM)
    scoring_prompt_template = """You are evaluating AI assistant responses based on a specific user's scoring preferences.

Prompt: "{prompt}"

{examples}

Based on how this user scored previous responses to the same prompt, please rate the following response on a scale from 1 to 100, where:
- 1 = completely unacceptable/lowest quality
- 100 = excellent/highest quality

{use_examples_text}

Response to evaluate: "{response}"

Provide ONLY a numerical score between 1 and 100 (e.g., 75). Do not include any explanation or additional text."""

    # Score IPS response
    ips_prompt = scoring_prompt_template.format(
        prompt=prompt,
        examples=examples_text,
        use_examples_text=use_examples_text,
        response=response_row['model_response_ips']
    )

    ips_response = llm_judge.query(ips_prompt, max_tokens=10)
    ips_score = parse_score(ips_response) / 100.0

    # Score noIPS response
    noips_prompt = scoring_prompt_template.format(
        prompt=prompt,
        examples=examples_text,
        use_examples_text=use_examples_text,
        response=response_row['model_response_noips']
    )

    noips_response = llm_judge.query(noips_prompt, max_tokens=10)
    noips_score = parse_score(noips_response) / 100.0

    return ips_score, noips_score


def parse_score(gpt_output):
    """
    Parse an integer score from GPT output.

    Args:
        gpt_output (str): Raw GPT response text.

    Returns:
        int: Parsed score between 1 and 100, or 50 if parsing fails.
    """
    if gpt_output is None:
        return 50  # Default neutral score on error

    # Try to extract an integer between 1 and 100
    match = re.search(r'(\d+)', gpt_output.strip())

    if match:
        try:
            score = int(match.group(1))
            # Clamp to [1, 100] range
            score = max(1, min(100, score))
            return score
        except ValueError:
            pass

    # If parsing fails, return neutral score
    print(f"Warning: Could not parse score from: {gpt_output}")
    return -1


def run_eval_llm_judge_score(survey_df, utterances_df, run_name="run1"):
    """
    Evaluate models using LLM-as-judge with overall and grouped metrics.

    Args:
        survey_df (pd.DataFrame): User survey data with LLM frequency.
        utterances_df (pd.DataFrame): User utterances with historical scores.
        run_name (str): Name for this evaluation run.

    Returns:
        float: Overall accuracy.
    """
    final_df = load_df(survey_df)
    gpt_obj = GPTQuery()

    ips_scores = []
    noips_scores = []
    valid_indices = []

    print("Computing LLM judge scores...")
    for index, row in tqdm(final_df.iterrows(), total=len(final_df)):
        ips_score, noips_score = calc_llm_score(gpt_obj, utterances_df, row)

        # Skip rows with parsing errors
        if ips_score < 0 or noips_score < 0:
            continue

        ips_scores.append(ips_score)
        noips_scores.append(noips_score)
        valid_indices.append(index)

    # Filter to valid rows
    final_df = final_df.loc[valid_indices].copy()
    final_df['ips_score'] = ips_scores
    final_df['noips_score'] = noips_scores

    # Calculate overall and grouped metrics
    metrics = calculate_metrics_by_group(final_df)

    # Save results
    output_dir = "out/results/" + run_name
    save_grouped_results(final_df, metrics, output_dir, "llmjudge")

    # Generate visualization
    visualize_grouped_metrics(metrics, output_dir, "llmjudge", metric_name="LLM Judge Score")

    return metrics['overall']['accuracy']


if __name__ == '__main__':
    run_name = "run_grouped_graph"

    survey_df = load_survey_data()

    # Run BERT similarity evaluation
    seed()
    ipw_result = run_eval_ipw_similarity(survey_df, run_name)
    print(f"\nOverall Accuracy: {ipw_result:.6f}")

    # Run LLM Judge evaluation
    utterances_df = load_dataset(
        "json",
        data_files="hf://datasets/HannahRoseKirk/prism-alignment/utterances.jsonl",
        split="train"
    ).to_pandas()

    seed()
    llm_judge_result = run_eval_llm_judge_score(survey_df, utterances_df, run_name)
    print(f"\nOverall Accuracy: {llm_judge_result:.6f}")
