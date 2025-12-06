# DPO Fine-tuning with Inverse Propensity Weighting (IPW)

This project implements IPW Direct Preference Optimization (DPO) fine-tuning for language models and runs experiments and evaluations. The code fine-tunes Google's Gemma-2-2b-it model on the PRISM alignment dataset and compares performance between models trained with IPS weighting, without IPS weighting, and using all unbiased data.

## Environment Setup and Dependencies

The conda environment dependencies are provided in `environment.yml` and the one used in colab is `requirements_colab.txt`. In addition to the requirements, we assume the following prerequisites:
- Python 3.8+
- Google Colab: all training and ipynb are set up to run on Google Colab by uploading the entire directory to Google Drive and mounting the notebook to the code folder.
- HuggingFace account and API token (for accessing Gemma model and PRISM dataset)
- Weights & Biases (wandb) account for experiment tracking
- OpenAI API key (for LLM-as-judge evaluation in `eval_ipw.py`)

## Repository Structure

```
.
├── README.md                       # This file
├── requirements_colab.txt          # Python dependencies for running Colab training
├── environment.yml                 # Conda environment for running data processing, training, and eval locally
│
├── custom_dpo.py                   # Core DPO training implementation
├── dpo_dataset.py                  # Dataset loader for DPO training
├── data_processing.py              # Data preprocessing and sampling utilities
├── util.py                         # Utility functions (seeding)
│
├── run.py                          # Main training script (for local testing, not used)
├── eval_ipw.py                     # Eval script for BERT similarity and LLM-as-judge (uses results from out/ folder)
│
├── run.ipynb                       # Full training workflow notebook for entire PRISM dataset training
├── run_toy.ipynb                   # Toy example training (all unbiased training)
├── run_toy_ips.ipynb               # Toy example training (with IPS)
├── run_toy_noips.ipynb             # Toy example training (without IPS)
│
├── eval_test_all.ipynb             # Notebook for inference outputs (using checkpoint of all data model)
├── eval_test_ips.ipynb             # Notebook for inference outputs (using checkpoint of IPS model)
├── eval_test_noips.ipynb           # Notebook for inference outputs (using checkpoint of no-IPS model)
│
├── data/                           # Dataset directory
│   ├── all_dpo_dataset.jsonl      # Full DPO dataset
│   ├── sampled_dpo_dataset.jsonl  # IPS-weighted sampled dataset
│   ├── *_200.jsonl                # Filtered datasets (first 200 users)
│   ├── *_val_*.jsonl              # Validation sets
│   └── *_toy*.jsonl               # Various small toy datasets used for experimenting with the toy model
│
├── out/                            # Output directory
│   ├── gen_result*.jsonl          # Generated model responses (used the *fixtemplate ones)
│   └── results/                   # Evaluation results (final report results from the run_grouped_graph folder)
│
└── wandb/                          # Weights & Biases logs
```

### Key File Descriptions

**Core Training Files:**
- `custom_dpo.py`: Implements DPO loss computation, gradient steps, and fine-tuning loop with optional IPS weighting
- `dpo_dataset.py`: PyTorch Dataset class for loading preference pairs from JSONL files
- `run.py`: Main training script that orchestrates model loading, dataset preparation, and DPO fine-tuning (this is a basic initial run script but my local MPS was insufficient so not used)

**Data Processing:**
- `data_processing.py`: Creates DPO training datasets from the PRISM alignment dataset, implements IPS sampling based on user LLM usage frequency

**Evaluation:**
- `eval_ipw.py`: Evaluates models using two methods:
  - BERT similarity: Compares semantic similarity of model outputs to gold standard
  - LLM-as-judge: Uses GPT to score responses based on user's historical preferences
  - Both methods calculate overall accuracy and per-group breakdowns by user LLM usage frequency

**Notebooks:**
- `run.ipynb`: This is the full DPO training workflow, used to train the DPO models. This colab is modified for the different all, IPS, and noIPS configurations by updating the `default_config` variable's "ips" and "sample" parameters to determine if ips should be used or not (True/False) and if we should sample from the datsaet or not (True = ips/noips, False=all).
- `run_toy*.ipynb`: This is an adaptation of the `run.ipynb` file for just a single toy dataset. The different variation show the results from ips and noips for their respective training styles to store the outputs.
- `eval_test*.ipynb`: Notebook to generate model inference outputs. The outputs of running inference for each model (all/ips/noips) on the eval set are provided in the ipynb file outputs and the generated results are saved in `out/gen_result*_fixtemplate.jsonl`.

## Step-by-Step Guide to Reproduce Results

### Step 1: Data Preparation

The project uses the PRISM alignment dataset from HuggingFace. To prepare the data:

```bash
python data_processing.py
```

This script will:
1. Download the PRISM alignment dataset (`HannahRoseKirk/prism-alignment`)
2. Process user preferences into DPO preference pairs
3. Create a few datasets:
   - `data/all_dpo_dataset.jsonl`: All available preference pairs (and a reduced first 200 user filtered version)
   - `data/sampled_dpo_dataset.jsonl`: IPS-weighted sampled dataset based on user LLM usage frequency (and a reduced first 200 user filtered version)
   - `data/all_dpo_dataset_val_200.jsonl`: a sample of 200 rows from the unsampled dataset for validation

### Step 2: Training Models

For toy models, the following three colab notebooks contain the runs and output results for each experiment:
1. Baseline (no IPS, all data): `run_toy.ipynb`

2. With IPS weighting: `run_toy_ips.ipynb`

3. Without IPS weighting (sampled data): `run_toy_noips.ipynb`

For actual PRISM user data training, run the following colab: `run.ipynb`
- To run the different experiment variations, modify the default_config in the following ways:
  - all: `ips: False, sample: False`
  - ips: `ips: True, sample: True`
  - noips: `ips: False, sample: True`
- For each new run, it is recommended to rename the `run_name` so that it will save in a distinct directory and artifact path on wandb.
- Models are automatically saved to Weights & Biases every 5 epochs.

### Step 3: Generate Model Outputs

After training, generate responses from each model on the validation set. The following notebooks contain the results of funning inference for the three models on the PRISM dataset:
- `eval_test_all.ipynb`: Model trained on all data
- `eval_test_ips.ipynb`: Model trained with IPS
- `eval_test_noips.ipynb`: Model trained without IPS
Note that to reproduce these results with a new run, you will need to update the artifact path to point to the saved wandb artifact from a new training run. The artifact paths will depend on the run name and desired version number chosen from the wandb logged artifact results. I chose the version numbers based on the best performing validation loss value (IPW val for sampled IPS/no-IPS models, normal val for the all model)

These notebooks will:
1. Load the fine-tuned model from wandb artifacts (will need to be updated to point to the appropriate saved artifact on wandb)
2. Generate responses for validation prompts
3. Save outputs to `out/gen_result*.jsonl`

### Step 4: Evaluate Results

Run the evaluation script to compare model outputs using two complementary methods:

```bash
python eval_ipw.py
```

**Note:** The LLM-as-judge evaluation requires an OpenAI API key set in your environment:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

This script performs two types of evaluation:

#### 4a. BERT Similarity Evaluation
1. Loads generated outputs from all three models
2. Computes BERT semantic similarity scores between the "all data" model (reference standard) and the IPS/no-IPS models
3. Calculates accuracy: percentage of cases where IPS model output is more similar to the reference standard than no-IPS model
4. Saves results to `out/results/{run_name}/`:
   - `similarity_results.csv`: Detailed per-sample scores with user metadata
   - `similarity_result.txt`: Overall accuracy and average scores
   - `similarity_grouped_results.txt`: Breakdown by user LLM frequency groups
   - `similarity_grouped_chart.png`: Graphical output of the results by user LLM frequency groups

#### 4b. LLM-as-Judge Evaluation
1. Loads user's historical scores from PRISM utterances dataset
2. For each response, queries GPT-4o-mini to score both IPS and no-IPS outputs (1-100 scale)
3. GPT scoring is calibrated using the user's previous ratings for the same prompt
4. Calculates accuracy: percentage of cases where IPS model gets a higher score than no-IPS model
5. Saves results to `out/results/{run_name}/`:
   - `llmjudge_results.csv`: Detailed per-sample scores with user metadata
   - `llmjudge_result.txt`: Overall accuracy and average scores
   - `llmjudge_grouped_results.txt`: Breakdown by user LLM frequency groups
   - `llmjudge_grouped_chart.png`: Graphical output of the results by user LLM frequency groups

### Step 5: Analysis

Review the results in the output directory:
- Compare training curves in Weights & Biases dashboard
- Analyze detailed scores in `out/results/{run_name}/similarity_results.csv` and `llmjudge_results.csv`
- Examine overall metrics in `*_result.txt` files
- Review grouped breakdowns in `*_grouped_results.txt` to see how IPS weighting performs across different user segments
- Compare BERT similarity vs LLM-as-judge evaluations to understand model performance

## Expected Runtime and Computational Requirements

### Hardware Requirements

All training runs (run*.ipynb files) were done using a L4 GPU on Google Colab. Either L4 GPU or A100 will work as the models need at least 22GB RAM in order to complete the 5 training epochs.

### Expected Runtimes

**Training (all run-.ipynb files):**

Training times for the toy dataset experiments were pretty reasonable, within 15-30 minutes using the L4 GPUs. For the full PRISM dataset, running on the entire train set was unreasonable (tqdm quoted 20+ hours) so I ran on a filtered subset for the first 200 users and it took around 5-6 hrs for the full (all) model training and 3-4 hours for the sampled (IPS and no IPS) model training.

**Inference (all eval-.ipynb files):**
For inference, I used the T4 GPUs. Inference took around 30mins to an hour for each notebook.

**Evaluation (eval_ipw.py):**
- BERT Similarity: < 5 minutes for 200 samples (runs locally, no GPU needed)
- LLM-as-Judge: < 10 minutes for 200 samples (runs locally using OpenAI API, no GPU needed)

## Required Datasets and Data Sources

**PRISM Alignment Dataset**
- Source: HuggingFace Datasets Hub
- Dataset ID: `HannahRoseKirk/prism-alignment`
- Files used:
  - `utterances.jsonl`: User prompts, model responses, preference labels, and numerical scores (1-100)
    - Used for creating DPO training data and calibrating LLM-as-judge evaluations
  - `survey.jsonl`: User demographics including LLM usage frequency (`lm_frequency_use`)
    - Used for IPS sampling weights and grouped metric analysis
- URL: https://huggingface.co/datasets/HannahRoseKirk/prism-alignment

The dataset is automatically downloaded when running `data_processing.py` and `eval_ipw.py`.

The sampled datasets created for response bias simulation are also provided in the code within the data/ directory as described in the file structure above.