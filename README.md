# Sentimatic: Emotion Shift Detection & Data Generation Pipeline from Customer Dialogues

This project is designed to analyze emotional shifts in customer support dialogues and automatically generate positive/negative text data for model training through a structured pipeline.

⸻

## 1. Create Conda Environment

First, create a conda environment using the provided sentimatic.yaml:

```python
conda env create -f sentimatic.yaml
conda activate sentimatic
```

⸻

## 2. Detect Emotion Shift (sentiment_analysis.py)

Run the script to detect emotional changes between “before” and “after” customer messages.

```python
python sentiment_analysis.py \
  --model_name gpt-4o \
  --use_openai \
  --partition train
```

Arguments:
	•	--model_name: Choose from gpt-4o, o3-mini, gemma3, llama3.2
	•	--use_openai or --no_openai: Whether to use OpenAI API or local model
	•	--partition: Select dataset partition from train, test, or valid

⸻

## 3. Merge Sentiment CSVs (concat.py)

Merge sentiment analysis results for a specific dataset split.

```python
python concat.py --split train
```
⸻

## 4. Classify by Sentiment Score Difference (classify.py)

Classify samples based on the difference in sentiment scores:
	•	If the difference > 0 → positive
	•	If the difference < 0 → negative

```python
python classify.py --split train
```

⸻

## 5. Train the Generation Model (train.py)

Train a model to generate positive/negative response pairs.

```python
python train.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --config_path config/train_config.yaml \
  --data_dir data/ \
  --output_dir model/ \
  --train_type merged
```

Note: train_type can be positive, negative, or merged

⸻

## 6. Generate Data using Trained Model (infer.py)

Generate positive/negative response data using the trained model.

```python
python infer.py \
  --model_dir model/ \
  --data_dir data/ \
  --output_dir output/ \
  --train_type merged \
  --data_type train \
  --gpu 0
```


⸻

## 7. Preprocess Generated Text (pre-processing.py)

Preprocess the generated text and finalize the PO dataset.

```python 
python pre-processing.py \
  --data_path output/generated_data.csv \
  --train_type merged \
  --data_type train
```
### ---------------------- For Reproducing Experiment -------------------------

Sure! Here’s an English README.md section that clearly explains your experiment comparing ORPO and SFT training methods using the LLM-as-a-Judge approach. It’s structured to be informative and easy to follow:

⸻

ORPO vs SFT: Evaluating PO-Trained Dialogue Models with LLM-as-a-Judge

This experiment evaluates two dialogue models trained using different PO (Preference Optimization) learning methods — ORPO and SFT — on their ability to handle real-world customer interactions.

We compare the models using an LLM-as-a-Judge approach across three key criteria:
	1.	Context Appropriateness – Is the response contextually relevant and coherent?
	2.	Problem Solving Approach – Does the response actively and effectively address the user’s issue?
	3.	Negative Emotion Management – How well does the response handle or defuse user frustration or dissatisfaction?

⸻

1. Train with ORPO (orpo.py)

Train a dialogue model using the ORPO (Offline Reinforcement Preference Optimization) method on the PO data generated from the sentimatic pipeline.
```python 
python orpo.py \
  --data_dir data/po/ \
  --output_dir model/orpo/
```
Uses generated positive/negative pairs from the Sentimatic pipeline.

⸻

2. Train with SFT (sft.py)

Train a dialogue model using Supervised Fine-Tuning (SFT) on publicly available seed datasets.
```python 
python sft.py \
  --data_dir data/sft_seed/ \
  --output_dir model/sft/
```
Uses open-domain SFT data as seed data for baseline comparison.

⸻

3. Evaluate with LLM-as-a-Judge (LLM-as-a-judge.py)

Run model evaluation using an LLM to judge model responses from ORPO and SFT.
```python 
python LLM-as-a-judge.py \
  --orpo_model_path model/orpo/ \
  --sft_model_path model/sft/ \
  --eval_data_path data/eval_prompts.json \
  --output_path results/judge_raw.json
```
The judge model scores responses on:
	•	Context Appropriateness
	•	Problem Solving Approach
	•	Negative Emotion Management

⸻

4. Aggregate Results (LLM-as-a-judge-result.py)

Summarize and aggregate the evaluation results to determine which training method performs better.
```python
python LLM-as-a-judge-result.py \
  --input_path results/judge_raw.json \
  --output_path results/summary.json
```
