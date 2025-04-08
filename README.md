### Sentimatic: Emotion Shift Detection & Data Generation Pipeline from Customer Dialogues

This project is designed to analyze emotional shifts in customer support dialogues and automatically generate positive/negative text data for model training through a structured pipeline.

⸻

1. Create Conda Environment

First, create a conda environment using the provided sentimatic.yaml:

conda env create -f sentimatic.yaml
conda activate sentimatic



⸻

2. Detect Emotion Shift (sentiment_analysis.py)

Run the script to detect emotional changes between “before” and “after” customer messages.

python sentiment_analysis.py \
  --model_name gpt-4o \
  --use_openai \
  --partition train

Arguments:
	•	--model_name: Choose from gpt-4o, o3-mini, gemma3, llama3.2
	•	--use_openai or --no_openai: Whether to use OpenAI API or local model
	•	--partition: Select dataset partition from train, test, or valid

⸻

3. Merge Sentiment CSVs (concat.py)

Merge sentiment analysis results for a specific dataset split.

python concat.py --split train



⸻

4. Classify by Sentiment Score Difference (classify.py)

Classify samples based on the difference in sentiment scores:
	•	If the difference > 0 → positive
	•	If the difference < 0 → negative

python classify.py --split train



⸻

5. Train the Generation Model (train.py)

Train a model to generate positive/negative response pairs.

python train.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --config_path config/train_config.yaml \
  --data_dir data/ \
  --output_dir model/ \
  --train_type merged

Note: train_type can be positive, negative, or merged

⸻

6. Generate Data using Trained Model (infer.py)

Generate positive/negative response data using the trained model.

python infer.py \
  --model_dir model/ \
  --data_dir data/ \
  --output_dir output/ \
  --train_type merged \
  --data_type train \
  --gpu 0



⸻

7. Preprocess Generated Text (pre-processing.py)

Preprocess the generated text and finalize the PO dataset.

python pre-processing.py \
  --data_path output/generated_data.csv \
  --train_type merged \
  --data_type train



⸻

Pipeline Overview
	1.	Set up conda environment
	2.	Analyze emotional shift in dialogues
	3.	Merge sentiment results
	4.	Classify based on score difference
	5.	Train model to generate pairs
	6.	Generate new data using the model
	7.	Preprocess and finalize dataset

⸻

Let me know if you’d like to include example outputs, project structure, or usage tips!
