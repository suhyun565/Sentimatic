import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, GenerationConfig
from datasets import Dataset
import pandas as pd
import numpy as np
import re
import os
import argparse
import yaml
import nltk
from sklearn import metrics
from evaluate import load as load_metric
from utils import string_to_float


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Replace -100 in labels (ignored index) with tokenizer.pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Prepare data for BLEU and other metrics
    # BLEU expects a list of strings for predictions and a list of lists of strings for references
    decoded_preds = [pred.strip() for pred in decoded_preds]  # Clean up predictions
    decoded_labels = [[label.strip()] for label in decoded_labels]  # Wrap labels in lists for BLEU

    print("Sample decoded prediction:", decoded_preds[0])
    print("Sample decoded label:", decoded_labels[0])

    # Compute BLEU, ROUGE, and METEOR metrics
    metric_bleu = load_metric('sacrebleu')
    metric_rouge = load_metric('rouge')
    metric_meteor = load_metric('meteor')

    bleu_result = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_result = metric_rouge.compute(predictions=decoded_preds, references=[ref[0] for ref in decoded_labels])
    meteor_result = metric_meteor.compute(predictions=decoded_preds, references=[ref[0] for ref in decoded_labels])

    rouges = {key: value for key, value in rouge_result.items()}

    return {
        "bleu": bleu_result['score'],  # Use 'bleu' key from metric output
        'rouge1': rouges['rouge1'],
        'rouge2': rouges['rouge2'],
        'rougeL': rouges['rougeL'],
        "meteor": meteor_result['meteor'],
    }

def preprocess_function(batch, tokenizer):
    input_ids_list = []
    labels_list = []

    for customer, agent in zip(batch["customer_01"], batch["agent"]):
        # Step 1: make prompt (system + user)
        prompt_messages = [
            {"role": "system", "content": "You are a customer service chatbot. Generate a agent's response to the following customer message."},
            {"role": "user", "content": customer}
        ]
        input_prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).squeeze(0)

        # Step 2: agent tokenization
        agent_tokens = tokenizer(agent, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)

        # Step 3: full prompt
        full_prompt = torch.cat([input_prompt, agent_tokens], dim=0)

        # Step 4: labels
        labels = full_prompt.clone()
        labels[:len(input_prompt)] = -100

        input_ids_list.append(full_prompt)
        labels_list.append(labels)

    # Padding and attention masks
    padded_inputs = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    attention_mask = (padded_inputs != tokenizer.pad_token_id).long()

    model_inputs = {
        "input_ids": padded_inputs,
        "labels": padded_labels,
        "attention_mask": attention_mask
    }
    return model_inputs
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    args.add_argument("--config_path", type=str, help="path to config file")
    args.add_argument("--data_dir", type=str, help="path to data dir")
    args.add_argument("--output_dir", type=str, help="path to save model")
    args.add_argument("--train_type", type=str, choices=["positive", "negative","merged"], required=True,
                  help="Choose which dataset to use: 'positive' or 'negative' or 'merged'")
    
    args = args.parse_args()
    
    # Load config file
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Load datasets
    train_file = os.path.join(args.data_dir, 'train', f'{args.train_type}_train.csv')
    valid_file = os.path.join(args.data_dir, 'valid', f'{args.train_type}_valid.csv')
    
    train_df = pd.read_csv(train_file)[['customer_01', 'agent']]
    valid_df = pd.read_csv(valid_file)[['customer_01', 'agent']]
    
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("-" * 50)
    print(f"Using {device} for training")
    print("-" * 50)
    
    generation_config = GenerationConfig.from_model_config(model.config)
    new_config = dict()
    new_config['max_length'] = 400
    new_config['min_length'] = 4
    generation_config.update(**new_config)
    config["train"]["generation_config"] = generation_config
    
    model = model.to(device)
    
    # Dataset preparation
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    train_tokenized_dataset = train_dataset.map(lambda batch: preprocess_function(batch, tokenizer), batched=True)
    valid_tokenized_dataset = valid_dataset.map(lambda batch: preprocess_function(batch, tokenizer), batched=True)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(**config["train"], output_dir=args.output_dir)
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=valid_tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    
