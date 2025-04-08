import pandas as pd
from utils import concat
import random
from trl import ORPOConfig, ORPOTrainer
import nltk
import numpy as np
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,T5Config, AutoConfig
import os
import argparse

def preprocess_function(batch):
    
    inputs = [
        f"Generate an agent's response to the following customer message: {customer}"
        for customer in batch["customer_01"]
    ]
    pos = [
        f"agent: {response}"
        for response in batch["pos"]
    ]
    neg = [
        f"agent: {response}"
        for response in batch["neg"]
    ]
    
    return {
        "prompt": inputs,
        "chosen": pos,
        "rejected": neg
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/home/station_06/Sentimatic/dataset/Tweetsumm_orpo")
    cli_args = parser.parse_args()
    
    if not os.path.exists(cli_args.output_dir):
        os.makedirs(cli_args.output_dir, exist_ok=True)  # 디렉토리 생성 (필요 시)
        
        train_dataset = concat("train")
        train_dataset.to_csv(os.path.join(cli_args.output_dir, "train_orpo.csv"))

        valid_dataset = concat("valid")
        valid_dataset.to_csv(os.path.join(cli_args.output_dir, "valid_orpo.csv"))
    else:
        train_dataset = pd.read_csv(os.path.join(cli_args.output_dir, "train_orpo.csv"))
        valid_dataset = pd.read_csv(os.path.join(cli_args.output_dir, "valid_orpo.csv"))
    
    train_dataset = Dataset.from_pandas(train_dataset)
    valid_dataset = Dataset.from_pandas(valid_dataset)
    
    original_columns= train_dataset.column_names
    train_dataset = train_dataset.map(
        preprocess_function,
        remove_columns=original_columns,
        batched=True
    )
    valid_dataset = valid_dataset.map(
        preprocess_function,
        remove_columns=original_columns,
        batched=True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' 
    tokenizer.truncation_side = 'left' 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Global Parameters
    L_RATE = 5e-5
    BATCH_SIZE = 8
    PER_DEVICE_EVAL_BATCH = 4
    WEIGHT_DECAY = 0.01
    SAVE_TOTAL_LIM = 1
    NUM_EPOCHS = 10

    orpo_config = ORPOConfig(
        output_dir="/home/station_06/DATA01/sentimatic-outputs/t5-small-orpo",               
        per_device_train_batch_size=BATCH_SIZE,         
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=L_RATE, 
        beta = 0.1,                   
        lr_scheduler_type="cosine",            
        logging_steps=2,                       
        save_total_limit=10,                     
        evaluation_strategy="epoch",           
        push_to_hub=False,   
        metric_for_best_model="loss",
        save_strategy="epoch",                        
    )
    
    # Example usage in your training script
    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model("../results")

        
    
