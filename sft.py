import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, GenerationConfig
from datasets import Dataset
import pandas as pd
import numpy as np
import os
import argparse
import yaml
from evaluate import load as load_metric
from sklearn import metrics
from utils import string_to_float  # 필요시 사용

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # -100 값을 tokenizer.pad_token_id로 대체
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    
    # 예측과 라벨 디코딩
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]  # BLEU 계산을 위해 wrap
    
    print("Sample decoded prediction:", decoded_preds[0])
    print("Sample decoded label:", decoded_labels[0])
    
    # BLEU, ROUGE, METEOR 계산
    metric_bleu = load_metric('sacrebleu')
    metric_rouge = load_metric('rouge')
    metric_meteor = load_metric('meteor')
    
    bleu_result = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_result = metric_rouge.compute(predictions=decoded_preds, references=[ref[0] for ref in decoded_labels])
    meteor_result = metric_meteor.compute(predictions=decoded_preds, references=[ref[0] for ref in decoded_labels])
    
    rouges = {key: value for key, value in rouge_result.items()}
    
    return {
        "bleu": bleu_result['score'],
        'rouge1': rouges['rouge1'],
        'rouge2': rouges['rouge2'],
        'rougeL': rouges['rougeL'],
        "meteor": meteor_result['meteor'],
    }

def preprocess_function(batch, tokenizer):
    inputs = []
    targets = []
    for customer, agent in zip(batch["customer_01"], batch["agent"]):
        # 고객 메시지를 기반으로 프롬프트 구성
        input_text = f"Generate an agent's response to the following customer message: {customer}"
        inputs.append(input_text)
        targets.append(agent)
    
    # 입력과 타겟을 각각 토크나이즈 (padding, truncation 적용)
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google-t5/t5-small",
                        help="Pretrained model name or path (default: google-t5/t5-small)")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, help="Path to save model and outputs")
    parser.add_argument("--train_type", type=str, choices=["positive", "negative", "merged"], required=True,
                        help="Dataset type: 'positive', 'negative' or 'merged'")
    args = parser.parse_args()
    
    # Config 파일 로드 (학습 인자들이 포함되어 있어야 함)
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # 데이터셋 로드
    train_file = os.path.join(args.data_dir, 'train', f'{args.train_type}_train.csv')
    valid_file = os.path.join(args.data_dir, 'valid', f'{args.train_type}_valid.csv')
    
    train_df = pd.read_csv(train_file)[['customer_01', 'agent']]
    valid_df = pd.read_csv(valid_file)[['customer_01', 'agent']]
    
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    
    # 토크나이저와 모델 로드 (T5 모델)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # T5는 보통 eos_token을 pad_token으로 사용합니다.
    tokenizer.pad_token = tokenizer.eos_token
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print("-" * 50)
    print(f"Using {device} for training")
    print("-" * 50)
    
    # 생성 설정 업데이트 (예시로 max_length 등)
    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.update(max_length=400, min_length=4)
    config["train"]["generation_config"] = generation_config
    
    # 데이터 전처리 및 토큰화
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    train_tokenized_dataset = train_dataset.map(lambda batch: preprocess_function(batch, tokenizer), batched=True)
    valid_tokenized_dataset = valid_dataset.map(lambda batch: preprocess_function(batch, tokenizer), batched=True)
    
    # Seq2SeqTrainingArguments 생성 (config 파일 내 학습 인자 사용)
    training_args = Seq2SeqTrainingArguments(**config["train"], output_dir=args.output_dir)
    
    # Trainer 생성 및 학습 시작
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