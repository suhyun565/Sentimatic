from transformers import T5ForConditionalGeneration, AutoTokenizer, GenerationConfig, AutoModelForCausalLM
import torch
from datasets import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from evaluate import load
from tqdm import tqdm
import os
import argparse
import yaml

def clean_output(output):
    return tokenizer.decode(output[0], skip_special_tokens=True).split("assistant\n\n")[-1].split("https:")[0]

def compute_bertscore(predictions, references, lang="en"):
    results = bertscore.compute(predictions=predictions, references=references, lang=lang)
    return results["f1"]

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_dir", type=str, default="model")
    args.add_argument("--data_dir", type=str, help="path to data dir")
    args.add_argument("--output_dir", type=str, help="path to save model")
    args.add_argument("--train_type", type=str, choices=["positive", "negative", "merged"], required=True,
                  help="Choose which dataset to use: 'positive' or 'negative' or 'merged'")
    args.add_argument("--data_type", type=str, choices=["train", "test", "valid"], required=True,
                  help="Choose which dataset to use: 'train' or 'test' or 'valid'")
    args.add_argument("--gpu", type=int, default=0, help="GPU index to use (e.g., 0 for cuda:0)")


    args = args.parse_args()
    
    bertscore = load("bertscore")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    checkpoint = os.path.join(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model = model.to(device)
    model.eval()
    
    generation_config = GenerationConfig.from_model_config(model.config)
    
    print("-" * 50)
    print(f"Using {device} for inference")
    print("-" * 50)
    
    # Load test dataset
    test_file = os.path.join(args.data_dir, f'{args.data_type}',f'{args.train_type}_{args.data_type}.csv')
    df = pd.read_csv(test_file)[['customer_01', 'agent']]
    
    print("Processing test dataset with various sampling strategies...")
    
    responses = []
    num_return_sequences = 3

    beam_outputs = []
    top_k_outputs = []
    top_p_outputs = []

    for idx, customer in tqdm(enumerate(df["customer_01"]), total=len(df["customer_01"])):
        messages = [
            {"role": "system", "content": "You are a customer service chatbot. Generate a agent's response to the following customer message."},
            {"role": "user", "content": customer},
        ]
        input_prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        
        padded_inputs = torch.nn.utils.rnn.pad_sequence(input_prompt, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = (padded_inputs != tokenizer.pad_token_id).long()
        
        model_inputs = {
            "input_ids": padded_inputs.to(device),
            "attention_mask": attention_mask.to(device),
        }
        
        for i in range(3):
            with torch.no_grad():
                if i == 0:
                    output0 = model.generate(model_inputs["input_ids"], attention_mask=model_inputs["attention_mask"], num_beams=num_return_sequences, num_return_sequences=num_return_sequences, early_stopping=True,
                                            max_length=330, temperature=0.9)
                if i == 1:
                    output1 = model.generate(model_inputs["input_ids"], attention_mask=model_inputs["attention_mask"], top_k=50, do_sample=True, num_return_sequences=num_return_sequences, early_stopping=True,
                                            max_length=330, temperature=0.9)
                elif i == 2:
                    output2 = model.generate(model_inputs["input_ids"], attention_mask=model_inputs["attention_mask"], top_p=0.9, do_sample=True, num_return_sequences=num_return_sequences, early_stopping=True,
                                            max_length=330, temperature=0.9)
        
        beam_outputs.append(clean_output(output0))
        top_k_outputs.append(clean_output(output1))
        top_p_outputs.append(clean_output(output2))
        
        # print("="*50)
        # print(f"Customer: {customer}")
        # print(f"Agent (beam search): {beam_outputs[-1]}")
        # print(f"Agent (top-k sampling): {top_k_outputs[-1]}")
        # print(f"Agent (top-p sampling): {top_p_outputs[-1]}")
        # print("="*50)

    beam_score = compute_bertscore(beam_outputs, df["agent"].tolist())
    top_k_score = compute_bertscore(top_k_outputs, df["agent"].tolist())
    top_p_score = compute_bertscore(top_p_outputs, df["agent"].tolist())

    df['beam_res'] = beam_outputs
    df['top_k_res'] = top_k_outputs
    df['top_p_res'] = top_p_outputs

    df['beam_score'] = beam_score
    df['top_k_score'] = top_k_score
    df['top_p_score'] = top_p_score

    df.to_csv(os.path.join(args.output_dir, f"{args.train_type}_{args.data_type}_infer_results.csv"), index=False)
    print(f"Beam search bertscore: {beam_score}")
    print(f"Top-k sampling bertscore: {top_k_score}")
    print(f"Top-p sampling bertscore: {top_p_score}")
