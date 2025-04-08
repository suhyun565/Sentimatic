import os
import json
from tqdm import tqdm 
from openai import OpenAI
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import time
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
import ollama
import time
load_dotenv()

class SentimentAnalysis:
    def __init__(self, model_name, device='cpu', api_model=False):
        self.model_name = model_name
        self.device = device
        self.api_model = api_model
        
        if api_model:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.model = model_name
        else:
            self.model = model_name
                
                
    def prompt(self,c1, agent, c2):
        prompt = f'''
    c1 (Before agent's response): {c1}
    Agent's response: {agent}
    c2 (After agent's response): {c2}

    Please analyze the emotions in the conversation.

    Calculate the change in emotion using the formula: (c2's emotional score - c1's emotional score).
    Respond with a single float number only, within the range of -2 to 2.
    Do not include any explanation or additional text.
    '''
        return prompt

    
    def inference(self,prompt):
        
        time.sleep(np.random.uniform(0, 3))
        
        if self.api_model:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        
        else:
            response = ollama.chat(model=self.model, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
            ])
            return response.message.content

    def run(self, data_dir, csv_file, save_dir, temp_file, final_file):
        
        df = pd.read_csv(os.path.join(data_dir, csv_file))
        df = df[['customer_01', 'agent', 'customer_02']]

        print("--------------------------------------------------------------")
        print(f"Total number of data: {len(df)}")
        print("--------------------------------------------------------------")

        for idx, key in tqdm(df.iterrows()):
            prompt = self.prompt(key['customer_01'], key['agent'], key['customer_02'])
            answer = self.inference(prompt)

            df.loc[idx, 'c2 - c1'] = answer
                
            if idx % 100 == 0 and idx != 0:
                df.to_csv(os.path.join(save_dir, temp_file), index=False)
                print(f'Index: {idx}, csv saved!')
            
        df.to_csv(os.path.join(save_dir, final_file), index=False)
        print('csv saved!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='gpt-4o or o3-mini or gemma3 or llama3.2')
    parser.add_argument('--use_openai', action='store_true', help='Use OpenAI API instead of local model')
    parser.add_argument('--no_openai', dest='use_openai', action='store_false', help='Do not use OpenAI API')
    parser.add_argument('--partition', type=str, default='train', choices=['train', 'test', 'valid'], help='Dataset partition (train, test, valid)')
    args = parser.parse_args()
    
    print("--------------------------------------------------------------")
    print(f"Model name: {args.model_name}")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    print(f"Use openai or not: {args.use_openai}")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    print(f"{args.partition} is processing now")
    print("--------------------------------------------------------------")
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_dir = f"/home/station_06/Sentimatic/dataset/Tweetsumm single turn"
    csv_file = f"{args.partition}.csv"
    save_dir = f"/home/station_06/Sentimatic/dataset/Tweetsumm sentiment analysis/{args.partition}/{args.model_name}"
    os.makedirs(save_dir, exist_ok=True)
    temp_file = f"{args.partition}_SentimentLabel_temp.csv"
    final_file = f"{args.partition}_SentimentLabel.csv"

    model = SentimentAnalysis(model_name=args.model_name, device=device, api_model=args.use_openai)
    model.run(data_dir, csv_file, save_dir, temp_file, final_file)
