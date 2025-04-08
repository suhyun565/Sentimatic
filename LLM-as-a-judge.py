import os
import argparse
import pandas as pd
from openai import OpenAI
import json
from tqdm import tqdm
from dotenv import load_dotenv
import time
import random
import numpy as np
import ollama

# 반드시 자신의 OpenAI API 키를 설정하세요.
# openai.api_key = "YOUR_OPENAI_API_KEY"

load_dotenv()
client = OpenAI()

def judge_pair(customer, response_A, response_B, judge_model="gpt-4o"):
    
    prompt = f"""Customer message: {customer}

Below are two responses for the same customer message:
Response A: {response_A}
Response B: {response_B}

Evaluate these responses based on the following criteria:
1. 맥락 적절성 (Context appropriateness)
2. 문제 해결 접근법 (Problem solving approach)
3. 부정적 감정관리 (Negative emotion management)

Decide which response is better. If one response is clearly superior, answer "A wins" or "B wins". 
If they are equally good, answer "Draw".

Provide your answer in JSON format like:
{{"Context appropriateness": "A Win", "Problem solving approach": "Draw", "Negative emotion management": "B wins"}}
no other explanation is needed.
"""
    try:
        if judge_model == "llama3":
            response = ollama.chat(model='llava', messages=[
            {
                'role': 'user',
                'content': prompt,
            },
            ])
            result_text = response.message.content
        else:
            completion = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert judge for evaluating customer service responses."},
                    {"role": "user", "content": prompt}
                ],
            )
            result_text = completion.choices[0].message.content
            # result_json = json.loads(result_text)
    except Exception as e:
        print("Error in judge_pair:", e)
        result_json = {"result": "Error"}
        result_text = "Error"
    return result_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df1_csv", type=str, required=True, help="Path to first CSV file (with 'customer_01' and 'response' columns)")
    parser.add_argument("--df2_csv", type=str, required=True, help="Path to second CSV file (with 'customer_01' and 'response' columns)")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output csv with judge results")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Judge model name (e.g., gpt-4o or o3-mini)")
    args = parser.parse_args()
    
    # 두 CSV 파일 로드 (각 파일에는 'customer_01'와 'response' 칼럼이 있음)
    df1 = pd.read_csv(args.df1_csv)
    df2 = pd.read_csv(args.df2_csv)
    
    # customer_01 열을 기준으로 두 DataFrame을 병합합니다.
    # 양쪽의 customer_01 데이터가 동일하다고 가정합니다.
    merged_df = pd.merge(df1[['customer_01', 'response']], df2[['customer_01', 'response']], 
                         on="customer_01", suffixes=("_A", "_B"))
    
    judge_results = []
    for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        customer = row["customer_01"]
        response_A = row["response_A"]
        response_B = row["response_B"]
        judgment = judge_pair(customer, response_A, response_B, judge_model=args.judge_model)
        judge_results.append(judgment)
        # print(f"Judgment for pair {idx+1}:", judgment)
        time.sleep(random.random() + random.random() * 2)
    
    csv_path = os.path.join(args.output_dir, f"judge_results_{args.judge_model}.csv")
    merged_df["judge_result"] = judge_results
    merged_df.to_csv(csv_path, index=False)
    print("Judgment completed. Results saved to", csv_path)