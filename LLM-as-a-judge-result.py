import pandas as pd
import json
import re

def parse_judge_result(text):
    """
    judge_result 문자열에서 불필요한 마크다운 문법(예: ```json)을 제거한 후, JSON 객체로 변환합니다.
    """
    text = text.strip()
    # 백틱과 json 태그 제거
    text = re.sub(r'^```json', '', text)
    text = re.sub(r'```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print("JSON decode error:", e, "in text:", text)
        return {}

def compute_percentages(df, model_name):
    """
    주어진 데이터 프레임(df)의 judge_result 칼럼에서 각 평가 기준별로 'A wins' 비율을 계산합니다.
    """
    categories = ["Context appropriateness", "Problem solving approach", "Negative emotion management"]
    counts = {cat: {"B wins": 0, "total": 0} for cat in categories}
    
    for res in df["judge_result"]:
        parsed = parse_judge_result(res)
        for cat in categories:
            if cat in parsed:
                counts[cat]["total"] += 1
                if parsed[cat].strip().lower() == "b wins":
                    counts[cat]["B wins"] += 1

    percentages = {}
    print(f"Model: {model_name}")
    for cat in categories:
        if counts[cat]["total"] > 0:
            pct = counts[cat]["B wins"] / counts[cat]["total"] * 100
        else:
            pct = 0.0
        percentages[cat] = pct
        print(f"  {cat}: B wins = {pct:.2f}%")
    print()
    return percentages

# 예시: 세 모델의 CSV 파일을 각각 로드 (각 CSV에는 'customer_01'와 'judge_result' 칼럼이 있다고 가정)
df_gpt4 = pd.read_csv("/home/station_06/DATA01/sentimatic-outputs/llm-judge/judge_results_gpt-4o.csv")      # 예: GPT-4o 예측 결과 CSV
df_gpto3 = pd.read_csv("/home/station_06/DATA01/sentimatic-outputs/llm-judge/judge_results_o3-mini.csv")      # 예: GPT-o3 예측 결과 CSV
df_gpt35 = pd.read_csv("/home/station_06/DATA01/sentimatic-outputs/llm-judge/judge_results_gpt-3.5-turbo.csv")      # 예: GPT-3.5 예측 결과 CSV
df_gptchat = pd.read_csv("/home/station_06/DATA01/sentimatic-outputs/llm-judge/judge_results_chatgpt-4o-latest.csv")  

# 각 모델별로 'A wins' 비율 계산
compute_percentages(df_gpt4, "GPT-4o")
compute_percentages(df_gpto3, "GPT-o3")
compute_percentages(df_gpt35, "GPT-3.5")
compute_percentages(df_gptchat, "GPT-chat")
