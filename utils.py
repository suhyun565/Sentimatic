import re
import pandas as pd 
from datasets import Dataset

def string_to_float(x):
    float_pattern = r"[-+]?\d*\.\d+|\d+"
    match = re.findall(float_pattern, x)
    return float(match[0]) if match else 0.0

def clean_text(text):
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    # remove urls
    text = re.sub(r"http\S+", "", text)
    return text

def concat(data_type):
    # ----- NEGATIVE 데이터 -----
    neg_pos = pd.read_csv(f'/home/station_06/DATA01/sentimatic-outputs/inference/pos/llama3.2-2288/negative_{data_type}_infer_results_cleaned.csv')
    neg_pos = neg_pos.rename(columns={
        "beam_res": "beam_res_pos",
        "top_k_res": "top_k_res_pos",
        "top_p_res": "top_p_res_pos"
    })
    neg_neg = pd.read_csv(f'/home/station_06/Sentimatic/dataset/Tweetsumm_sentiment_analysis/{data_type}/negative_{data_type}.csv')

    pos_values = pd.concat([
        neg_pos['beam_res_pos'],
        neg_pos['top_k_res_pos'],
        neg_pos['top_p_res_pos']
    ], ignore_index=True)
    
    # neg_pos 기준에서 불필요한 열 제거하고
    exclude_cols = ['beam_res_pos', 'top_k_res_pos', 'top_p_res_pos', 'beam_score', 'top_k_score', 'top_p_score']

    # neg_pos와 neg_neg에 공통으로 존재하는 열만 copy_columns로 선택
    copy_columns = [col for col in neg_pos.columns if col not in exclude_cols and col in neg_neg.columns]

    base_info = neg_neg[copy_columns].loc[neg_pos.index.repeat(3)].reset_index(drop=True)

    neg = base_info.copy()
    neg['pos'] = pos_values
    neg = neg.rename(columns={"agent": "neg"})

    # ----- POSITIVE 데이터 -----
    pos_neg = pd.read_csv(f'/home/station_06/DATA01/sentimatic-outputs/inference/neg/llama3.2-1425/positive_{data_type}_infer_results_cleaned.csv')
    pos_neg = pos_neg.rename(columns={
        "beam_res": "beam_res_neg",
        "top_k_res": "top_k_res_neg",
        "top_p_res": "top_p_res_neg"
    })
    pos_pos = pd.read_csv(f'/home/station_06/Sentimatic/dataset/Tweetsumm_sentiment_analysis/{data_type}/positive_{data_type}.csv')

    neg_values = pd.concat([
        pos_neg['beam_res_neg'],
        pos_neg['top_k_res_neg'],
        pos_neg['top_p_res_neg']
    ], ignore_index=True)
    
    exclude_cols = ['beam_res_neg', 'top_k_res_neg', 'top_p_res_neg', 'beam_score', 'top_k_score', 'top_p_score']
    copy_columns = [col for col in pos_neg.columns if col not in exclude_cols and col in pos_pos.columns]
    
    base_info = pos_pos[copy_columns].loc[pos_pos.index.repeat(3)].reset_index(drop=True)

    pos = base_info.copy()
    pos['neg'] = neg_values
    pos = pos.rename(columns={"agent": "pos"})

    # ----- CONCAT -----
    combined_df = pd.concat([neg, pos], axis=0).reset_index(drop=True)
    combined_dataset = Dataset.from_pandas(combined_df)

    return combined_dataset