from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import torch
import pandas as pd
from tqdm import tqdm
import os
import argparse

def clean_output(output):
    # pipeline의 결과는 이미 디코딩된 문자열이므로 첫번째 항목의 'translation_text'에서 문자열을 추출합니다.
    return output[0]['translation_text'].strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="model", help="Path to fine-tuned model directory")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, help="Path to save results")
    parser.add_argument("--train_type", type=str, choices=["positive", "negative", "merged"], required=True,
                        help="Choose which dataset to use: 'positive', 'negative' or 'merged'")
    parser.add_argument("--data_type", type=str, choices=["train", "test", "valid"], required=True,
                        help="Choose which dataset to use: 'train', 'test' or 'valid'")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use (e.g., 0 for cuda:0)")
    args = parser.parse_args()
    
    device_index = args.gpu if torch.cuda.is_available() else -1
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # fine-tuned 모델 파라미터로부터 모델과 토크나이저 로드 (args.model_dir에 저장된 모델 사용)
    checkpoint = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
   
    # pipe = pipeline("text-generation", model=model, tokenizer=checkpoint, device=device_index)
    model = pipeline('text2text-generation', model = checkpoint, device=device_index)
    
    print("-" * 50)
    print(f"Using device: {device}")
    print("-" * 50)
    
    # 테스트 데이터셋 로드 (예: customer_01, agent 열 존재)
    test_file = os.path.join(args.data_dir, f"{args.data_type}", f"{args.train_type}_{args.data_type}.csv")
    df = pd.read_csv(test_file)[['customer_01', 'agent']]
    
    print("Processing test dataset using fine-tuned model for generation...")
    
    responses = []
    for idx, customer in tqdm(enumerate(df["customer_01"]), total=len(df["customer_01"])):
        # 프롬프트 문자열 구성
        input_text = f"Generate an agent's response to the following customer message: {customer}"
        # 파이프라인 호출 (기본 생성 전략 사용)
        result = model(input_text, max_length=512, do_sample=True)[0]['generated_text']
        print(result)
        responses.append(result)
    
    # 생성된 응답을 DataFrame에 저장하고 CSV로 내보내기
    df['response'] = responses
    df.to_csv(os.path.join(args.output_dir, f"{args.train_type}_{args.data_type}_infer_results.csv"), index=False)
    print("Inference completed. Results saved.")