import pandas as pd
import argparse
import re

def remove_redundant_sentences(text):
    if pd.isnull(text):
        return text
    
    sentences = re.split(r'(?<=[.?!])\s+', text)
    seen = set()
    result = []
    
    for i, sentence in enumerate(sentences):
        sentence_strip = sentence.strip()
        if sentence_strip in seen:
            continue
        if result:
            prev = result[-1]
            overlap_len = 10
            for j in range(overlap_len, min(len(prev), len(sentence_strip))):
                if prev[-j:] == sentence_strip[:j]:
                    if sentence_strip in prev:
                        break
                    if prev in sentence_strip:
                        result[-1] = sentence_strip
                        break
            else:
                result.append(sentence_strip)
        else:
            result.append(sentence_strip)
        seen.add(sentence_strip)

    return ' '.join(result)

def pre_process(dataframe, col):
    def clean_text(text):
        if pd.isnull(text):
            return text

        # 문장 분리
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Thank you로 시작하는 문장 여러 개 -> 하나만 남김
        thank_you_sentences = [s for s in sentences if re.match(r"(?i)^thank you", s.strip())]
        if len(thank_you_sentences) > 1:
            sentences = [s for s in sentences if not re.match(r"(?i)^thank you", s.strip())]
            sentences.append(thank_you_sentences[0])
        
        # Please로 시작하는 문장 여러 개 -> 하나만 남김
        please_sentences = [s for s in sentences if re.match(r"(?i)^please", s.strip())]
        if len(please_sentences) > 1:
            sentences = [s for s in sentences if not re.match(r"(?i)^please", s.strip())]
            sentences.append(please_sentences[0])
        
        # 문장 다시 합치기
        text = ' '.join(sentences)

        # ^, / 제거
        text = re.sub(r"[\^/]", " ", text)

        # 분모/분자 숫자 제거 (예: 2/2)
        text = re.sub(r"\b\d+/\d+\b", "", text)

        # 괄호 제거
        text = re.sub(r"[()]", "", text)
        
        #url 제거
        text = re.sub(r"http\S+", "", text)
        
        # - 제거
        text = re.sub(r"-", " ", text)

        # 영어+숫자 혼합된 이상한 단어 제거 (@숫자 제외)
        text = re.sub(r'(?<!@)\b(?=[A-Za-z]*\d)[A-Za-z0-9]+\b', "", text)

        # 개행 제거
        text = re.sub(r"\n", " ", text)

        # 중복 공백 제거
        text = re.sub(r"\s+", " ", text).strip()

        # 문장 중복 제거
        text = remove_redundant_sentences(text)

        return text

    dataframe[col] = dataframe[col].astype(str).apply(clean_text)
    return dataframe

if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="path to data file")
    parser.add_argument("--train_type", type=str, choices=["positive", "negative","merged"], required=True,
                  help="Choose which dataset to use: 'positive' or 'negative' or 'merged'")
    parser.add_argument("--data_type", type=str, choices=["train", "test", "valid"], required=True,
                  help="Choose which dataset to use: 'train' or 'test' or 'valid'")
    args = parser.parse_args()
    df = pd.read_csv(f"{args.data_path}/{args.train_type}_{args.data_type}_infer_results.csv")
    
    # cols_to_clean = ['beam_res', 'top_k_res', 'top_p_res']
    cols_to_clean = ['response']
    df[cols_to_clean] = df[cols_to_clean].apply(lambda col: pre_process(df, col.name)[col.name])
    df.to_csv(f"{args.data_path}/{args.train_type}_{args.data_type}_infer_results_cleaned.csv",index=False)
    