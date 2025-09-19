import json
import random
import warnings
from datetime import datetime

import openai
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-nine")
warnings.filterwarnings("ignore")
client = openai.OpenAI(api_key = '-')
input_file = 'unix_1024_localizer/test.jsonl'


def ask_for_detection(data):
    def ask_gpt(data):
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict security vulnerability detection AI. Your task is to analyze known defective code and decide whether the defect is a security vulnerability.\n\nOnly respond with '1' if the defect is about - buffer overflow, memory leak, crash and corruption, etc..\n\nIf it’s **not** a security vulnerability (ex: general bugs, logic errors, or anything that lacks clear security impact), respond with '0'. "
                },
                {
                    "role": "user",
                    "content": f"The following code contains a defect. If it is about a security vulnerability, respond with '1'. If not, respond with '0'. Output a single digit only.\n\nCode:\n```{data}``` \n\n Some lines may begin by special token <!>, indicating lines to look more carefully. Judge the code's vulnerability utilizing this information."
                }
            ],
            stream=True,)
        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        return response.strip()
    return ask_gpt(data)


def save_prediction(input_file, a):
    TP = TN = FP = FN = 0
    TP2 = TN2 = FP2 = FN2 = 0

    # Open the input JSON file and output JSON file in append mode 0 267개, 1 233개
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        random.seed(123456)
        random.shuffle(lines)
        for ind,line in enumerate(lines):
            entry = json.loads(line)
            data = entry.get("func", "")
            lines = entry.get("lines", "")

            tokens=tokenizer.tokenize(data)[:2000]
            ids = tokenizer.convert_tokens_to_ids(tokens)
            truncated_data = tokenizer.decode(ids, skip_special_tokens=True)
            emphasized = []
            for i, line in enumerate(truncated_data.splitlines()):
                if i in lines:
                    emphasized.append(f"{a}{line}")
                else:
                    emphasized.append(line)
            emphasized_data= "\n".join(emphasized)
            label = entry.get("target", 0)   # invert label if needed
            gpt_response_with_lines = ask_for_detection(emphasized_data)
            gpt_response_without_lines = ask_for_detection(data)
            print(f"{ind} Predicted with lines: {gpt_response_with_lines}, without: {gpt_response_without_lines}, Label: {label}, ", end='')

            if gpt_response_with_lines == '1':
                if label == 1:
                    TP += 1
                else:
                    FP += 1
            elif gpt_response_with_lines == '0':
                if label == 0:
                    TN += 1
                else:
                    FN += 1
            if gpt_response_without_lines == '1':
                if label == 1:
                    TP2 += 1
                else:
                    FP2 += 1
            elif gpt_response_without_lines == '0':
                if label == 0:
                    TN2 += 1
                else:
                    FN2 += 1

            precision = TP / (TP + FP + 1e-10)
            recall = TP / (TP + FN + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)

            precision2 = TP2 / (TP2 + FP2 + 1e-10)
            recall2 = TP2 / (TP2 + FN2 + 1e-10)
            f1_2 = 2 * precision2 * recall2 / (precision2 + recall2 + 1e-10)
            accuracy2 = (TP2 + TN2) / (TP2 + TN2 + FP2 + FN2 + 1e-10)

            print(f"Accuracy with lines:  {accuracy:.4f}, without: {accuracy2:.4f};;;  {TP+FP}/{TN+FN} : {TP2+FP2}/{TN2+FN2} ;;; {f1} : {f1_2}")

        print("With line info:")
        print("TP,TN,FP,FN",TP,TN,FP,FN)
        print(f"\nPrecision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Accuracy:  {accuracy:.4f}\n")

        print("Without line info:")
        print("TP,TN,FP,FN",TP2,TN2,FP2,FN2)
        print(f"\nPrecision: {precision2:.4f}")
        print(f"Recall:    {recall2:.4f}")
        print(f"F1 Score:  {f1_2:.4f}")
        print(f"Accuracy:  {accuracy2:.4f}")

        with open("gpt_result.txt", "a") as f:
            f.write(f"-------------------------{datetime.now()}------------------------- \n")
            f.write("With line info:\n")
            f.write(f"TP,TN,FP,FN {TP},{TN},{FP},{FN}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1 Score:  {f1:.4f}\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n\n")

            f.write("Without line info:\n")
            f.write(f"TP,TN,FP,FN {TP2},{TN2},{FP2},{FN2}\n")
            f.write(f"Precision: {precision2:.4f}\n")
            f.write(f"Recall:    {recall2:.4f}\n")
            f.write(f"F1 Score:  {f1_2:.4f}\n")
            f.write(f"Accuracy:  {accuracy2:.4f}\n")
            f.write("=" * 50 + "\n")


save_prediction(input_file, "<!>")
save_prediction(input_file, "<!>")
save_prediction(input_file, "<!>")
