import json
from collections import Counter
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import difflib
import json
from transformers import AutoTokenizer
import csv
import sys



# add line level signals
def find_lines_with_char_positions(func_code, fixed_func):
    func_code_lines = func_code.splitlines(keepends=True)  # Keep newline characters for correct indexing
    fixed_func_lines = fixed_func.splitlines(keepends=True)

    # Use Differ to find line-level differences
    differ = difflib.Differ()
    diff = list(differ.compare(func_code_lines, fixed_func_lines))
    func_code_index = 0
    lines = []
    for line in diff:
        if line.startswith("- "):  # Line deleted from original code
            deleted_line_text = line[2:]  # Strip the "- " prefix
            if deleted_line_text.strip():  # Ignore empty or whitespace-only deletions
                if func_code_index not in lines:
                    lines.append(func_code_index)
            func_code_index += 1  # Move to the next line in the original code
        elif line.startswith("  "):  # Line unchanged
            func_code_index += 1  # Move to the next line in the original code

    return lines


def load_primevul_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        json_objects = []
        for line in file:
            json_objects.append(json.loads(line))

        for i in range(len(json_objects)):
            current_obj = json_objects[i]

            if i % 2 == 0:
                if i + 1 < len(json_objects):
                    after_obj = json_objects[i + 1]
                    lines_ground_truth = find_lines_with_char_positions(
                        current_obj["func"],
                        after_obj["func"]
                    )

                    entry = {
                        "func": current_obj["func"],
                        "lines_ground_truth": lines_ground_truth,
                        'idx': current_obj['idx'],
                        "target": current_obj["target"]
                    }
                    data.append(entry)

            else:
                if i - 1 >= 0:
                    before_obj = json_objects[i - 1]
                    lines_ground_truth = find_lines_with_char_positions(
                        current_obj["func"],
                        before_obj["func"]
                    )

                    entry = {
                        "func": current_obj["func"],
                        "lines_ground_truth": lines_ground_truth,
                        'idx': current_obj['idx'],
                        "target": current_obj["target"]
                    }
                    data.append(entry)

    return data


# 메인 처리 함수
def process_primevul_files():
    jsonl_files = ["primevul_test_paired.jsonl", "primevul_train_paired.jsonl",
                   "primevul_valid_paired.jsonl"]
    output_files = ["primevul_test_paired_gt.jsonl", "primevul_train_paired_gt.jsonl", "primevul_valid_paired_gt.jsonl"]

    for file_name, output_name in zip(jsonl_files, output_files):
        print(f"Processing {file_name}...")
        data = load_primevul_data(file_name)

        processed_data = []

        for entry in data:
            processed_data.append(entry)

        with open(output_name, 'w', encoding='utf-8') as output_file:
            for entry in processed_data:
                output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"Saved {len(processed_data)} entries to {output_name}")

    print("All files processed successfully!")

process_primevul_files()