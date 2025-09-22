import json
from collections import Counter
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import difflib

from collections import Counter

import json
from transformers import AutoTokenizer

# Filter out none vulnerable functions
with open("MSR_data_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)
print(len(data))
with open("MSR_filtering1.jsonl", "w", encoding="utf-8") as out_file:
    for entry in data.values():
        if "CWE ID" in entry and entry.get("vul") == "1":
            filtered_entry = {
                "func_before": entry.get("func_before", ""),
                "func_after": entry.get("func_after", ""),
                "CWE ID": entry["CWE ID"]
            }
            out_file.write(json.dumps(filtered_entry) + "\n")


# Filter and save entries with CWE ID count >= 20
data = []
with open("MSR_filtering1.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))
print(len(data))

vuln_counts = Counter(entry["CWE ID"] for entry in data)
filtered_data = [entry for entry in data if vuln_counts[entry["CWE ID"]] >= 20]
cnt2=0
with open("MSR_filtering2.jsonl", "w", encoding="utf-8") as f:
    for entry in filtered_data:
        cnt2+=1
        f.write(json.dumps(entry) + "\n")
print(cnt2)

# Remove cases where func before and after is same
data = []
with open("MSR_filtering2.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# Count CWE ID occurrences
vuln_counts = Counter(entry["CWE ID"] for entry in data)

# Filter: CWE ID count >= 20 and func_before ≠ func_after
filtered_data = [
    entry for entry in data
    if entry.get("func_before") != entry.get("func_after")
]

cnt3=0
# Save filtered data
with open("MSR_filtering3.jsonl", "w", encoding="utf-8") as f:
    for entry in filtered_data:
        cnt3+=1
        f.write(json.dumps(entry) + "\n")


print(cnt3)


# add line level signals
def find_lines_with_char_positions(func_code, fixed_func):
    func_code_lines = func_code.splitlines(keepends=True)  # Keep newline characters for correct indexing
    fixed_func_lines = fixed_func.splitlines(keepends=True)

    # Use Differ to find line-level differences
    differ = difflib.Differ()
    diff = list(differ.compare(func_code_lines, fixed_func_lines))
    func_code_index = 0
    lines=[]
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


data = []
with open("MSR_filtering3.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

temp_list=[]
vuln_counts = Counter(entry["CWE ID"] for entry in data)
sorted_vuln_counts = sorted(vuln_counts.items(), key=lambda x: x[1], reverse=True)
print("Vulnerability Classification Statistics (Sorted, Only CWE ID > 20):")
for key, count in sorted_vuln_counts:
    print(f"{key}: {count}")
    temp_list.append(key)

for i in temp_list:
    print(f"'{i}', ", end='')
print("CWE classes:", len(temp_list))


filtered_data = []
filtered_count = 0
for entry in data:
    f1=entry["func_before"]
    f2=entry["func_after"]
    lines_ground_truth=find_lines_with_char_positions(f1,f2)
    entry["lines_ground_truth"] = lines_ground_truth
    if entry["CWE ID"]=="":
        entry["CWE ID"]="others"
    if len(lines_ground_truth):
        filtered_data.append(entry)

# Print statistics
print(f"Total entries: {len(data)}")

# Save the filtered data back to JSONL
with open("MSR_filtering4.jsonl", "w") as f:
    for entry in filtered_data:
        json.dump(entry, f)
        f.write("\n")


# final cross folding
def load_data(file_path):
    # Read the data from a JSONL file
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            js = json.loads(line.strip())
            data.append(js)
    return data


def create_stratified_splits(file_path, k=5):
    # Load data and get the labels for stratification
    data = load_data(file_path)
    labels = [entry["CWE ID"] for entry in data]  # Use 'CWE ID' for stratification

    # Perform stratified k-fold splitting
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    splits = []

    for train_idx, val_idx in skf.split(data, labels):
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        splits.append((train_data, val_data))

    return splits


def split_val_to_eval_and_test(val_data, split_ratio=0.5):
    # Split val_data into eval and test datasets keeping proportions
    labels = [entry["CWE ID"] for entry in val_data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=42)

    eval_data, test_data = [], []
    for train_idx, test_idx in sss.split(val_data, labels):
        eval_data = [val_data[i] for i in train_idx]
        test_data = [val_data[i] for i in test_idx]

    return eval_data, test_data


def save_to_jsonl(data, filename):
    # Save the given data to a JSONL file
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


# Define file path
file_path = "MSR_filtering4.jsonl"  # Path to your JSONL file

# Get stratified splits
splits = create_stratified_splits(file_path, k=5)  # You can use k=10 if you prefer

# For each fold
for fold_idx in range(5):
    print(f"Processing fold {fold_idx + 1}...")

    # Get datasets for this fold
    train_data, val_data = splits[fold_idx]
    eval_data, test_data = split_val_to_eval_and_test(val_data, split_ratio=0.5)

    # Save the datasets for the fold to JSONL files
    save_to_jsonl(train_data, f"CWE_train{fold_idx + 1}.jsonl")
    save_to_jsonl(eval_data, f"CWE_val{fold_idx + 1}.jsonl")
    save_to_jsonl(test_data, f"CWE_test{fold_idx + 1}.jsonl")

    print(f"Fold {fold_idx + 1} saved to CWE_train{fold_idx + 1}.jsonl, CWE_val{fold_idx + 1}.jsonl, and CWE_test{fold_idx + 1}.jsonl.")


# filter out cases where no modifiable lines are included within token limit
def convert_examples_to_features(input_file, output_file, tokenizer_name, token_limit=512):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    with open(output_file, 'w', encoding='utf-8') as output_file:
        with open(input_file, 'r', encoding='utf-8') as file:
            cnt=0
            for line in file:
                json_obj = json.loads(line)
                code = json_obj.get('func_before', '')
                line_info = json_obj.get('lines_ground_truth', '')

                lines = code.splitlines()
                if -1 in line_info:
                    continue
                sparse_matrix = [1 if x in line_info else 0 for x in range(len(lines))][:token_limit]
                sparse_matrix+=[-1]*(token_limit-len(lines))
                code_tokens = []
                for ind,line in enumerate(lines):
                    line_token=tokenizer.tokenize(line+"\n")
                    if len(code_tokens)+len(line_token)<=token_limit - 2:
                        code_tokens.extend(line_token)
                    else:
                        for i in range(ind,len(sparse_matrix)):
                            sparse_matrix[i]=-1
                        break
                if 1 not in sparse_matrix:
                    continue
                cnt+=1
                output_file.write(json.dumps(json_obj) + '\n')
            print(input_file, cnt)


# Load Corresponding tokenizer
bert_tokenizer = "microsoft/codebert-base"
unix_tokenizer = "microsoft/unixcoder-base-nine"
codet5p_tokenizer = "Salesforce/codet5p-220m"
jsonl_files = ["CWE_test.jsonl", "CWE_val.jsonl", "CWE_train.jsonl"]
for i in range(1,6):
    for j in range(len(jsonl_files)):
        convert_examples_to_features(jsonl_files[j][:-6]+str(i)+ jsonl_files[j][-6:], jsonl_files[j][:-6]+str(i)+"_unix_1024"+ jsonl_files[j][-6:], unix_tokenizer, 1024)
        convert_examples_to_features(jsonl_files[j][:-6]+str(i)+ jsonl_files[j][-6:], jsonl_files[j][:-6]+str(i)+"_unix_512"+ jsonl_files[j][-6:], unix_tokenizer, 512)
        convert_examples_to_features(jsonl_files[j][:-6]+str(i)+ jsonl_files[j][-6:], jsonl_files[j][:-6]+str(i)+"_bert_512"+ jsonl_files[j][-6:], bert_tokenizer, 512)
        convert_examples_to_features(jsonl_files[j][:-6]+str(i)+ jsonl_files[j][-6:], jsonl_files[j][:-6]+str(i)+"_codet5p_512"+ jsonl_files[j][-6:], codet5p_tokenizer, 512)


