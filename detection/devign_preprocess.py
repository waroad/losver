import json
import difflib
import subprocess
import json
import re

from transformers import AutoTokenizer


def extract_function_from_commit(repo_path, commit_id, function_name, function_name_short):
    try:
        # Step 1: Get the list of files modified in the commit
        result = subprocess.run(
            ["git", "-C", repo_path, "show", "--name-only", "--pretty=format:", commit_id],
            stdout=subprocess.PIPE,
            check=True,
            text=True
        )
        changed_files = result.stdout.strip().split("\n")
        print(f"Files modified in commit {commit_id}: {repo_path}")
        # Step 2: Look for the function in each file
        for file in changed_files:
            if file:  # Skip empty lines
                # Get the content of the file after the commit
                file_content = subprocess.run(
                    ["git", "-C", repo_path, "show", f"{commit_id}:{file}"],
                    stdout=subprocess.PIPE,
                    check=True,
                    text=True
                ).stdout.splitlines()
                # Search for the function
                inside_function = False
                brace_count = 999
                function_lines = []
                accumulate_ind=0
                file_content_sum=''.join(file_content)
                for line in file_content:
                    if not inside_function and re.search(rf"{re.escape(function_name)}", line):
                        correct_func=False
                        for i in range(accumulate_ind,accumulate_ind+500):
                            if file_content_sum[i]==';':
                                break
                            elif file_content_sum[i]=='{':
                                correct_func=True
                                break
                        if not correct_func:
                            continue
                        print(f"\n--- Function '{function_name}' found in file '{file}' ---")
                        inside_function = True

                    if inside_function:
                        function_lines.append(line)
                        if brace_count==999 and line.count("{"):
                            brace_count=line.count("{")
                            brace_count -= line.count("}")
                        else:
                            brace_count += line.count("{")
                            brace_count -= line.count("}")
                        if brace_count == 0:  # End of function
                            break
                    accumulate_ind+=len(line)

                if function_lines:
                    return "\n".join(function_lines)
                else:
                    accumulate_ind=0
                    for line in file_content:
                        if (not inside_function and line and line[0]!=' ' and
                                re.search(rf"{re.escape(function_name_short)}", line)):
                            correct_func = False
                            for i in range(accumulate_ind, accumulate_ind + 200):
                                if file_content_sum[i] == ';':
                                    break
                                elif file_content_sum[i] == '{':
                                    correct_func = True
                                    break
                            if not correct_func:
                                continue
                            print(f"\n--- Function '{function_name_short}' found in file2 '{file}' ---")
                            inside_function = True

                        if inside_function:
                            function_lines.append(line)
                            if brace_count == 999 and line.count("{"):
                                brace_count = line.count("{")
                                brace_count -= line.count("}")
                            else:
                                brace_count += line.count("{")
                                brace_count -= line.count("}")
                            if brace_count == 0:  # End of function
                                break
                        accumulate_ind+=len(line)

                    if function_lines:
                        return "\n".join(function_lines)

        return 0
    except subprocess.CalledProcessError as e:
        print(f"**Error: {e}")


def extract_first_function_decl(func_code):
    """
    Extract the first function declaration from the given function code.
    """
    for ind,i in enumerate(func_code):
        if i=='(':
            point=ind
        if i==")" or i==",":
            function_name=func_code[0:ind]
            break
    for i in range(point,-1,-1):
        if i<point-1 and func_code[i]==' ' or func_code[i]=='*':
            function_name_compact=func_code[i+1:point]
            break
        if i==0:
            function_name_compact=func_code[i:point]

    return function_name,function_name_compact


def normalize_code(code):
    # Normalize whitespace and line endings
    lines = code.splitlines()
    normalized_lines = [line.rstrip() for line in lines]  # Remove trailing whitespace
    return "\n".join(normalized_lines)  # Standardize to \n


def find_lines_with_char_positions(func_code, fixed_func):
    func_code_lines = func_code.splitlines(keepends=True)  # Keep newline characters for correct indexing
    fixed_func_lines = fixed_func.splitlines(keepends=True)

    # Use Differ to find line-level differences
    differ = difflib.Differ()
    diff = list(differ.compare(func_code_lines, fixed_func_lines))
    func_code_index = 0
    lines_gt=[]
    for line in diff:
        if line.startswith("- "):  # Line deleted from original code
            deleted_line_text = line[2:]  # Strip the "- " prefix
            if deleted_line_text.strip():  # Ignore empty or whitespace-only deletions
                if func_code_index not in lines_gt:
                    lines_gt.append(func_code_index)
            func_code_index += 1  # Move to the next line in the original code
        elif line.startswith("  "):  # Line unchanged
            func_code_index += 1  # Move to the next line in the original code

    return lines_gt


def process_jsonl_file(jsonl_path, output_path):
    """
    Read the JSONL file and process each entry.
    """
    with open(output_path, 'w', encoding='utf-8') as output_file:
        with open(jsonl_path, 'r') as file:
            cnt=0
            for ind,line in enumerate(file):
                print(f"*****************************\n             {ind}\n*****************************")
                try:
                    entry = json.loads(line)
                    project = entry.get("project")
                    commit_id = entry.get("commit_id")
                    func_code = entry.get("func")

                    if not all([project, commit_id, func_code]):
                        continue
                    function_name, function_name_short = extract_first_function_decl(func_code)
                    if not function_name:
                        continue
                    fixed_func=extract_function_from_commit(project, commit_id, function_name, function_name_short)
                    func_code = normalize_code(func_code)
                    entry["func"]=func_code
                    if fixed_func:
                        fixed_func = normalize_code(fixed_func)
                        lines_gt = find_lines_with_char_positions(func_code,fixed_func)
                        print(lines_gt)
                        if len(lines_gt) ==0:
                            lines_gt=[-1]
                    else:
                        print(function_name_short, "No changes, or deletion")
                        lines_gt= [-1]
                    if -1 not in lines_gt:
                        print(cnt)
                        cnt+=1
                    entry["lines_ground_truth"]=lines_gt
                    output_file.write(json.dumps(entry) + '\n')
                except Exception as e:
                    print(f"Error processing line: {line}\nError: {e}")
                    entry["lines_ground_truth"]=[-1]
                    output_file.write(json.dumps(entry) + '\n')


def remove_unnecessary_padding(input_file, output_file):
    with open(output_file, 'w', encoding='utf-8') as output_file:
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                func_text = json_obj.get('func', '')  # Get the 'func' field
                func_text2 = func_text.replace("\n\n", "\n")  # Replace double newlines with a single newline
                json_obj['func'] = func_text2  # Update the 'func' field with the cleaned text
                output_file.write(json.dumps(json_obj) + '\n')


def convert_examples_to_features(input_file, output_file, tokenizer_name, token_limit=512):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    with open(output_file, 'w', encoding='utf-8') as output_file:
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                code = json_obj.get('func', '')
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
                output_file.write(json.dumps(json_obj) + '\n')


def process_files(jsonl_files, output_files):
    # Ensure the output files are opened for writing

    for j in range(len(jsonl_files)):
        output_file_temp1="temp1.jsonl"
        output_file_temp2="temp2.jsonl"
        remove_unnecessary_padding(jsonl_files[j],output_file_temp1)
        process_jsonl_file(output_file_temp1, output_file_temp2)
        for k in range(len(output_files)):
            convert_examples_to_features(output_file_temp2,output_files[k][j], output_files[k][3], output_files[k][4])


# File paths to the original jsonl files and the new output files

bert_tokenizer = "microsoft/codebert-base"
unix_tokenizer = "microsoft/unixcoder-base-nine"
codet5p_tokenizer = "Salesforce/codet5p-220m"
jsonl_files = ["test.jsonl", "valid.jsonl", "train.jsonl"]
output_files = [["test_unix_1024.jsonl","val_unix_1024.jsonl","train_unix_1024.jsonl", unix_tokenizer, 1024],
                ["test_unix_512.jsonl","val_unix_512.jsonl","train_unix_512.jsonl", unix_tokenizer, 512],
                ["test_bert_512.jsonl","val_bert_512.jsonl","train_bert_512.jsonl", bert_tokenizer, 512],
                ["test_codet5p_512.jsonl","val_codet5p_512.jsonl","train_codet5p_512.jsonl", codet5p_tokenizer, 512]]
process_files(jsonl_files, output_files)
