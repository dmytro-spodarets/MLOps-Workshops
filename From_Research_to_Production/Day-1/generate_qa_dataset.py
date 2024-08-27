import os
import re
import json
from glob import glob

def extract_qa_pairs(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    pattern = r'<summary>(?:<b>)?(?:\s*You typing <code>.*?</code> but )?(.+?)(?:</b>)?</summary><br>(?:<b>\s*|\s*)(.+?)(?:\s*</b>)?\s*</details>'
    matches = re.findall(pattern, content, re.DOTALL)

    qa_pairs = []
    for match in matches:
        question = match[0].strip()
        answer = match[1].strip().replace('\n', ' ')
        qa_pairs.append({"question": question, "answer": answer})

    return qa_pairs

def process_files(directory):
    all_qa_pairs = []
    for filepath in glob(f"{directory}/**/README.md", recursive=True):
        all_qa_pairs.extend(extract_qa_pairs(filepath))
    return all_qa_pairs

def save_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    directory = "./temp"
    output_file = "devops_qa_dataset.json"

    qa_dataset = process_files(directory)
    save_to_json(qa_dataset, output_file)

    print(f"Dataset created with {len(qa_dataset)} question-answer pairs.")
    print(f"Saved to {output_file}")