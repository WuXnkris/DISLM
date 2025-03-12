#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
data_utils.py - Collection of data processing utilities
"""

import json
import re
import random
from tqdm import tqdm

# ===================== Basic Data I/O Functions =====================

def read_jsonl(file_path):
    """
    Read data from a JSONL format file
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        list: List of data items
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def read_json(file_path):
    """
    Read data from a JSON format file
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        dict/list: Data object
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def write_json(data, file_path, indent=4):
    """
    Write data to a JSON format file
    
    Args:
        data (dict/list): Data to write
        file_path (str): Path to the file
        indent (int, optional): Number of spaces for indentation. Defaults to 4.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)
    print(f"Data saved to {file_path}")


def write_jsonl(data, file_path):
    """
    Write data to a JSONL format file
    
    Args:
        data (list): List of data items to write
        file_path (str): Path to the file
    """
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Data saved to {file_path}")


# ===================== Error Identification and Filtering Functions =====================

def find_mistakes_match(file_name, output_dir):
    """
    Find mismatched incorrect answers and save them
    
    Args:
        file_name (str): Input file path
        output_dir (str): Output file path
    """
    mistakes = []
    data = read_jsonl(file_name)
    for example in data:
        if example["exact_match"] != 1.0:
            mistakes.append(example)
    write_json(mistakes, output_dir)
    print(f"Found {len(mistakes)} incorrect answers out of {len(data)} samples")


def find_mistakes_multichoice(file_name, output_dir):
    """
    Find incorrect answers in multiple choice questions and save them
    
    Args:
        file_name (str): Input file path
        output_dir (str): Output file path
    """
    mistakes = []
    data = read_jsonl(file_name)
    for example in data:
        if example["acc"] != 1.0:
            mistakes.append(example)
    write_json(mistakes, output_dir)
    print(f"Found {len(mistakes)} incorrect answers out of {len(data)} samples")


# ===================== Text Extraction and Processing Functions =====================

def extract_mistake_analysis(text):
    """
    Extract mistake analysis section from correction text
    
    Args:
        text (str): Correction text
        
    Returns:
        str: Extracted mistake analysis or None if not found
    """
    pattern = r"(?i)Mistake Analysis:(.+?)(?:Correct Solution:|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_correct_solution(text):
    """
    Extract correct solution section from correction text
    
    Args:
        text (str): Correction text
        
    Returns:
        str: Extracted correct solution or None if not found
    """
    pattern = r"(?i)Correct Solution:(.+?)(?:$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def filter_reasoning_path(example):
    """
    Extract correct reasoning path from LLM response
    
    Args:
        example (str): LLM response text
        
    Returns:
        str: Extracted correct reasoning path or None if not found
    """
    pattern = r"Correct\s*solution\s*[:](.*)"
    match = re.search(pattern, example, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def get_correct_option(item):
    """
    Extract correct option from multiple choice item
    
    Args:
        item (dict): Item containing options and correct answer
    
    Returns:
        str: Content of the correct option or None if not found
    """
    options_str = item['options']
    correct_option = item['correct']
    options = options_str.split(' , ')

    # Find the correct option
    for option in options:
        option = option.strip()
        if ' ) ' in option:
            letter, content = option.split(' ) ', 1)
            if letter == correct_option:
                return content
    
    return None


# ===================== Model Output Processing Functions =====================

def filter_LLM_correct(file_name, output_dir):
    """
    Extract correction paths from large language model outputs
    
    Args:
        file_name (str): Input file path
        output_dir (str): Output file path
    """
    data = read_json(file_name)
    pattern = "(?i)(Correct Solution:|correct solution:|Correct solution:)([\\s\\S]*)"
    for item in data:
        matches = re.findall(pattern, item["LLM correct"])
        if matches:
            item["lm_correct"] = matches[0][1].strip()
    write_json(data, output_dir)


def filter_Teacher_correct(file_name, output_dir):
    """
    Extract correction information from teacher model outputs
    
    Args:
        file_name (str): Input file path
        output_dir (str): Output file path
    """
    data = read_json(file_name)
    pattern = r"(?i)(LLM solution:|LLM solution :)([\s\S]*)"
    mixdata = []
    for item in data:
        solution = re.findall(pattern, item["LLM correct SLM"])
        if len(solution):
            item["teacher_correct"] = (
                solution[0][1].strip().replace("<", "").replace(">", "")
            )
            mixdata.append(item)
        else:
            continue
    write_json(data, output_dir)


def filter_corrections(input_file, output_file):
    """
    Find incorrect answers that need correction and save them
    
    Args:
        input_file (str): Input file path
        output_file (str): Output file path
    """
    data = read_json(input_file)
    filtered_data = []
    
    for item in data:
        if "SLM correct" not in item:
            continue
            
        correction = item["SLM correct"]
        mistake_analysis = extract_mistake_analysis(correction)
        correct_solution = extract_correct_solution(correction)
        
        if mistake_analysis and correct_solution:
            filtered_item = {
                "question": item.get("question", ""),
                "original_solution": item.get("original solution", ""),
                "gold_answer": item.get("gold answer", ""),
                "mistake_analysis": mistake_analysis,
                "correct_solution": correct_solution
            }
            filtered_data.append(filtered_item)
    
    write_json(filtered_data, output_file)
    print(f"Filtered {len(filtered_data)} valid corrections out of {len(data)} samples")


# ===================== Data Conversion and Preparation Functions =====================

def process_correct_data_to_sft(file_name, output_dir):
    """
    Process correct data to generate supervised fine-tuning (SFT) data
    
    Args:
        file_name (str): Input file path
        output_dir (str): Output file path
    """
    pattern = "(?i)(Correct Solution:|correct solution:|Correct solution:)([\\s\\S]*)"
    data = read_json(file_name)
    mixdata = []
    for item in data:
        tmp = {'input': "Question: " + item['question']}
        response = re.findall(pattern, item['LLM correct'])
        if len(response):
            tmp['output'] = "Answer: " + re.findall(pattern, item['LLM correct'])[0][1].strip()
            mixdata.append(tmp)
    write_json(mixdata, output_dir)
    print(f"Generated {len(mixdata)} SFT training samples")


def prepare_training_data(file_path, output_path):
    """
    Prepare training data from filtered corrections
    
    Args:
        file_path (str): Path to filtered corrections file
        output_path (str): Path to output training data
    """
    data = read_json(file_path)
    training_data = []
    
    for item in data:
        input_text = f"""Question: {item['question']}
Original Solution: {item['original_solution']}
Gold Answer: {item['gold_answer']}"""

        output_text = f"""Mistake Analysis: {item['mistake_analysis']}
Correct Solution: {item['correct_solution']}"""

        training_item = {
            "input": input_text,
            "output": output_text
        }
        training_data.append(training_item)
    
    write_json(training_data, output_path)
    print(f"Prepared {len(training_data)} training samples")


def process_self_correction_data(input_path, output_path):
    """
    Process self-correction data and prepare for fine-tuning
    
    Args:
        input_path (str): Input file path
        output_path (str): Output file path
    
    Returns:
        list: Processed data
    """
    # Read input data
    data = read_json(input_path)
    
    # Process data
    data_count = 0
    processed_data = []
    
    for example in data:
        # Extract reasoning path
        reasoning_path = filter_reasoning_path(example.get('SLM correct', ''))
        
        if reasoning_path:
            # Create training example
            training_example = {
                'input': f"Q:{example['question']}\nA:", 
                'output': reasoning_path
            }
            processed_data.append(training_example)
            data_count += 1
    
    # Print statistics
    print('Processed data count:', data_count)
    
    # Save processed data
    write_json(processed_data, output_path)
    
    return processed_data


# ===================== Data Balancing and Sampling Functions =====================

def balance_dataset(input_file, output_file, max_samples=None, seed=42):
    """
    Balance dataset by sampling from different error types
    
    Args:
        input_file (str): Input file path
        output_file (str): Output file path
        max_samples (int, optional): Maximum number of samples. Defaults to None (all samples).
        seed (int, optional): Random seed. Defaults to 42.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load data
    data = read_json(input_file)
    
    # Group data by error types (if available)
    error_types = {}
    
    for item in data:
        # Try to extract error type from mistake analysis
        error_type = "unknown"
        if "mistake_analysis" in item:
            analysis = item["mistake_analysis"].lower()
            
            # Simple heuristic to categorize errors
            if "calculation" in analysis or "arithmetic" in analysis:
                error_type = "calculation"
            elif "misunderstanding" in analysis or "misinterpret" in analysis:
                error_type = "misunderstanding"
            elif "logic" in analysis or "reasoning" in analysis:
                error_type = "logic"
            elif "formula" in analysis or "equation" in analysis:
                error_type = "formula"
        
        # Add to appropriate group
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(item)
    
    # Print statistics
    print("Error type distribution:")
    for error_type, items in error_types.items():
        print(f"  {error_type}: {len(items)} samples")
    
    # Determine how many samples to take from each group
    total_samples = sum(len(items) for items in error_types.values())
    if max_samples is None or max_samples >= total_samples:
        # Use all samples if max_samples is None or greater than total
        balanced_data = data
    else:
        # Calculate samples per group proportionally
        balanced_data = []
        remaining = max_samples
        
        # First pass: calculate minimum samples per group
        min_samples_per_group = max(1, max_samples // len(error_types))
        
        # Take minimum samples from each group
        for error_type, items in error_types.items():
            samples_to_take = min(min_samples_per_group, len(items))
            balanced_data.extend(random.sample(items, samples_to_take))
            remaining -= samples_to_take
        
        # Second pass: distribute remaining samples proportionally
        if remaining > 0:
            remaining_items = []
            for error_type, items in error_types.items():
                # Exclude already selected items
                remaining_items.extend([item for item in items if item not in balanced_data])
            
            # Take remaining samples
            samples_to_take = min(remaining, len(remaining_items))
            if samples_to_take > 0:
                balanced_data.extend(random.sample(remaining_items, samples_to_take))
    
    # Shuffle the balanced dataset
    random.shuffle(balanced_data)
    
    # Save balanced dataset
    write_json(balanced_data, output_file)
    
    print(f"Balanced dataset contains {len(balanced_data)} samples, saved to {output_file}")


# ===================== Evaluation Processing Functions =====================

def process_evaluation_results(file_path, output_path=None, split_ratio=0.5):
    """
    Process evaluation results and calculate accuracy
    
    Args:
        file_path (str): Evaluation results file path
        output_path (str, optional): Path to save processed results. Defaults to None.
        split_ratio (float, optional): Ratio to split the data. Defaults to 0.5.
    
    Returns:
        tuple: Accuracy and processed data
    """
    # Read evaluation data
    data = read_jsonl(file_path)
    
    # Split data based on ratio
    split_index = int(len(data) * split_ratio)
    data = data[split_index:]  # Take second half
    
    # Further split for analysis
    data1 = data[:700]
    data2 = data[700:]
    
    # Calculate accuracy
    correct_count = 0
    for item in data1:
        if item['exact_match'] == 1.0:
            correct_count += 1
    
    accuracy = correct_count / len(data1)
    
    # Print statistics
    print('Accuracy:', accuracy)
    print('Data1 size:', len(data1), 'Data2 size:', len(data2))
    print('Total data size:', len(data))
    
    # Save processed data if output path is provided
    if output_path:
        write_json(data, output_path)
    
    return accuracy, data


if __name__ == "__main__":
    # Example usage
    print("Data processing utilities module")
    print("Please import this module to use its functions") 