#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eval_utils.py - Evaluation utilities for mathematical reasoning tasks
"""

import json
import re
import os
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== Basic Evaluation Functions =====================

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


# ===================== MathQA Evaluation Functions =====================

def extract_answer(example):
    """
    Extract answer from model output
    
    Args:
        example (str): Model output text
        
    Returns:
        list: Extracted answer tokens
    """
    pattern = r"(?:Therefore, the answer is|the final answer is|the answer is|the ans is|the answer|answer is|####)\s*(.+)"
    match = re.search(pattern, example, flags=re.IGNORECASE)
    if match:
        extracted = match.group(1).strip()
        # Try to match \boxed{...} pattern
        boxed_match = re.search(r"\\boxed\{(.+?)\}", extracted)
        if boxed_match:
            return boxed_match.group(1).strip().lower().split()
        return extracted.lower().split()
    return "?_+"


def extract_last_letter(example):
    """
    Extract the last letter (a, b, c, d, e) from model output
    
    Args:
        example (str): Model output text
        
    Returns:
        str: Extracted letter or error marker
    """
    pattern = r"\b([abcde])\b"
    matches = re.findall(pattern, example, flags=re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    return "?_+"


def extract_last_number(example):
    """
    Extract the last number from model output
    
    Args:
        example (str): Model output text
        
    Returns:
        str: Extracted number or error marker
    """
    pattern = r"\b(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)\b"
    matches = re.findall(pattern, example)
    if matches:
        return matches[-1]
    return "?_+"


def extract_value(options_str, correct_option):
    """
    Extract the value of the correct option from options string
    
    Args:
        options_str (str): Options string
        correct_option (str): Correct option letter (a, b, c, d, e)
        
    Returns:
        str: Value of the correct option or None
    """
    pattern = r"([abcde])\s*\)\s*([^,]+)"
    matches = re.findall(pattern, options_str, flags=re.IGNORECASE)
    options = {key.lower(): value.strip() for key, value in matches}
    return options.get(correct_option, None)


def evaluate_mathqa_completion(completion, gold_answer, options):
    """
    Evaluate if the model's completion matches the gold answer
    
    Args:
        completion (str): Model's completion
        gold_answer (str): Gold answer (a, b, c, d, e)
        options (str): Options string
        
    Returns:
        bool: True if the completion matches the gold answer
    """
    # Check if the extracted letter matches the gold answer
    if gold_answer == extract_last_letter(completion):
        return True
    
    # Check if the extracted number is in the correct option's value
    if extract_last_number(completion) in extract_value(options, gold_answer):
        return True
    
    return False


# ===================== Few-shot Evaluation Templates =====================

MATHQA_PROMPT_TEMPLATE = "The following are multiple choice questions (with answers) about mathematical reasoning. Choose a correct answer that appears in the candidate answers. Your response should conclude with the format \"Therefore, the answer is \\boxed{{answer}}\".\n\n"

FEW_SHOT_TEMPLATE = """Q: John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is?
Answer Choices: a ) 50 , b ) 45 , c ) 65 , d ) 78 , e ) 64
A: If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50.
Therefore, the answer is \\boxed{{a}}.

Q: If a / b = 3/4 and 8a + 5b = 22,then find the value of a.
Answer Choices: a ) 1/2 , b ) 3/2 , c ) 5/2 , d ) 4/2 , e ) 7/2
A: If a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22. So a is equal to 3/2.
Therefore, the answer is \\boxed{{b}}.

Q: A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance?
Answer Choices: a ) 53 km , b ) 55 km , c ) 52 km , d ) 60 km , e ) 50 km
A: The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km.
Therefore, the answer is \\boxed{{e}}.

Q: How many keystrokes are needed to type the numbers from 1 to 500?
Answer Choices: a ) 1156 , b ) 1392 , c ) 1480 , d ) 1562 , e ) 1788
A: There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401 three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 1392.
Therefore, the answer is \\boxed{{b}}.

Q: {Problem}
Answer Choices: {options}
A: 
"""

def get_mathqa_template(example):
    """
    Get the few-shot template for MathQA evaluation
    
    Args:
        example (dict): Example containing Problem and options
        
    Returns:
        str: Formatted template
    """
    template = MATHQA_PROMPT_TEMPLATE + FEW_SHOT_TEMPLATE.format(
        Problem=example['Problem'], 
        options=example['options']
    )
    return template


# ===================== MathQA Evaluation Class =====================

class MathQAEvaluator:
    """MathQA evaluation class"""
    
    def __init__(self, args):
        """
        Initialize MathQA evaluator
        
        Args:
            args: Object containing evaluation parameters
        """
        self.args = args
        
        # Set visible GPUs
        if hasattr(args, 'gpu_visible'):
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_visible
        
        # Load model if provided
        if hasattr(args, 'model') and args.model:
            self.model = args.model
        else:
            self.model = None
        
        # Set sampling parameters if provided
        if hasattr(args, 'sampling_params') and args.sampling_params:
            self.sampling_params = args.sampling_params
        else:
            self.sampling_params = None
        
        # Load dataset
        if hasattr(args, 'dataset_path') and args.dataset_path:
            self.data = read_json(args.dataset_path)
        else:
            self.data = load_dataset("allenai/math_qa", trust_remote_code=True)['test']
        
        print(f"Loaded {len(self.data)} MathQA test samples")

    def process_data(self):
        """
        Process data for evaluation
        
        Returns:
            list: Processed data
        """
        processed_data = []
        for example in self.data:
            tmp = {}
            tmp['prompt'] = get_mathqa_template(example)
            tmp['Problem'] = example['Problem']
            tmp['options'] = example['options']
            tmp['rationale'] = example.get('Rationale', '')
            tmp['gold answer'] = example['correct']
            processed_data.append(tmp)
        return processed_data

    def generate_completions(self):
        """
        Generate completions using the model
        
        Returns:
            list: Data with completions
        """
        if not self.model:
            raise ValueError("Model not provided for generation")
        
        # Process data
        self.data = self.process_data()
        
        # Generate completions
        print("Generating completions...")
        prompts = [example['prompt'] for example in self.data]
        completions = self.model.generate(prompts, self.sampling_params)
        
        # Add completions to data
        for i, example in enumerate(self.data):
            example['completion'] = completions[i].outputs[0].text
        
        # Save results
        if hasattr(self.args, 'output_path') and self.args.output_path:
            os.makedirs(os.path.dirname(self.args.output_path), exist_ok=True)
            write_json(self.data, self.args.output_path)
        
        return self.data

    def evaluate_completions(self, data=None):
        """
        Evaluate completions against gold answers
        
        Args:
            data (list, optional): Data with completions. Defaults to None.
            
        Returns:
            tuple: Accuracy and evaluated data
        """
        if data is None:
            data = self.data
        
        # Evaluate completions
        print("Evaluating completions...")
        correct_count = 0
        
        for item in tqdm(data):
            if 'completion' not in item:
                continue
                
            is_correct = evaluate_mathqa_completion(
                item['completion'], 
                item['gold answer'], 
                item['options']
            )
            
            item['is_correct'] = is_correct
            if is_correct:
                correct_count += 1
        
        # Calculate accuracy
        accuracy = correct_count / len(data)
        print(f"Accuracy: {accuracy:.4f} ({correct_count}/{len(data)})")
        
        # Save evaluation results
        if hasattr(self.args, 'eval_output_path') and self.args.eval_output_path:
            os.makedirs(os.path.dirname(self.args.eval_output_path), exist_ok=True)
            
            eval_results = {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_count": len(data),
                "data": data
            }
            
            write_json(eval_results, self.args.eval_output_path)
        
        return accuracy, data

    def run_evaluation(self):
        """
        Run the complete evaluation pipeline
        
        Returns:
            tuple: Accuracy and evaluated data
        """
        # Generate completions if model is provided
        if self.model:
            self.generate_completions()
        
        # Evaluate completions
        return self.evaluate_completions()


if __name__ == "__main__":
    # Example usage
    print("Evaluation utilities module")
    print("Please import this module to use its functionality") 