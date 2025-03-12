#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
models.py - Collection of model implementations
"""

import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import HTTPError, Timeout
import time

# Try to import vLLM, provide warning if not available
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM is not installed, SLM model will not be available. Please install with 'pip install vllm'.")

# Try to import OpenAI, provide warning if not available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI API is not installed, LLM and Teacher models will not be available. Please install with 'pip install openai'.")

# Import prompt templates
from core.prompts import SLM_get_prompt_correct, LLM_correct_prompt, Teacher_correct_prompt

# Import data processing functions
from core.data_utils import read_json, write_json


class SLM:
    """Small Language Model inference class"""
    
    def __init__(self, args):
        """
        Initialize Small Language Model
        
        Args:
            args: Object containing model parameters
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed, cannot use SLM model. Please install with 'pip install vllm'.")
            
        # Set visible GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_visible
        self.args = args
        
        # Initialize model
        self.llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
        
        # Set tokenizer and stop tokens
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        stop_id_sequence = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        stop_tokens = ['</s>', '---', 'Question:']
        
        # Set sampling parameters
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            stop_token_ids=stop_id_sequence,
            skip_special_tokens=True,
            stop=stop_tokens
        )
        
        # Load data
        self.data = read_json(args.file_path)
        print(f"Loaded {len(self.data)} samples")

    def generate(self):
        """
        Generate correction information using small language model
        """
        print("Generating correction information...")
        
        # Prepare prompts
        prompts = []
        for example in self.data:
            prompts.append(SLM_get_prompt_correct(example))
        
        # Generate responses
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        # Process outputs
        results = []
        for i, output in enumerate(outputs):
            result = self.data[i].copy()
            result["SLM correct"] = output.outputs[0].text
            results.append(result)
        
        # Save results
        write_json(results, self.args.output_dir)
        
        print(f"Generated {len(results)} correction results, saved to {self.args.output_dir}")
        return results


class LLMCorrect:
    """Large Language Model correction class"""
    
    def __init__(self, args):
        """
        Initialize Large Language Model correction
        
        Args:
            args: Object containing model parameters
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI API is not installed, cannot use LLM model. Please install with 'pip install openai'.")
            
        self.args = args
        
        # Initialize API client
        self.client = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url
        )
        
        # Load data
        self.data = read_json(args.file_path)
        self.multichoice = args.multichoice
        print(f"Loaded {len(self.data)} samples")

    def process_item(self, item):
        """
        Process a single sample
        
        Args:
            item (dict): Sample to process
            
        Returns:
            dict: Processed sample
        """
        try:
            # Generate prompt
            prompt = LLM_correct_prompt(item, self.multichoice)
            
            # Call API
            response = self.client.chat.completions.create(
                model=self.args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.args.temperature
            )
            
            # Save result
            item["LLM correct"] = response.choices[0].message.content
            return item
        except (HTTPError, Timeout) as e:
            print(f"Error processing sample: {e}")
            time.sleep(5)  # Wait before retrying
            return self.process_item(item)
        except Exception as e:
            print(f"Unexpected error: {e}")
            item["LLM correct"] = "Error: Processing failed"
            return item

    def generate(self):
        """
        Generate LLM corrections using parallel processing
        """
        print("Generating LLM corrections...")
        results = []
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
            future_to_item = {executor.submit(self.process_item, item): item for item in self.data}
            
            # Use tqdm to show progress
            for future in tqdm(as_completed(future_to_item), total=len(self.data), desc="Processing samples"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing sample: {e}")
        
        # Save results
        write_json(results, self.args.output_dir)
        
        print(f"Generated {len(results)} LLM correction results, saved to {self.args.output_dir}")
        return results


class TeacherCorrect:
    """Teacher model verification class"""
    
    def __init__(self, args):
        """
        Initialize Teacher model verification
        
        Args:
            args: Object containing model parameters
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI API is not installed, cannot use Teacher model. Please install with 'pip install openai'.")
            
        self.args = args
        
        # Initialize API client
        self.client = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url
        )
        
        # Load data
        self.data = read_json(args.file_path)
        self.multichoice = args.multichoice
        print(f"Loaded {len(self.data)} samples")

    def process_item(self, item):
        """
        Process a single sample
        
        Args:
            item (dict): Sample to process
            
        Returns:
            dict: Processed sample
        """
        try:
            # Generate prompt
            prompt = Teacher_correct_prompt(item, self.multichoice)
            
            # Call API
            response = self.client.chat.completions.create(
                model=self.args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.args.temperature
            )
            
            # Save result
            item["LLM correct SLM"] = response.choices[0].message.content
            return item
        except (HTTPError, Timeout) as e:
            print(f"Error processing sample: {e}")
            time.sleep(5)  # Wait before retrying
            return self.process_item(item)
        except Exception as e:
            print(f"Unexpected error: {e}")
            item["LLM correct SLM"] = "Error: Processing failed"
            return item

    def generate(self):
        """
        Generate Teacher verifications using parallel processing
        """
        print("Generating Teacher verifications...")
        results = []
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
            future_to_item = {executor.submit(self.process_item, item): item for item in self.data}
            
            # Use tqdm to show progress
            for future in tqdm(as_completed(future_to_item), total=len(self.data), desc="Processing samples"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing sample: {e}")
        
        # Save results
        write_json(results, self.args.output_dir)
        
        print(f"Generated {len(results)} Teacher verification results, saved to {self.args.output_dir}")
        return results


class MultiChoiceProcessor:
    """Multiple choice processor class"""
    
    def __init__(self, args):
        """
        Initialize multiple choice processor
        
        Args:
            args: Object containing model parameters
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed, cannot use multiple choice processor. Please install with 'pip install vllm'.")
            
        # Set visible GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        self.args = args
        
        # Load data
        self.data = read_json(args.input_file)
        print(f"Loaded {len(self.data)} multiple choice samples")
        
        # Initialize model
        self.llm = LLM(args.model_path, gpu_memory_utilization=0.9)
        
        # Set sampling parameters
        self.sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            min_tokens=5,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )

    def get_correct_option(self, item):
        """
        Extract correct option from multiple choice item
        
        Args:
            item (dict): Item containing options and correct answer
        
        Returns:
            str: Content of the correct option
        """
        from core.data_utils import get_correct_option
        return get_correct_option(item)

    def generate(self):
        """
        Process multiple choice questions
        
        Returns:
            list: Processing results
        """
        print("Processing multiple choice questions...")
        
        # Prepare questions
        questions = ['Q:' + item['doc']['Problem'] + "\nA:" for item in self.data]
        
        # Generate answers
        answers = self.llm.generate(questions, self.sampling_params)
        
        # Process results
        results = []
        for i in range(len(answers)):
            correct_answer = self.get_correct_option(self.data[i]['doc'])
            results.append({
                "question": questions[i],
                "original solution": answers[i].outputs[0].text,
                "gold answer": correct_answer
            })
        
        # Save results
        write_json(results, self.args.output_file)
        
        print(f"Processed {len(results)} multiple choice questions, results saved to {self.args.output_file}")
        return results


if __name__ == "__main__":
    # Example usage
    print("Model implementation module")
    print("Please import this module to use its functionality") 