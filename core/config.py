#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
config.py - Configuration and parameter parsing module
"""

import argparse
import os
import json


def get_base_parser():
    """
    Get base argument parser
    
    Returns:
        argparse.ArgumentParser: Base argument parser
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--file_path", type=str, help="Input file path")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--multichoice", type=int, default=0, help="Whether the task is multiple choice (1) or not (0)")
    return parser


def get_slm_parser():
    """
    Get Small Language Model argument parser
    
    Returns:
        argparse.ArgumentParser: SLM argument parser
    """
    parser = argparse.ArgumentParser(
        description="Generate corrections using Small Language Model",
        parents=[get_base_parser()]
    )
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--file_path", type=str, required=True, help="Input file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--multichoice", type=int, required=True, help="Whether the task is multiple choice (1) or not (0)")
    
    # Optional arguments
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--min_tokens", type=int, default=10, help="Minimum tokens to generate")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type for model weights")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--gpu_visible", type=str, default="0,1,2,3", help="Visible GPU devices")
    
    return parser


def get_llm_parser():
    """
    Get Large Language Model argument parser
    
    Returns:
        argparse.ArgumentParser: LLM argument parser
    """
    parser = argparse.ArgumentParser(
        description="Generate corrections using Large Language Model API",
        parents=[get_base_parser()]
    )
    
    # Required arguments
    parser.add_argument("--api_key", type=str, required=True, help="API key for LLM service")
    parser.add_argument("--base_url", type=str, required=True, help="Base URL for API endpoint")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--file_path", type=str, required=True, help="Input file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_workers", type=int, required=True, help="Number of parallel workers")
    parser.add_argument("--multichoice", type=int, required=True, help="Whether the task is multiple choice (1) or not (0)")
    
    return parser


def get_teacher_parser():
    """
    Get Teacher model argument parser
    
    Returns:
        argparse.ArgumentParser: Teacher argument parser
    """
    # Teacher model uses the same arguments as LLM
    return get_llm_parser()


def get_multichoice_parser():
    """
    Get multiple choice processor argument parser
    
    Returns:
        argparse.ArgumentParser: Multiple choice processor argument parser
    """
    parser = argparse.ArgumentParser(description="Process multiple choice questions")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--input_file", type=str, required=True, help="Input file path")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    
    # Optional arguments
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="GPU device IDs")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    
    return parser


def get_process_parser():
    """
    Get evaluation results processor argument parser
    
    Returns:
        argparse.ArgumentParser: Evaluation results processor argument parser
    """
    parser = argparse.ArgumentParser(description="Process evaluation results")
    
    # Required arguments
    parser.add_argument("input_file", type=str, help="Input file path")
    
    # Optional arguments
    parser.add_argument("output_file", type=str, nargs="?", help="Output file path")
    parser.add_argument("--split_ratio", type=float, default=0.5, help="Data split ratio")
    
    return parser


def get_self_correction_parser():
    """
    Get self-correction processor argument parser
    
    Returns:
        argparse.ArgumentParser: Self-correction processor argument parser
    """
    parser = argparse.ArgumentParser(description="Process self-correction data")
    
    # Required arguments
    parser.add_argument("input_file", type=str, help="Input file path")
    parser.add_argument("output_file", type=str, help="Output file path")
    
    return parser


def get_balance_parser():
    """
    Get dataset balancer argument parser
    
    Returns:
        argparse.ArgumentParser: Dataset balancer argument parser
    """
    parser = argparse.ArgumentParser(description="Balance dataset")
    
    # Required arguments
    parser.add_argument("input_file", type=str, help="Input file path")
    parser.add_argument("output_file", type=str, help="Output file path")
    
    # Optional arguments
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser


def load_config_from_file(config_file):
    """
    Load configuration from file
    
    Args:
        config_file (str): Configuration file path
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file {config_file} does not exist")
        
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def save_config_to_file(config, config_file):
    """
    Save configuration to file
    
    Args:
        config (dict): Configuration dictionary
        config_file (str): Configuration file path
    """
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print(f"Configuration saved to {config_file}")


def args_to_dict(args):
    """
    Convert arguments object to dictionary
    
    Args:
        args: Arguments object
        
    Returns:
        dict: Arguments dictionary
    """
    return vars(args)


def dict_to_args(args_dict, parser):
    """
    Convert dictionary to arguments object
    
    Args:
        args_dict (dict): Arguments dictionary
        parser (argparse.ArgumentParser): Argument parser
        
    Returns:
        argparse.Namespace: Arguments object
    """
    args = parser.parse_args([])
    for key, value in args_dict.items():
        setattr(args, key, value)
    return args


if __name__ == "__main__":
    # Example usage
    print("Configuration module")
    print("Please import this module to use its functionality") 