#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dislm.py - Unified entry script for the DISLM project
"""

import sys
import os
import argparse
from core.config import (
    get_slm_parser, get_llm_parser, get_teacher_parser,
    get_multichoice_parser, get_process_parser,
    get_self_correction_parser, get_balance_parser,
    load_config_from_file, save_config_to_file
)
from core.models import SLM, LLMCorrect, TeacherCorrect, MultiChoiceProcessor
from core.data_utils import (
    process_evaluation_results, process_self_correction_data,
    balance_dataset, filter_corrections, prepare_training_data
)
from core.eval_utils import MathQAEvaluator


def run_slm(args):
    """
    Run Small Language Model correction
    
    Args:
        args: Command line arguments
    """
    print("=== Running Small Language Model Correction ===")
    slm = SLM(args)
    slm.generate()


def run_llm(args):
    """
    Run Large Language Model correction
    
    Args:
        args: Command line arguments
    """
    print("=== Running Large Language Model Correction ===")
    llm = LLMCorrect(args)
    llm.generate()


def run_teacher(args):
    """
    Run Teacher model verification
    
    Args:
        args: Command line arguments
    """
    print("=== Running Teacher Model Verification ===")
    teacher = TeacherCorrect(args)
    teacher.generate()


def run_multichoice(args):
    """
    Run multiple choice processing
    
    Args:
        args: Command line arguments
    """
    print("=== Running Multiple Choice Processing ===")
    processor = MultiChoiceProcessor(args)
    processor.generate()


def run_process(args):
    """
    Run evaluation results processing
    
    Args:
        args: Command line arguments
    """
    print("=== Running Evaluation Results Processing ===")
    process_evaluation_results(args.input_file, args.output_file, args.split_ratio)


def run_self_correction(args):
    """
    Run self-correction processing
    
    Args:
        args: Command line arguments
    """
    print("=== Running Self-Correction Processing ===")
    process_self_correction_data(args.input_file, args.output_file)


def run_balance(args):
    """
    Run dataset balancing
    
    Args:
        args: Command line arguments
    """
    print("=== Running Dataset Balancing ===")
    balance_dataset(args.input_file, args.output_file, args.max_samples, args.seed)


def run_filter(args):
    """
    Run correction filtering
    
    Args:
        args: Command line arguments
    """
    print("=== Running Correction Filtering ===")
    filter_corrections(args.input_file, args.output_file)


def run_prepare_training(args):
    """
    Run training data preparation
    
    Args:
        args: Command line arguments
    """
    print("=== Running Training Data Preparation ===")
    prepare_training_data(args.input_file, args.output_file)


def run_mathqa_eval(args):
    """
    Run MathQA evaluation
    
    Args:
        args: Command line arguments
    """
    print("=== Running MathQA Evaluation ===")
    evaluator = MathQAEvaluator(args)
    evaluator.run_evaluation()


def run_from_config(config_file, task):
    """
    Run task from configuration file
    
    Args:
        config_file (str): Configuration file path
        task (str): Task name
    """
    config = load_config_from_file(config_file)
    
    if task == "slm":
        parser = get_slm_parser()
        args = parser.parse_args([])
        for key, value in config.items():
            setattr(args, key, value)
        run_slm(args)
    elif task == "llm":
        parser = get_llm_parser()
        args = parser.parse_args([])
        for key, value in config.items():
            setattr(args, key, value)
        run_llm(args)
    elif task == "teacher":
        parser = get_teacher_parser()
        args = parser.parse_args([])
        for key, value in config.items():
            setattr(args, key, value)
        run_teacher(args)
    elif task == "mathqa-eval":
        parser = get_mathqa_eval_parser()
        args = parser.parse_args([])
        for key, value in config.items():
            setattr(args, key, value)
        run_mathqa_eval(args)
    else:
        print(f"Unknown task: {task}")


def get_mathqa_eval_parser():
    """
    Get MathQA evaluation argument parser
    
    Returns:
        argparse.ArgumentParser: MathQA evaluation argument parser
    """
    parser = argparse.ArgumentParser(description="Evaluate models on MathQA dataset")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save generation results")
    parser.add_argument("--eval_output_path", type=str, required=True, help="Path to save evaluation results")
    
    # Optional arguments
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset (if not using default)")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Size of tensor parallelism")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type for model weights")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--gpu_visible", type=str, default="0,1,2,3", help="Visible GPU devices")
    
    return parser


def main():
    """
    Main function
    """
    # Create main parser
    parser = argparse.ArgumentParser(description="DISLM - Framework for detecting and correcting reasoning errors in language models")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")
    
    # Add sub-commands
    slm_parser = subparsers.add_parser("slm", help="Generate corrections using Small Language Model", parents=[get_slm_parser()], add_help=False)
    slm_parser.set_defaults(func=run_slm)
    
    llm_parser = subparsers.add_parser("llm", help="Generate corrections using Large Language Model", parents=[get_llm_parser()], add_help=False)
    llm_parser.set_defaults(func=run_llm)
    
    teacher_parser = subparsers.add_parser("teacher", help="Verify corrections using Teacher model", parents=[get_teacher_parser()], add_help=False)
    teacher_parser.set_defaults(func=run_teacher)
    
    multichoice_parser = subparsers.add_parser("multichoice", help="Process multiple choice questions", parents=[get_multichoice_parser()], add_help=False)
    multichoice_parser.set_defaults(func=run_multichoice)
    
    process_parser = subparsers.add_parser("process", help="Process evaluation results", parents=[get_process_parser()], add_help=False)
    process_parser.set_defaults(func=run_process)
    
    self_correction_parser = subparsers.add_parser("self-correction", help="Process self-correction data", parents=[get_self_correction_parser()], add_help=False)
    self_correction_parser.set_defaults(func=run_self_correction)
    
    balance_parser = subparsers.add_parser("balance", help="Balance dataset", parents=[get_balance_parser()], add_help=False)
    balance_parser.set_defaults(func=run_balance)
    
    # Add filtering and training data preparation sub-commands
    filter_parser = subparsers.add_parser("filter", help="Filter correction results")
    filter_parser.add_argument("input_file", type=str, help="Input file path")
    filter_parser.add_argument("output_file", type=str, help="Output file path")
    filter_parser.set_defaults(func=run_filter)
    
    prepare_parser = subparsers.add_parser("prepare", help="Prepare training data")
    prepare_parser.add_argument("input_file", type=str, help="Input file path")
    prepare_parser.add_argument("output_file", type=str, help="Output file path")
    prepare_parser.set_defaults(func=run_prepare_training)
    
    # Add MathQA evaluation sub-command
    mathqa_eval_parser = subparsers.add_parser("mathqa-eval", help="Evaluate models on MathQA dataset", parents=[get_mathqa_eval_parser()], add_help=False)
    mathqa_eval_parser.set_defaults(func=run_mathqa_eval)
    
    # Add sub-command to run from configuration file
    config_parser = subparsers.add_parser("config", help="Run from configuration file")
    config_parser.add_argument("config_file", type=str, help="Configuration file path")
    config_parser.add_argument("task", type=str, choices=["slm", "llm", "teacher", "mathqa-eval"], help="Task to run")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command is provided, show help
    if not args.command:
        parser.print_help()
        return
    
    # If running from configuration file
    if args.command == "config":
        run_from_config(args.config_file, args.task)
        return
    
    # Run the corresponding function
    args.func(args)


if __name__ == "__main__":
    main() 