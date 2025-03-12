# DISLM

A framework for detecting and correcting reasoning errors in language models for mathematical problem solving.

## Overview

DISLM is a comprehensive framework designed to identify and correct reasoning errors made by language models when solving mathematical problems. The framework implements a multi-stage correction process:

1. **Error Detection**: Using a Small Language Model (SLM) to identify errors in mathematical reasoning
2. **Error Correction**: Using a Large Language Model (LLM) to correct the identified errors
3. **Verification**: Using a Teacher model to verify the corrections

The project supports various mathematical datasets including GSM8K, MathQA, ASDiv, and LogiQA, and works with different language models such as Qwen2 and Llama-3.1.

## Project Structure

```
DISLM/
├── core/                # Core functionality modules
│   ├── config.py        # Configuration and parameter parsing
│   ├── data_utils.py    # Data processing utilities
│   ├── eval_utils.py    # Evaluation utilities
│   ├── models.py        # Model implementations
│   └── prompts.py       # Prompt templates
├── data/                # Data storage directory
├── models/              # Model storage directory
├── scripts/             # Utility scripts
│   └── mathqa_eval.py   # Standalone MathQA evaluation script
├── dislm.py             # Main entry script
├── README.md            # This file
└── requirements.txt     # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/DISLM.git
cd DISLM
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

DISLM provides a unified command-line interface that can execute various tasks through different sub-commands.

### Basic Usage

```bash
python dislm.py <command> [options]
```

### Available Commands

- `slm`: Generate corrections using Small Language Model
- `llm`: Generate corrections using Large Language Model
- `teacher`: Verify corrections using Teacher model
- `multichoice`: Process multiple choice questions
- `process`: Process evaluation results
- `self-correction`: Process self-correction data
- `balance`: Balance dataset
- `filter`: Filter correction results
- `prepare`: Prepare training data
- `mathqa-eval`: Evaluate models on MathQA dataset using few-shot prompting
- `config`: Run tasks from configuration file

### Examples

#### Generate Corrections Using Small Language Model

```bash
python dislm.py slm \
  --model_path /path/to/model \
  --file_path /path/to/input.json \
  --output_dir /path/to/output.json \
  --multichoice 0
```

#### Generate Corrections Using Large Language Model

```bash
python dislm.py llm \
  --api_key YOUR_API_KEY \
  --base_url https://api.example.com/v1 \
  --model model-name \
  --temperature 0.7 \
  --file_path /path/to/slm_output.json \
  --output_dir /path/to/llm_output.json \
  --num_workers 4 \
  --multichoice 0
```

#### Verify Corrections Using Teacher Model

```bash
python dislm.py teacher \
  --api_key YOUR_API_KEY \
  --base_url https://api.example.com/v1 \
  --model model-name \
  --temperature 0.7 \
  --file_path /path/to/llm_output.json \
  --output_dir /path/to/teacher_output.json \
  --num_workers 4 \
  --multichoice 0
```

#### Process Multiple Choice Questions

```bash
python dislm.py multichoice \
  --model_path /path/to/model \
  --input_file /path/to/input.json \
  --output_file /path/to/output.json \
  --gpu_ids 0,1,2,3
```

#### Process Self-Correction Data

```bash
python dislm.py self-correction \
  /path/to/input.json \
  /path/to/output.json
```

#### Evaluate Models on MathQA Dataset

```bash
python dislm.py mathqa-eval \
  --model_path /path/to/model \
  --output_path /path/to/output/results.json \
  --eval_output_path /path/to/output/evaluation.json \
  --temperature 0.5 \
  --gpu_visible 0,1,2,3
```

#### Run Tasks From Configuration File

```bash
python dislm.py config config.json slm
```

## Workflow

1. **Data Preparation**: Prepare mathematical problems with original solutions and gold answers
2. **SLM Correction**: Run SLM to identify errors in the original solutions
3. **LLM Correction**: Use LLM to correct the identified errors
4. **Teacher Verification**: Verify the corrections using the Teacher model
5. **Evaluation**: Evaluate the correction quality using the evaluation scripts

## Datasets

The framework supports the following datasets:

- **GSM8K**: Grade school math problems
- **MathQA**: Multiple choice math questions
- **ASDiv**: Arithmetic problems
- **LogiQA**: Logical reasoning problems

## Models

The framework has been tested with the following models:

- **Qwen2-7B** and **Qwen2-7B-Instruct**
- **Llama-3.1-8B**
- Other OpenAI-compatible models via API

## MathQA Evaluation

For MathQA evaluation, we use a few-shot prompting approach instead of lm-evaluation-harness. This approach provides several advantages:

1. **Customized Prompting**: Allows for tailored few-shot examples that better guide the model
2. **Flexible Evaluation**: Supports both option letter (a, b, c, d, e) and numerical answer extraction
3. **Detailed Analysis**: Provides comprehensive evaluation results with per-example correctness

### Few-shot Template

The evaluation uses a 4-shot template with carefully selected examples:

```
The following are multiple choice questions (with answers) about mathematical reasoning. 
Choose a correct answer that appears in the candidate answers. 
Your response should conclude with the format "Therefore, the answer is \boxed{answer}".

Q: John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is?
Answer Choices: a ) 50 , b ) 45 , c ) 65 , d ) 78 , e ) 64
A: If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50.
Therefore, the answer is \boxed{a}.

[3 more examples...]

Q: {Problem}
Answer Choices: {options}
A: 
```

### Standalone Evaluation Script

For convenience, a standalone evaluation script is also provided:

```bash
python scripts/mathqa_eval.py \
  --model_path /path/to/model \
  --output_dir /path/to/output \
  --model_type base \
  --model_name ModelName \
  --temperature 0.5 \
  --gpu_visible 0,1,2,3
```

## Configuration Files

You can use JSON format configuration files to store common parameters:

```json
{
  "model_path": "/path/to/model",
  "file_path": "/path/to/input.json",
  "output_dir": "/path/to/output.json",
  "temperature": 0.7,
  "multichoice": 0
}
```

Then use the `config` command to run tasks:

```bash
python dislm.py config config.json slm
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{dislm2023,
  author = {Author},
  title = {DISLM: A Framework for Detecting and Correcting Reasoning Errors in Language Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/username/DISLM}}
}
``` 