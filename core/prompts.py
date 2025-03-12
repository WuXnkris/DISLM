#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
prompts.py - Centralized management of all prompt templates
"""

def SLM_get_prompt_correct(example):
    """
    Generate correction prompt for Small Language Model
    
    Args:
        example (dict): Example containing question, original solution, and gold answer
        
    Returns:
        str: Formatted prompt
    """
    prompt = """As an intelligent assistant specializing in reasoning tasks, your role is to analyze the given question and the original solution. Identify any mistakes in the reasoning process, explain why these errors occurred, and then provide the correct reasoning path.
Question:
Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?
Original Solution:
Step 1: Identify the number of yellow flowers, which is 10. Step 2: Calculate the number of purple flowers, which is 80% more than the yellow flowers: 10 x 1.8 = 18. Step 3: Calculate the number of green flowers, which is 25% of the total number of yellow and purple flowers: (10 + 18) x 0.25 = 14. Step 4: Add the number of yellow, purple, and green flowers to find the total number of flowers: 10 + 18 + 14 = 42. The answer is 42.
Gold Answer:
There are 80/100 * 10 = <<80/100*10=8>>8 more purple flowers than yellow flowers.\nSo in Mark's garden, there are 10 + 8 = <<10+8=18>>18 purple flowers.\nPurple and yellow flowers sum up to 10 + 18 = <<10+18=28>>28 flowers.\nThat means in Mark's garden there are 25/100 * 28 = <<25/100*28=7>>7 green flowers.\nSo in total Mark has 28 + 7 = <<28+7=35>>35 plants in his garden.
Mistake Analysis: 
The original solution incorrectly calculated the number of green flowers. Green flowers are supposed to be 25% of the total number of yellow and purple flowers, but the calculation incorrectly used a higher number for the green flowers, resulting in an overestimation.
The issue lies specifically in Step 3, where the correct calculation should be: (10+18) × 0.25 = 7 instead of 14.
Correct Solution:
Step 1: Identify the number of yellow flowers, which is 10. Step 2: Calculate the number of purple flowers, which is 80% more than the yellow flowers: 10 x 1.8 = 18. Step 3: Calculate the number of green flowers, which is 25% of the total number of yellow and purple flowers: (10 + 18) x 0.25 = 7. Step 4: Add the number of yellow, purple, and green flowers to find the total number of flowers: 10 + 18 + 7 = 35. The answer is 35.

Question:
Carly is trying to get in shape to try out for the soccer team. She starts by running 2 miles a week. The second week, she runs twice as long plus 3 extra miles per week. The third week she runs 9/7 as much as she ran the second week. The week after that, she sprains her ankle and has to reduce her running time by 5 miles per week. How many miles did she run the week she was injured?
Original Solution:
Step 1: Identify the initial amount of miles Carly ran the first week, which is 2 miles. Step 2: Calculate the amount she ran the second week, which is 2 times the first week plus 3 extra miles: 2 + 3 = 5. Step 3: Calculate the amount she ran the third week, which is 9/7 times the amount she ran the second week: 5 * 9/7 = 6.25. Step 4: Calculate the amount she ran the fourth week, which is the amount she ran the third week minus 5 miles: 6.25 - 5 = 1.25. The answer is 1.25.
Gold Answer:
First find how long Carly runs the second week: 2 miles * 2 + 3 = <<2*2+3=7>>7 miles.\nThen find how long she runs in the third week: 9/7 * 7 miles = <<9/7*7=9>>9 miles.\nThen subtract the 5 miles reduction from when she was injured: 9 miles - 5 miles = <<9-5=4>>4 miles.
Mistake Analysis:
The original solution incorrectly calculated the miles Carly ran during the second week in Step 2. The problem states that the second week's distance is twice as long as the first week plus 3 extra miles, but the solution only added 3 miles to the first week's distance instead of doubling it first.
Specifically: Step 2 should have been: 2 * 2 + 3 = 4 + 3 = 7
Correct Solution:
Step 1: Identify the initial amount of miles Carly ran the first week, which is 2 miles. Step 2: Calculate the amount she ran the second week, which is 2 times the first week plus 3 extra miles: 2 * 2 + 3 = 7. Step 3: Calculate the amount she ran the third week, which is 9/7 times the amount she ran the second week: 7 * 9/7 = 9. Step 4: Calculate the amount she ran the fourth week, which is the amount she ran the third week minus 5 miles: 9 - 5 = 4 miles. The answer is 4.

Question:
{question}
Original solution:
{original_solution}
Gold Answer:
{gold_answer}
Mistake Analysis:"""
    
    return prompt.format(
        question=example.get("question", ""),
        original_solution=example.get("original solution", ""),
        gold_answer=example.get("gold answer", "")
    )


def LLM_correct_prompt(example, is_multichoice=0):
    """
    Generate correction prompt for Large Language Model
    
    Args:
        example (dict): Example containing question, original solution, and gold answer
        is_multichoice (int, optional): Flag indicating if the task is multiple choice. Defaults to 0.
    
    Returns:
        str: Formatted prompt
    """
    # Same prompt template for both multiple choice and open-ended questions
    prompt = """You will be provided with a question and an original solution. Your task is to identify any mistakes in the reasoning process, explain why these errors occurred, and then provide the correct reasoning path. Please provide the mistake analysis and correct solution as the format below:

Mistake Analysis: <Find the mistake and analyze the reason of this mistake>
Correct Solution: <Correct the mistake and provide the correct solution>

Question: {question}
Original Solution: {original_solution}
Gold answer: {gold_answer}

Please analyze the original solution, identify any mistakes, and provide the correct solution."""
    
    # Format the prompt with example data
    formatted_prompt = prompt.format(
        question=example.get("question", ""),
        original_solution=example.get("original solution", ""),
        gold_answer=example.get("gold answer", "")
    )
    
    return formatted_prompt


def Teacher_correct_prompt(example, is_multichoice=0):
    """
    Generate verification prompt for Teacher model
    
    Args:
        example (dict): Example containing question, original solution, gold answer, and SLM correction
        is_multichoice (int, optional): Flag indicating if the task is multiple choice. Defaults to 0.
    
    Returns:
        str: Formatted prompt
    """
    # Same prompt template for both multiple choice and open-ended questions
    prompt = """You are a teacher evaluating a student's solution to a math problem. You will be provided with:
1. The original question
2. The original incorrect solution
3. The correct answer
4. A correction attempt by a language model

Your task is to evaluate the language model's correction and provide your own analysis. Please follow this format:

LLM solution: <Evaluate the language model's correction. Is it accurate? Does it correctly identify and fix the mistakes?>
Teacher solution: <Provide your own correct solution, explaining the reasoning step by step>

Question: {question}
Original Solution: {original_solution}
Gold Answer: {gold_answer}
LLM Correction: {slm_correct}

Please evaluate the LLM's correction and provide your own solution."""
    
    # Format the prompt with example data
    formatted_prompt = prompt.format(
        question=example.get("question", ""),
        original_solution=example.get("original solution", ""),
        gold_answer=example.get("gold answer", ""),
        slm_correct=example.get("SLM correct", "")
    )
    
    return formatted_prompt


# Example prompts for different types of math problems
def get_example_prompts():
    """
    Get example prompts for different types of math problems
    
    Returns:
        dict: Dictionary of example prompts
    """
    examples = {
        "arithmetic": {
            "question": "If John has 5 apples and Mary gives him 3 more, how many apples does John have?",
            "original_solution": "John has 5 apples. Mary gives him 3 more. So John has 5 + 3 = 9 apples.",
            "gold_answer": "8"
        },
        "algebra": {
            "question": "If 2x + 5 = 11, what is the value of x?",
            "original_solution": "2x + 5 = 11\n2x = 11 - 5\n2x = 6\nx = 6/2\nx = 3",
            "gold_answer": "3"
        },
        "word_problem": {
            "question": "A train travels at 60 miles per hour. How far will it travel in 2.5 hours?",
            "original_solution": "Distance = Speed × Time\nDistance = 60 × 2.5\nDistance = 150 miles",
            "gold_answer": "150 miles"
        }
    }
    
    # Generate prompts for each example
    example_prompts = {}
    for problem_type, example in examples.items():
        example_dict = {
            "question": example["question"],
            "original solution": example["original_solution"],
            "gold answer": example["gold_answer"]
        }
        example_prompts[problem_type] = {
            "slm": SLM_get_prompt_correct(example_dict),
            "llm": LLM_correct_prompt(example_dict),
            "teacher": Teacher_correct_prompt(example_dict)
        }
    
    return example_prompts


if __name__ == "__main__":
    # Example usage
    example = {
        "question": "If a rectangle has a length of 8 cm and a width of 5 cm, what is its area?",
        "original solution": "The area of a rectangle is calculated by multiplying its length by its width. So, the area is 8 × 5 = 45 cm².",
        "gold answer": "40 cm²",
        "SLM correct": "Mistake Analysis: The original solution incorrectly calculates the product of 8 and 5. The correct calculation is 8 × 5 = 40, not 45.\nCorrect Solution: The area of a rectangle is calculated by multiplying its length by its width. So, the area is 8 × 5 = 40 cm²."
    }
    
    print("SLM prompt example:")
    print(SLM_get_prompt_correct(example))
    print("\nLLM prompt example:")
    print(LLM_correct_prompt(example))
    print("\nTeacher prompt example:")
    print(Teacher_correct_prompt(example)) 