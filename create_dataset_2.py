import agi
import os
import random
import json
import re
from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd

# # Load the tokenizer - you can replace this with your specific model tokenizer
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# Initialize AGI client
client = agi.Client("FFaFFFp7bVnRY95NGh8uf8VM5SlRmeNOALJmdVtRlJ0")

def add_thinking_level(dataset):
    """Add a random thinking level (1-10) to each example in the dataset."""
    return dataset.map(lambda x: {"thinking_level": random.randint(1, 10)})

def generate_prompt_with_thinking(problem, solution, thinking, thinking_level):
    """
    Generate a prompt with thinking structure similar to generate_r1_prompt.
    
    Args:
        problem: The math problem description
        solution: The solution to the problem
        thinking: The thinking/reasoning process
        thinking_level: Level of thinking (1-10) to control thinking depth
        
    Returns:
        Dictionary with prompt, target, and thinking components
    """
    # Create the system and user messages
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
        },
        { 
            "role": "user",
            "content": f"{problem} Thinking level for this question is: {thinking_level}. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags. Think step by step inside <think> tags. The magnitude of thinking is controlled by a thinking level from 1 - 10, where 1 is no thinking and 10 is large thinking time i.e if you are given level is 1 then you should only think for 1 or 2 sentences and if you given level is 10 then you can think for a lot more sentences and perform a longer reasoning for your answer."
        }]
    
    # Create the expected full response (thinking + answer)
    full_response = f"Let me solve this step by step.\n<think>{thinking}</think>\n<answer>{solution}</answer>"
    
    return {
        "prompt": messages,
        "problem": problem,
        "target": solution,
        "thinking_response": thinking,
        "full_response": full_response,
        "thinking_level": thinking_level,
    }

def process_agi_data(data_df, num_examples=100):
    """
    Process the data from AGI client into the required format.
    
    Args:
        data_df: DataFrame containing the AGI data
        num_examples: Number of examples to include (max)
        
    Returns:
        List of processed examples
    """
    processed_data = []
    
    # Limit to the specified number of examples
    data_df = data_df.head(num_examples)
    data_df = data_df[data_df['verifier_score'] == 1]
    
    for _, row in data_df.iterrows():
        # Extract problem, solution, and reasoning
        problem = row.get('question', '')
        solution = row.get('answer_content', '')
        
        # Extract thinking/reasoning - this will depend on your actual data structure
        # Adjust these fields based on what's available in your AGI data
        thinking = row.get('reasoning_content', '')
        if not thinking and 'trace' in row:
            thinking = row['trace']
        if not thinking:
            thinking = "Let me think through this problem step by step."
        
        # Assign a random thinking level
        thinking_level = random.randint(1, 10)
        
        # Generate the formatted example
        example = generate_prompt_with_thinking(problem, solution, thinking, thinking_level)
        processed_data.append(example)
    
    return processed_data

def create_dataset(save_path="equation_dataset.json"):
    """
    Create a dataset using AGI client data.
    
    Args:
        save_path: Path to save the dataset
        
    Returns:
        The created dataset
    """
    # Download data from AGI client
    print("Downloading data from AGI client...")
    data = client.data.get(task='high-school-math', model='DeepSeek-R1')
    data_df = pd.read_json(data, lines=True)
    
    print(f"Downloaded {len(data_df)} examples from AGI client")
    
    # Process the data
    processed_data = process_agi_data(data_df)
    
    # Save the dataset
    with open(save_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(processed_data)
    
    return dataset, data_df

if __name__ == "__main__":
    # Create the dataset
    dataset, raw_data = create_dataset(save_path="equation_dataset.json")
    print(f"Created dataset with {len(dataset)} examples")
    
    # Display a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print("\nSample prompt:")
        print(sample["prompt"])
        print("\nSample thinking response:")
        print(sample["thinking_response"])
        print("\nSample target (solution):")
        print(sample["target"])
        print("\nSample full response:")
        print(sample["full_response"])
    
    # Print the raw data columns for reference
    print("\nRaw data columns:")
    print(raw_data.columns.tolist())






