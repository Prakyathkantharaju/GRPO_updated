import pandas as pd
import datasets
import json
import requests
from tqdm import tqdm
import numpy as np

# GPT_teacher_dataset = 'GPTeacher/Instruct/gpt4-instruct-dedupe-only-dataset.json'
# GPT_role_play = '/home/ubuntu/GRPOruns/GPTeacher/Roleplay/roleplay-simple-deduped-roleplay-instruct.json'
# GPT_toolformer = '/home/ubuntu/GRPOruns/GPTeacher/Toolformer/toolformer-dedupe-only-dataset.json'

# teacher_dataset = json.load(open(GPT_teacher_dataset))
# rool_play_dataset = json.load(open(GPT_rool_play))

print('Down loading the data')

def embed_text(text):
    """
    Sends a POST request to get embeddings for a text or list of texts.
    
    Args:
        text (str or list): The text (or list of texts) to embed.
        
    Returns:
        list: The embeddings returned from the service.
    """
    with requests.post("http://127.0.0.1:8080/embed", json={'inputs': text}) as response:
        response.raise_for_status()
        return response.json()

def process_dataset(dataset, output_dir, batch_size=32):
    """
    Process a dataset: computes embeddings in batches and saves the extended
    dataset (original data + embeddings) in HuggingFace format.
    
    Args:
        dataset (list): List of dictionaries containing at least 'response' 
                        (and 'instruction'; optionally 'input').
        output_dir (str): The output directory where the dataset will be saved.
        batch_size (int): Number of entries to process per batch.
        
    Returns:
        datasets.Dataset: The processed HuggingFace dataset.
    """
    embeddings = []
    for batch_idx in tqdm(range(0, len(dataset), batch_size), desc=f"Embedding {output_dir}"):
        batch_data = dataset[batch_idx:batch_idx+batch_size]
        responses = [item['response'] for item in batch_data]
        embeddings.extend(embed_text(responses))
    
    print("Embeddings shape:", np.array(embeddings).shape)
    
    # Create a new list of dictionary entries which include embeddings
    dataset_entries = []
    for item, embedding in zip(dataset, embeddings):
        entry = {
            'instruction': item['instruction'],
            'input': item.get('input', ''),  # use empty string if 'input' is missing
            'response': item['response'],
            'embedding': embedding
        }
        dataset_entries.append(entry)
    
    # Convert the list to a HuggingFace Dataset and save it
    hf_dataset = datasets.Dataset.from_list(dataset_entries)
    hf_dataset.save_to_disk(output_dir)
    
    print(f"Created dataset with {len(hf_dataset)} entries")
    print(f"Dataset features: {hf_dataset.features}")
    return hf_dataset

if __name__ == "__main__":
    # Define file paths for the teacher and roleplay datasets
    teacher_dataset_path = 'GPTeacher/Instruct/gpt4-instruct-dedupe-only-dataset.json'
    roleplay_dataset_path = '/home/ubuntu/GRPOruns/GPTeacher/Roleplay/roleplay-simple-deduped-roleplay-instruct.json'
    
    # Load datasets
    print("Loading teacher dataset...")
    teacher_dataset = json.load(open(teacher_dataset_path))
    
    print("Loading roleplay dataset...")
    roleplay_dataset = json.load(open(roleplay_dataset_path))
    
    # Process teacher dataset and save the output
    print("Processing teacher dataset...")
    teacher_hf_dataset = process_dataset(teacher_dataset, "teacher_dataset_with_embeddings")
    
    # Process roleplay dataset and save the output
    print("Processing roleplay dataset...")
    roleplay_hf_dataset = process_dataset(roleplay_dataset, "roleplay_dataset_with_embeddings")
    dataset = datasets.load_dataset('yahma/alpaca-cleaned', split='train')
    # Convert to list of dictionaries in the required format
    alpaca_list = [
        {
            'instruction': item['instruction'],
            'input': item['input'],
            'response': item['output']  # Note: Alpaca uses 'output' instead of 'response'
        }
        for item in dataset
    ]

    # Process alpaca dataset and save the output
    print("Processing alpaca dataset...")
    alpaca_hf_dataset = process_dataset(alpaca_list, "alpaca_dataset_with_embeddings")










