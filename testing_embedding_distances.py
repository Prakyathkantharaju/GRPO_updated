from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


model_path = 'Alibaba-NLP/gte-large-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)



def get_embedding(input_text):
    # Tokenize the input texts
    batch_dict = tokenizer(input_text, max_length=8192, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict)
    embeddings = outputs.last_hidden_state[:, 0]
    # (Optionally) normalize embeddings
    return F.normalize(embeddings, p=2, dim=1)



def compute_embedding_distance(sentence1, sentence2):
    # Get embeddings for both sentences
    embedding1 = get_embedding(sentence1)
    embedding2 = get_embedding(sentence2)
    
    # Normalize embeddings
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.mm(embedding1, embedding2.transpose(0, 1))
    
    # Convert to distance (1 - similarity)
    distance = 1 - similarity.item()
    
    return distance

# Example usage
if __name__ == "__main__":
    sentence1 = "The cat sits on the mat"
    sentence2 = "A kitten is resting on a rug"
    
    distance = compute_embedding_distance(sentence1, sentence2)
    print(f"Embedding distance between sentences: {distance:.4f}")
