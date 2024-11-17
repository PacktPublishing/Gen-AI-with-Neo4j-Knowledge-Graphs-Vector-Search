# Import libraries
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import gc

# Load the GitHub issues dataset
issues_dataset = load_dataset("lewtun/github-issues", split="train")
issues_dataset = issues_dataset.filter(lambda x: not x["is_pull_request"] and len(x["comments"]) > 0)

# Keep only necessary columns
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(issues_dataset.column_names) - set(columns_to_keep)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)

# Explode comments and preprocess
issues_dataset.set_format("pandas")
df = issues_dataset[:]
comments_df = df.explode("comments", ignore_index=True)
comments_dataset = Dataset.from_pandas(comments_df)
comments_dataset = comments_dataset.map(lambda x: {"comment_length": len(x["comments"].split())}, num_proc=1)
comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)

# Concatenate text fields
def concatenate_text(examples):
    return {"text": examples["title"] + " \n " + examples["body"] + " \n " + examples["comments"]}

comments_dataset = comments_dataset.map(concatenate_text, num_proc=1)

# Load smaller model
model_ckpt = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to("cpu")

# Embedding function
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to("cpu")
    with torch.no_grad():
        model_output = model(**encoded_input)
    return cls_pooling(model_output).numpy()

# Compute embeddings
comments_dataset = comments_dataset.map(
    lambda batch: {"embeddings": [get_embeddings([text])[0] for text in batch["text"]]},
    batched=True,
    batch_size=100,
    num_proc=1
)
gc.collect()  # Clear memory after processing

# Perform semantic search using cosine similarity
question = "How can I load a dataset offline?"
query_embedding = get_embeddings([question]).reshape(1, -1)
embeddings = np.vstack(comments_dataset["embeddings"])
similarities = cosine_similarity(query_embedding, embeddings).flatten()

# Get top results
top_indices = np.argsort(similarities)[::-1][:5]
for idx in top_indices:
    result = comments_dataset[int(idx)]  # Convert NumPy integer to native Python integer
    print(f"COMMENT: {result['comments']}")
    print(f"SCORE: {similarities[idx]}")
    print(f"TITLE: {result['title']}")
    print(f"URL: {result['html_url']}")
    print("=" * 50)
