import pandas as pd
import random
import csv
import os
import json
import time
def generate_few_prompts(path, seed=None):  # Fixed 'seeds' to 'seed' for consistency
    data = pd.read_csv(path)
    sample_data = data.sample(3, random_state=seed)  # Correct seed usage

    prompts = []
    for _, row in sample_data.iterrows():
        prompts.append({
            "query": row[0].split(" Question: ")[0],
            "Question": row[0].split(" Question: ")[1],
            "answer": row[1].split(" Reasoning: ")[0],
            "Reasoning": row[1].split(" Reasoning: ")[1]
        })
    
    return prompts
