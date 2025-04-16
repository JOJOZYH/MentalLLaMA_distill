from openai import OpenAI
import json
import os
import sys
import pandas as pd
import random
import re

# client = OpenAI(api_key="sk-3027dc386bba4840abd9a5b16fcab47e", base_url="https://api.deepseek.com")

class PromptBuilder:
    """
    A class for building prompts by combining expert examples and training questions.
    
    This class handles the selection of expert examples from multiple datasets and
    manages the progression through the training dataset to generate prompts for
    model inference.
    """
    
    def __init__(self, expert_dfs, train_df, num_examples_per_dataset=2):
        """
        Initialize the PromptBuilder with expert examples and training data.
        
        Args:
            expert_dfs (list): A list of pandas DataFrames containing expert examples
            train_df (DataFrame): A pandas DataFrame containing training questions
            num_examples_per_dataset (int, optional): Number of examples to sample from each expert dataset. Defaults to 2.
        """
        self.expert_dfs = expert_dfs
        self.train_df = train_df
        self.num_examples_per_dataset = num_examples_per_dataset
        self.train_pointer = 0 

    def sample_expert_examples(self):
        """
        Sample a fixed number of examples from each expert dataset.
        
        Returns:
            DataFrame: A combined DataFrame containing sampled examples from all expert datasets
        """
        sampled_dfs = []
        for df in self.expert_dfs:
            sampled = df.sample(n=self.num_examples_per_dataset, random_state=random.randint(0, 10000))
            sampled_dfs.append(sampled)
        combined_df = pd.concat(sampled_dfs, ignore_index=True)
        return combined_df
    
    def get_next_training_questions(self, n=5):
        """
        Get the next batch of training questions from the training dataset.
        
        Args:
            n (int, optional): Number of questions to retrieve. Defaults to 5.
            
        Returns:
            list: A list of question strings, empty if end of dataset is reached
        """
        if self.train_pointer >= len(self.train_df):
            return []
        end = min(self.train_pointer + n, len(self.train_df))
        questions = self.train_df.iloc[self.train_pointer:end]['query'].tolist()
        self.train_pointer = end 
        return questions

    def build_prompt(self, instruction):
        """
        Build a complete prompt combining instruction, expert examples, and target questions.
        
        Args:
            instruction (str): The instruction to guide the model response
            
        Returns:
            tuple: (prompt, target_questions) where prompt is the full prompt string and
                  target_questions is a list of questions to be answered
        """
        examples_df = self.sample_expert_examples()
        target_questions = self.get_next_training_questions()

        prompt = instruction.strip() + "\n\n"
        prompt += "### Expert-written examples:\n"

        for _, row in examples_df.iterrows():
            example_json = {
                "query": row['query'],
                "answer": row['gpt-3.5-turbo']
            }
            prompt += json.dumps(example_json, ensure_ascii=False) + "\n"

        prompt += "\n### Now answer the following queries in the following format: {\"response\": \"...\", \"reasoning\": \"...\"}\n"
        for i, q in enumerate(target_questions, 1):
            prompt += f"Q{i}: {q}\n"

        return prompt, target_questions
    
def generate_mental_health_case(builder, instruction, client):
    """
    Generate mental health case responses using an AI model.
    
    Args:
        builder (PromptBuilder): The prompt builder to construct prompts
        instruction (str): Instruction for the AI model
        client: The API client used to communicate with the AI model
        
    Returns:
        tuple: (results, queries) where results is the model's response text and
              queries is the list of questions that were asked
    """
    prompt, queries = builder.build_prompt(instruction)
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens= 1500)
        results = response.choices[0].message.content.strip()
        return results, queries
    except Exception as e:
        print(f"Generation failed: {str(e)}")
        return None, None
    

def extract_response_reasoning_plain_text(result_text):
    """
    Extract response and reasoning pairs from the result text using regex.
    
    Args:
        result_text (str): The text response from the AI model
        
    Returns:
        list: A list of (response, reasoning) tuples extracted from the text
    """
    pattern = r'response":\s*"([^"]+)"\s*,\s*"reasoning":\s*"([^"]+)"'
    matches = re.findall(pattern, result_text)

    return matches 

def generate_pipeline(builder, instruction, output_file="generated_dataset.csv", num_batches=1):
    """
    Run the complete generation pipeline for multiple batches of questions.
    
    Args:
        builder (PromptBuilder): The prompt builder to construct prompts
        instruction (str): Instruction for the AI model
        output_file (str, optional): Path to save the output CSV. Defaults to "generated_dataset.csv".
        num_batches (int, optional): Number of batches to generate. Defaults to 1.
    """
    rows = []
    failed_rows = []
    for i in range(num_batches):
        print(f"Generating batch {i+1}/{num_batches}...")
        result_text, queries = generate_mental_health_case(builder, instruction)
        if not queries:
            print("Reached end of training dataset. Stopping.")
            break
        if result_text and queries:
            pairs = extract_response_reasoning_plain_text(result_text)
            if len(pairs) != len(queries):
                print(f"Warning: Mismatch between number of queries ({len(queries)}) and responses ({len(pairs)})")
                failed_rows.extend(queries)
                continue
            for query, (response, reasoning) in zip(queries, pairs):
                rows.append({
                    "query": query,
                    "response": response,
                    "reasoning": reasoning
                })

        else:
            print("Skipping this batch due to API or parsing error.")

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\nSaved {len(df)} rows to {output_file}")
    if failed_rows:
        failed_df = pd.DataFrame(failed_rows, columns=["query"])
        failed_df.to_csv("failed_generate.csv", index=False)
        print(f"Saved {len(failed_df)} failed queries to failed_generate.csv")


def main():
    """
    Main function to run the data generation pipeline.
    
    Loads datasets, initializes the PromptBuilder, and starts the generation process
    with the appropriate instruction.
    """
    # expert examples
    df_DR = pd.read_csv('../data/expert_data/DR.csv')
    df_dreaddit = pd.read_csv('../data/expert_data/dreaddit.csv')
    df_Irf = pd.read_csv('../data/expert_data/Irf.csv')
    df_MultiWD = pd.read_csv('../data/expert_data/MultiWD.csv')
    df_SAD = pd.read_csv('../data/expert_data/SAD.csv')

    DR_train = pd.read_csv('../data/released_data_mentalllama/train_data/instruction_data/DR/train.csv')

   

    # instructions
    # DR 
    instruction = ''': You will be presented with a post and an assigned label to identify whether the poster
    shows symptoms of depression. Consider the emotions expressed from post to explain the reasoning of the label step by step.
    Here are twenty examples:'''

    datasets = [
        df_DR,
        df_dreaddit,
        df_Irf,
        df_MultiWD,
        df_SAD
    ]
    builder = PromptBuilder(datasets, DR_train)
    generate_pipeline(builder, instruction, output_file="generated_dataset.csv", num_batches=10)

if __name__ == "__main__":
    main()
