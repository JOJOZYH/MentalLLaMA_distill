from openai import OpenAI
import json
import os
import sys
import pandas as pd
import random
import re

class PromptBuilder:
    """
    A class for building prompts by combining expert examples and training questions.
    
    This class handles the selection of expert examples from multiple datasets and
    manages the progression through the training dataset to generate prompts for
    model inference.
    """
    
    def __init__(self, expert_dfs, train_df, num_examples_per_dataset=10):
        """
        Initialize the PromptBuilder with expert examples and training data.
        
        Args:
            expert_dfs (list): A list of pandas DataFrames containing expert examples
            train_df (DataFrame): A pandas DataFrame containing training questions
            num_examples_per_dataset (int, optional): Number of examples to sample from each expert dataset. Defaults to 10.
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
        sampled = self.expert_dfs.sample(n=self.num_examples_per_dataset, random_state=random.randint(0, 10000))
        return sampled 
    
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
        batch = self.train_df.iloc[self.train_pointer:end]
        self.train_pointer = end 
        questions = []
        for _, row in batch.iterrows():
            query = row['query']
            full_response = row['gpt-3.5-turbo']
            label_match = re.match(r'^(Yes|No)', full_response.strip(), re.IGNORECASE)
            response_label = label_match.group(1) if label_match else "Unknown"
            reasoning_split = re.split(r'\s*Reasoning\s*:\s*', full_response, maxsplit=1, flags=re.IGNORECASE)
            response_before_reasoning = reasoning_split[0].strip() if reasoning_split else full_response
            questions.append((query, response_label, response_before_reasoning))
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
            prompt += f"Query: {row['query']}\n"
            prompt += f"Response: {row['gpt-3.5-turbo']}\n\n"

        prompt += "\n### Now provide the reasoning for the following queries and responses in the format: {\"reasoning\": \"...\"}\n"
        for i, (query, response_label, response_before_reasoning) in enumerate(target_questions, 1):
            prompt += f"Q{i}: {query}\nResponse: {response_label}\n"

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
    pattern = r'\s*"reasoning":\s*"([^"]+)"'
    matches = re.findall(pattern, result_text)

    return matches 

def generate_pipeline(builder, instruction, output_file=None, failed_output_file=None, client=None, num_batches=1):
    if not output_file or not failed_output_file:
        raise ValueError("Output file paths cannot be empty.")
    if not client:
        raise ValueError("API client cannot be None.")
    
    rows = []
    failed_rows = []

    for i in range(num_batches):
        print(f"Generating batch {i+1}/{num_batches}...")
        result_text, queries = generate_mental_health_case(builder, instruction, client)

        if not queries:
            print("Reached end of training dataset. Stopping.")
            break

        if result_text and queries:
            reasonings = extract_response_reasoning_plain_text(result_text)

            if len(reasonings) != len(queries):
                print(f"Warning: Mismatch between number of queries ({len(queries)}) and responses ({len(reasonings)})")
                for q in queries:
                    failed_rows.append(q[0])
                continue

            for (query, response_label, response_before_reasoning), reasoning in zip(queries, reasonings):
                combined = f"{response_before_reasoning} Reasoning: {reasoning}"
                rows.append({
                    "query": query,
                    "full_answer": combined
                })

        else:
            print("Skipping this batch due to API or parsing error.")

    # Save successful cases
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} rows to {output_file}")

    # Save failed cases
    if failed_rows:
        failed_df = pd.DataFrame(failed_rows, columns=["query"])
        failed_df.to_csv(failed_output_file, index=False)
        print(f"Saved {len(failed_df)} failed queries to {failed_output_file}")

def main():
    """
    Main function to run the data generation pipeline.
    
    Loads datasets, initializes the PromptBuilder, and starts the generation process
    with the appropriate instruction.
    """
  
    # instructions
    # DR 
    DR_instruction = ''': You will be presented with a post and an assigned label to identify whether the poster
    shows symptoms of depression. Consider the emotions expressed from post to explain the reasoning of the label step by step.
    Here are some examples:'''
    # dreaddit
    dreaddit_instruction = ''': You will be presented with a post and an assigned label to identify whether the poster suffers from stress. Consider the emotions expressed from this post
    to explain the reasoning of the label step by step. Here are some examples:'''
    # Irf
    Irf_instruction = ''': You will be presented with a post an assigned label to identify whether the post shows risk of perceived burdensomeness, considering the interpersonal
    risk factors of mental disturbance in the post. You must consider these information to explain the reasoning of the label step by step. Here are some examples:'''
    # MultiWD
    MultiWD_instruction = ''': You will be presented with a post and an assigned label to identify whether the wellness dimension of spiritual exists in the post, according to
    Dunn's model of psychological wellness. You must consider these information to explain the reasoning of the label step by step. Here are some examples:'''
    # SAD
    SAD_instruction = ''': You will be presented a post that shows stress, and an assigned label to show the cause of the stress from from the following stress causes
    list: School, Financial problem, Family issues, Social relationships, Work, Health issues, Emotional turmoil, Everyday decision making,
    Other causes. You must explain the reasoning of the assigned label step by step. Here are some examples:'''


    builder = PromptBuilder(df_DR, DR_train)
    myclient = OpenAI(api_key="", base_url="")
    generate_pipeline(builder, DR_instruction, output_file="generated_dataset.csv", failed_output_file = "failed_generate.csv",client = myclient, num_batches=1)
if __name__ == "__main__":
    main()
