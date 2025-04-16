from openai import OpenAI
import json
import os
import sys
import pandas as pd
import random
client = OpenAI(api_key="sk-222f957d1a354c51ac268b01f978403a", base_url="https://api.deepseek.com")
class PromptBuilder:
    def __init__(self, expert_dfs, train_df, num_examples_per_dataset=2):
        self.expert_dfs = expert_dfs
        self.train_df = train_df
        self.num_examples_per_dataset = num_examples_per_dataset
        self.train_pointer = 0 

    def sample_expert_examples(self):
        sampled_dfs = []
        for df in self.expert_dfs:
            sampled = df.sample(n=self.num_examples_per_dataset, random_state=random.randint(0, 10000))
            sampled_dfs.append(sampled)
        combined_df = pd.concat(sampled_dfs, ignore_index=True)
        return combined_df
    # n: number of target question each prompt has
    def get_next_training_questions(self, n=5):
        if self.train_pointer >= len(self.train_df):
            return []
        end = min(self.train_pointer + n, len(self.train_df))
        questions = self.train_df.iloc[self.train_pointer:end]['query'].tolist()
        self.train_pointer = end 
        return questions

    def build_prompt(self, instruction):
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
    
def generate_mental_health_case(builder, instruction):
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
    
import re
def extract_response_reasoning_plain_text(result_text):

    pattern = r'response":\s*"([^"]+)"\s*,\s*"reasoning":\s*"([^"]+)"'
    matches = re.findall(pattern, result_text)

    return matches 

# num_batches: number of iteration of prompts
def generate_pipeline(builder, instruction, output_file="generated_dataset.csv", num_batches=1):
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
