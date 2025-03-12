import csv
import random
import io
import os
import pandas as pd

def update_prompt_with_csv(prompt_path, csv_valid_path, csv_train_path, n):
    # Function to read CSV and return rows
    def read_csv_rows(csv_path):
        rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip header, then add rows
            next(reader, None)
            for row in reader:
                # Only add non-empty rows
                if any(field.strip() for field in row):
                    rows.append(row)
        return rows

    # Get lists of rows from both CSV files
    valid_rows = read_csv_rows(csv_valid_path)
    train_rows = read_csv_rows(csv_train_path)

    # Combine rows from both files
    combined_rows = valid_rows + train_rows

    # Sample n rows randomly from the combined list without replacement
    sample_combined = random.sample(combined_rows, min(n, len(combined_rows)))

    # Format the rows in the same style as in prompt_complete_DR.txt:
    # example: "Post: ..."," Question: ...", "yes. Reasoning: ..."
    formatted_lines = []
    for row in sample_combined:
        formatted_line = ",".join([f'"{field.strip()}"' for field in row])
        formatted_lines.append(formatted_line)

    # Now update prompt_complete_DR.txt:
    with open(prompt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find "Example data:" marker and remove everything that follows it.
    marker = "Example data:"
    if marker in content:
        # Keep everything up to (and including) the marker line.
        before, _sep, _after = content.partition(marker)
        new_content = before.rstrip() + "\n" + marker + "\n"
    else:
        # If the marker is not found, append it at the end.
        new_content = content.rstrip() + "\n" + marker + "\n"

    # Append the formatted CSV rows
    new_content += "\n".join(formatted_lines) + "\n"

    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"Refreshed '{prompt_path}' with {len(sample_combined)} new example rows.")

def generate_data(*args, iterations=1):
    """
    Unpacks variables from args in the following order:
    n, prompt_path, csv_valid_path, csv_train_path, output_csv, header, client_ds

    iterations: number of iterations to run.
    """
    if len(args) < 7:
        raise ValueError("Expected at least 7 arguments: n, prompt_path, csv_valid_path, csv_train_path, output_csv, header, client_ds")
        
    n, prompt_path, csv_valid_path, csv_train_path, output_csv, header, client_ds = args[:7]

    for i in range(iterations):
        try:
            # Update & read the prompt with new CSV data
            update_prompt_with_csv(prompt_path, csv_valid_path, csv_train_path, n)

            with open(prompt_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Generate data using the updated prompt
            response = client_ds.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a mental health data generation assistant"},
                    {"role": "user", "content": content},
                ],
                stream=False,
                temperature=1.5,
                max_tokens=8192,
                presence_penalty=1.0
            )

            # Get the response string containing CSV data.
            temp_csv_string = response.choices[0].message.content

            # Split the CSV string into individual lines.
            lines = temp_csv_string.strip().splitlines()
            valid_rows = []
            for li in lines:
                try:
                    # Parse the single line using csv.reader.
                    reader = csv.reader([li])
                    row = next(reader)
                    if len(row) == 3:
                        valid_rows.append(row)
                    else:
                        print(f"Iteration {i}: Skipping line (invalid number of columns): {li}")
                except Exception as e:
                    print(f"Iteration {i}: Error parsing line: {e}. Skipping line: {li}")
                    continue

            # If no valid rows were found, skip this iteration.
            if not valid_rows:
                print(f"Iteration {i}: No valid CSV rows found. Skipping iteration.")
                continue

            # Build a temporary DataFrame with the provided header.
            temp_df = pd.DataFrame(valid_rows, columns=header)

            # Append the new valid rows to the global CSV.
            if os.path.exists(output_csv) and os.stat(output_csv).st_size > 0:
                temp_df.to_csv(output_csv, mode='a', index=False, header=False)
            else:
                temp_df.to_csv(output_csv, mode='w', index=False, header=True)
            print(f"Iteration {i}: CSV file '{output_csv}' updated with {len(temp_df)} new rows of data.")
                
        except Exception as e:
            print(f"Error occurred in iteration {i}: {e}")
            continue