import csv
import random

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

    print(f"Updated '{prompt_path}' with {len(sample_combined)} new example rows.")