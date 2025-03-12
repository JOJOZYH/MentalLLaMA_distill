# import random

# # 读取文本文件
# with open("your_text_file.txt", "r", encoding="utf-8") as file:
#     lines = file.readlines()

# # 确保数据足够
# num_samples = min(100, len(lines))

# # 随机抽取 100 行
# sampled_lines = random.sample(lines, num_samples)

# # 保存到新文件
# with open("sampled_data.txt", "w", encoding="utf-8") as output_file:
#     output_file.writelines(sampled_lines)

# print(f"已成功从 {len(lines)} 行数据中随机抽取 {num_samples} 行，结果保存在 sampled_data.txt")


import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("./data/released_data_mentalllama/test_data/test_instruction/DR.csv")

# 随机抽取 100 行（如果数据不足 100 行，则抽取所有数据）
sampled_df = df.sample(n=min(100, len(df)), random_state=42)  # `random_state` 保证结果可复现

# 保存到新 CSV 文件
sampled_df.to_csv("sampled_data_test_instruction_DR.csv", index=False)

print(f"已成功从 {len(df)} 条数据中随机抽取 {len(sampled_df)} 条，结果保存在 sampled_data.csv")
