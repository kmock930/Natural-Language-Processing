import os
DEEPSEEK_MODEL_PATH = os.path.join(os.path.abspath(__file__), "..", "deepseek_model")
import pandas as pd
df_test_pred = pd.read_csv(os.path.join(DEEPSEEK_MODEL_PATH, "test_predictions.csv"))

import json
# Prepare results in the desired format
results = []
for idx, (text, label) in enumerate(zip(df_test_pred['original_text'], df_test_pred['predicted_label'])):
    results.append({
        "id": idx,
        "predicted_label": int(label),
        "raw_text": text
    })

# Define the output file path
output_file_path = os.path.join(DEEPSEEK_MODEL_PATH, "..", "..", "Results", "Result_llm_deepseek.jsonl")

# Save the results to a JSONL file
with open(output_file_path, "w") as jsonl_file:
    for record in results:
        jsonl_file.write(json.dumps(record) + "\n")

print(f"Results saved to {output_file_path}")
