import os
import pandas as pd
from datasets import Dataset, DatasetDict

# 1. Create a safe folder to store raw data
os.makedirs("raw_csv_data", exist_ok=True)

urls = {
    "train": "https://huggingface.co",
    "validation": "https://huggingface.co",
    "test": "https://huggingface.co"
}

try:
    dataset_splits = {}
    
    # 2. Download and process each file manually using pandas
    for split, url in urls.items():
        print(f"Downloading {split} split directly...")
        df = pd.read_csv(url, on_bad_lines='skip') # skips any corrupted text rows
        dataset_splits[split] = Dataset.from_pandas(df)
    
    # 3. Combine into a clean Hugging Face Dataset format
    final_dataset = DatasetDict(dataset_splits)
    final_dataset.save_to_disk("empathetic_dialogues_fixed")
    
    print("\nSUCCESS! Dataset saved locally to 'empathetic_dialogues_fixed'.")

except Exception as e:
    print("\nError downloading:", e)