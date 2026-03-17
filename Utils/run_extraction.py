import os
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).resolve().parent.parent

from Feature.ProtT5 import process_sequences_file

# Define paths relative to project root
input_file = BASE_DIR / "Dataset/d.catenatum/extracted_sites.fasta"
output_file = BASE_DIR / "Dataset/d.catenatum/extracted_sites_ProtT5_features.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Note: Feature/ProtT5.py hardcodes paths internally.
# To use your specific file, you can call its processing function directly:
print(f"Starting extraction for {input_file}...")
process_sequences_file(input_file, output_file, start_index=0)
print(f"Finished! Results saved to {output_file}")
