# RLSuccSite：Succinylation Sites Prediction Based on Reinforcement Learning Dynamic with Balanced Reward Mechanism and Three-Peaks Enhanced Method for Physicochemical Property Scores

## 1. Environment
```bash
pip install -r requirements.txt
```

## 2. Dataset Preparation
ProtT5 features need to be decompressed before use.

## 3. Usage

### Test
Run the voting model:
```bash
python Models/Vote.py
```

### Train
Run the PPO training scripts:
```bash
python ProtT5/TrainPPO_ProtT5.py
python TPEMPPS_CCP/TrainPPO_TPEMPPS_CCP.py
```

## 4. Inference on New Datasets

To use RLSuccSite for prediction on your own protein sequences, follow this multi-step process:

### Step 1: Extract 33-Residue Fragments
The model classifies 33-residue protein fragments centered on a Lysine (K) residue. Use the following script to generate fragments from your FASTA file:

```python
import sys
from Bio import SeqIO

def extract_fragments(input_fasta, output_fasta, window_size=33):
    half_window = window_size // 2
    with open(output_fasta, "w") as out_f:
        for record in SeqIO.parse(input_fasta, "fasta"):
            seq = str(record.seq)
            for i, res in enumerate(seq):
                if res == 'K':
                    left = max(0, i - half_window)
                    right = min(len(seq), i + half_window + 1)
                    fragment = seq[left:right]
                    
                    # Apply 'X' padding for fragments near sequence ends
                    if i < half_window:
                        fragment = 'X' * (half_window - i) + fragment
                    if (len(seq) - 1 - i) < half_window:
                        fragment = fragment + 'X' * (half_window - (len(seq) - 1 - i))
                    
                    out_f.write(f">{record.id}|{i+1}\n{fragment}\n")

if __name__ == "__main__":
    extract_fragments("your_input.fasta", "extracted_sites.fasta")
```

### Step 2: Feature Generation

#### A. ProtT5 Features
1. **Generate Full Embeddings**: Modify `Feature/ProtT5.py` to point to your `extracted_sites.fasta` and run it. This creates a CSV with 1024-dimensional embeddings per residue.
2. **Extract Center Residue**: Modify `Feature/ProtT5_K.py` to process the CSV from the previous step. It will extract the 16th residue's embedding into a final feature file (e.g., `my_protT5_K.csv`).

#### B. Physicochemical Features (TPEMPPS_CCP)
The scripts in `Feature/` (specifically `TPEMPPS.py`) can process the fragment FASTA files directly.

### Step 3: Prediction
Modify `Models/Vote.py` to load your new feature CSVs and fragment FASTA. The script will perform weighted voting using the pre-trained models and output prediction probabilities to `Models/Data/probabilities.csv`.

---
**Note:** The reinforcement learning implementation refers to the [PyTorch PPO Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html#).
