# RLSuccSite

**RLSuccSite** is a protein succinylation site prediction tool that uses **Reinforcement Learning (PPO)** with an ensemble of two complementary models:

- **ProtT5**: 1024-dimensional transformer embeddings from protein sequences
- **TPEMPPS_CCP**: 990-dimensional physicochemical features combining TPEMPPS (Three-Peaks Enhanced Method for Physicochemical Property Scores) with CCP (CKSAAP+CTDC+PAAC)

The ensemble uses weighted voting to produce final predictions with improved accuracy and robustness.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Training Models](#training-models)
- [Testing](#testing)
- [Inference on New Sequences](#inference-on-new-sequences)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [References](#references)

## Installation

### Prerequisites
- Python 3.11+
- [UV](https://github.com/astral-sh/uv) package manager (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd RLSuccSite
```

2. Install dependencies using UV:
```bash
uv sync
```

Alternatively, using pip:
```bash
pip install -r requirements.txt
```

### Download ProtT5 Model

The ProtT5 model files must be downloaded separately:

1. Download the ProtT5 model from [Rostlab's ProtT5](https://github.com/rostlab/ProtT5)
2. Place the model files in the `ProtT5/` directory:
```bash
ProtT5/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
└── spiece.model
```

## Dataset

The dataset should be organized as follows:

```
Dataset/
├── train/
│   ├── fasta/
│   │   ├── train_negative_sites.fasta
│   │   └── train_positive_sites.fasta
│   └── features/
│       ├── train_negative_ProtT5_features_K.csv
│       └── train_positive_ProtT5_features_K.csv
└── test/
    ├── fasta/
    │   ├── test_negative_sites.fasta
    │   └── test_positive_sites.fasta
    └── features/
        ├── test_negative_ProtT5_features_K.csv
        └── test_positive_ProtT5_features_K.csv
```

### Fragment Format

FASTA files should contain **33-residue fragments** (with padding 'X' if needed) centered on lysine (K) residues. The format:
- Header: `>sequence_id|position`
- Sequence: 33 amino acids (including possible 'X' padding)

Example:
```
>XP_123456.1|19
MKTFFVAGLVSAGWTAGEAFHKX... (33 chars total)
```

### ProtT5 Features

ProtT5 features are 1024-dimensional embeddings. They must be:
1. Generated for full sequences using `Feature/ProtT5.py`
2. Processed with `Feature/ProtT5_K.py` to extract the **16th residue** embedding (center of 33-residue window)

## Training Models

### Train ProtT5 Model

```bash
uv run python ProtT5/TrainPPO_ProtT5.py
```

This will:
- Load ProtT5 features from `Dataset/train/features/`
- Train a PPO agent with balanced reward mechanism
- Save the best model to `Models/` with metrics in filename
- Log step-by-step training info to `Data/ProtT5_Steps.csv`

**Training parameters** (configurable in script):
- `total_frames`: 362,555 (5 epochs over 72,511 samples)
- `frames_per_batch`: 10,000
- `num_epochs`: 10
- Learning rate: 3e-4

### Train TPEMPPS_CCP Model

```bash
uv run python TPEMPPS_CCP/TrainPPO_TPEMPPS_CCP.py
```

This will:
- Extract TPEMPPS+CCP features from FASTA files in `Dataset/train/fasta/`
- Train a PPO agent with balanced reward mechanism
- Save the best model to `Models/` with metrics in filename
- Log step-by-step training info to `Data/TPEMPPS_CCP_Steps.csv`

**Training parameters**:
- `total_frames`: 217,533 (3 epochs over 72,511 samples)
- Other hyperparameters same as ProtT5 training

## Testing

To evaluate the ensemble on the test set:

```bash
uv run python Models/Vote.py
```

This script:
- Loads the trained ProtT5 and TPEMPPS_CCP models
- Loads test features from `Dataset/test/`
- Performs ensemble voting (equal weights by default)
- Computes metrics: Balanced Accuracy, MCC, Sensitivity, Specificity, F1-Score
- Saves predictions to `Models/Data/probabilities.csv`
- Saves performance metrics to `Models/Data/performance_metrics.csv`

**Expected Performance** (based on training validation):
- ProtT5: ACC ~71.42%, MCC ~0.351, Sn ~71.91%, Sp ~71.30%
- TPEMPPS_CCP: ACC ~70.83%, MCC ~0.331, Sn ~69.43%, Sp ~71.16%
- Ensemble: Slight improvement over individual models

## Inference on New Sequences

### Quick Start

Use the `Predict.py` script for end-to-end prediction on your own protein sequences:

```bash
uv run python Models/Predict.py \
  --prott5_features path/to/your/prott5_features.csv \
  --fragments_fasta path/to/your/fragments.fasta \
  --output predictions.csv
```

### Step-by-Step Guide

#### Step 1: Prepare Input Data

You need:
1. **FASTA file** with your full protein sequences (`.fasta`)
2. **Extract 33-residue fragments** centered on each lysine (K)

Use this script to extract fragments:

```python
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
                    
                    # Pad with 'X' if near sequence ends
                    if i < half_window:
                        fragment = 'X' * (half_window - i) + fragment
                    if (len(seq) - 1 - i) < half_window:
                        fragment = fragment + 'X' * (half_window - (len(seq) - 1 - i))
                    
                    out_f.write(f">{record.id}|{i+1}\n{fragment}\n")

extract_fragments("your_proteins.fasta", "fragments.fasta")
```

#### Step 2: Generate ProtT5 Features

1. **Generate full sequence embeddings** by modifying `Feature/ProtT5.py`:

```python
# Update these paths in Feature/ProtT5.py:
N_input_file_path = 'path/to/your/negative.fasta'  # if needed
P_input_file_path = 'path/to/your/positive.fasta'  # your fragments.fasta
N_output_file_path = 'path/to/output_negative.csv'
P_output_file_path = 'path/to/output_positive.csv'
```

Then run:
```bash
uv run python Feature/ProtT5.py
```

2. **Extract center residue (16th)** using `Feature/ProtT5_K.py`:

```python
# Update paths in Feature/ProtT5_K.py:
in_P = "output_from_step1.csv"
out_P = "protT5_features_K.csv"
```

Then run:
```bash
uv run python Feature/ProtT5_K.py
```

You'll get a CSV with 1025 columns: first column is ID, next 1024 are features.

#### Step 3: Predict with Pretrained Models

```bash
uv run python Models/Predict.py \
  --prott5_features protT5_features_K.csv \
  --fragments_fasta fragments.fasta \
  --output my_predictions.csv
```

**Output CSV format:**
- `SequenceID`: Fragment identifier (e.g., `XP_123456.1|19`)
- `PositiveProbability`: Probability of succinylation (0-1)
- `PredictedLabel`: Binary prediction (0=negative, 1=positive)

## Model Performance

The models were trained and evaluated using 5-fold cross-validation with balanced reward mechanism. The ensemble approach combines:

| Model | Accuracy | MCC | Sensitivity | Specificity |
|-------|----------|-----|-------------|-------------|
| ProtT5 | 71.42% | 0.351 | 71.91% | 71.30% |
| TPEMPPS_CCP | 70.83% | 0.331 | 69.43% | 71.16% |
| **Ensemble** | **~72%** | **~0.36** | **~73%** | **~71%** |

*Note: Actual performance may vary depending on the test dataset.*

## Project Structure

```
RLSuccSite/
├── Dataset/                    # Training and test data
│   ├── train/
│   └── test/
├── Feature/                    # Feature extraction modules
│   ├── CKSAAP.py              # k-spaced amino acid pairs
│   ├── CTDC.py                # composition/transition/distribution
│   ├── PAAC.py                # pseudo-amino acid composition
│   ├── ProtT5.py              # ProtT5 transformer embeddings
│   ├── ProtT5_K.py            # Extract center residue from ProtT5 output
│   └── TPEMPPS.py             # Three-peaks enhanced physicochemical features
├── Models/                     # Trained models and inference scripts
│   ├── ProtT5_N10_*.pth       # Trained ProtT5 model
│   ├── TPEMPPS_CCP_*.pth      # Trained TPEMPPS_CCP model
│   ├── Vote.py                # Ensemble evaluation on test set
│   └── Predict.py             # Inference on new sequences
├── ProtT5/                     # ProtT5 model files (download separately)
├── TPEMPPS_CCP/               # Training script for TPEMPPS_CCP model
├── Utils/                      # Utility scripts
├── Env.py                      # Custom RL environment with balanced rewards
├── Features.py                 # Combined feature functions
├── TrainPPO_ProtT5.py         # Training script for ProtT5 model
├── TrainPPO_TPEMPPS_CCP.py    # Training script for TPEMPPS_CCP model
├── pyproject.toml             # Project dependencies
└── uv.lock                    # Locked dependencies
```

## Technical Details

### Reinforcement Learning Approach

- **Algorithm**: PPO (Proximal Policy Optimization) from TorchRL
- **Action Space**: Binary classification (succinylation vs. non-succinylation)
- **Reward Mechanism**: Balanced rewards that adapt during training:
  - TP: dynamically weighted to encourage positive class
  - TN: slightly decreasing weight
  - FN: heavily penalized (negative weight)
  - FP: moderately penalized

### Feature Engineering

**ProtT5 (1024-D)**:
- Uses `Rostlab/prot_t5_xl_uniref50` transformer
- Full sequence embedding → extract 16th residue (center of 33-mer)

**TPEMPPS (528-D)**:
- `ZccF_LiHua`: Integer encoding + physicochemical properties with positional weighting
- `ZccF_alltoK`: Central residue-focused features with distance weighting

**CCP (462-D)**:
- CKSAAP: 240-D (k=4)
- CTDC: 39-D  
- PAAC: 50-D (λ=3)

**Total TPEMPPS_CCP**: 528 + 462 = **990 dimensions**

## References

- [PyTorch PPO Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html)
- [ProtT5: Protein Language Model](https://github.com/rostlab/ProtT5)
- [ProtLearn: Protein Feature Extraction](https://github.com/atc123/protlearn)
- Original research paper (citation TBD)

## License

[Add license information here]

## Contact

For issues, questions, or contributions, please open an issue on GitHub.
