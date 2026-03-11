# RLSuccSite Workspace Instructions

**Project**: Succinylation Sites Prediction using Reinforcement Learning with PPO  
**Language**: Python 3.11+  
**Framework**: PyTorch, TorchRL, Reinforcement Learning (PPO)  
**Key Domain**: Bioinformatics - Protein Post-Translational Modification (PTM) Prediction

## Quick Start

### Environment Setup
```bash
pip install -r requirements.txt
# Or use uv package manager:
uv sync
```

### Running the Project
- **Test (Inference)**: `python Models/Vote.py` - Ensemble voting prediction on test dataset
- **Train ProtT5**: `python ProtT5/TrainPPO_ProtT5.py` - Train ProtT5-based feature model
- **Train TPEMPPS_CCP**: `python TPEMPPS_CCP/TrainPPO_TPEMPPS_CCP.py` - Train TPEMPPS_CCP-based feature model

## Project Architecture

### Core Components

1. **Env.py** - Custom RL Environment
   - `PPOEnvZcc`: TorchRL environment implementing succinylation site classification
   - Implements balanced reward mechanism that scales rewards over training duration
   - Tracks TP/TN/FP/FN metrics during training

2. **Feature Extraction** (Features.py and Feature/ folder)
   - Multiple feature extraction methods:
     - **ProtT5_K**: Pre-computed ProtT5 embeddings (CSV format)
     - **TPEMPPS_CCP**: Three-Peaks method combining CKSAAP, CTDC, PAAC
     - **CKSAAP**: Composition and K-Spaced Amino Acid Pair
     - **CTDC**: Composition, Transition, Distribution indices
     - **PAAC**: Pseudo Amino Acid Composition
   - Data loaders: `GetProtT5_K_4()`, `GetTPEMPPS_CCP()`, etc. return (X_train, y_train, X_test, y_test, ratio)

3. **Training Pipeline** (ProtT5/ and TPEMPPS_CCP/ folders)
   - PPO (Proximal Policy Optimization) algorithm using TorchRL
   - Key hyperparameters: lr=3e-4, gamma=0.99, lambda=0.95, clip_epsilon=0.2
   - GAE (Generalized Advantage Estimation) for value function
   - ClipPPOLoss for policy optimization
   - Metrics tracked: Accuracy, MCC, Sensitivity, Specificity

4. **Ensemble Prediction** (Models/Vote.py)
   - Loads two trained models: ProtT5 and TPEMPPS_CCP
   - Voting mechanism to combine predictions
   - Outputs probabilities and performance metrics

5. **Dataset Structure**
   - `Dataset/train/`: Training data (FASTA sequences + pre-computed features)
   - `Dataset/test/`: Test data (FASTA sequences + pre-computed features)
   - Format: Negative and positive site samples for balanced evaluation
   - Ratio parameter: Accounts for class imbalance between positive/negative samples

### File Organization

```
RLSuccSite/
├── Env.py                      # RL environment class
├── Features.py                 # Data loading functions
├── Feature/                    # Individual feature extractors
│   ├── CKSAAP.py
│   ├── CTDC.py
│   ├── PAAC.py
│   ├── TPEMPPS.py
│   ├── ProtT5_K.py
│   └── ProtT5.py
├── ProtT5/
│   └── TrainPPO_ProtT5.py     # Training script
├── TPEMPPS_CCP/
│   └── TrainPPO_TPEMPPS_CCP.py # Training script
├── Models/
│   ├── Vote.py                # Ensemble inference
│   ├── Data/                  # Output predictions and metrics
│   └── *.pth                  # Trained model weights
└── Dataset/
    ├── train/
    │   ├── fasta/             # Protein sequences
    │   └── features/          # Pre-computed features (CSV)
    └── test/
        ├── fasta/
        └── features/
```

## Key Development Patterns

### Data Loading Pattern
```python
from Features import GetProtT5_K_4
X_train, y_train, X_test, y_test, ratio = GetProtT5_K_4(
    train_negative_path, train_positive_path,
    test_negative_path, test_positive_path
)
```

### Model Training Pattern
1. Load features via `GetProtT5_K_4()` or similar
2. Create custom environment with `PPOEnvZcc(X_train, y_train, ratio, total_frames, model_name)`
3. Wrap with TorchRL transformations (`Compose`, `DoubleToFloat`, `StepCounter`, `TransformedEnv`)
4. Create policy network and value network using `nn.Sequential` with LazyLinear layers
5. Wrap with `TensorDictModule` and `ProbabilisticActor`
6. Use `SyncDataCollector` to collect rollouts
7. Train with `ClipPPOLoss` and `GAE`

### Model Inference Pattern
```python
# Load pre-trained model
model_path = 'model_name.pth'
policy_module = load_model(...)
policy_module.eval()

# Inference
with torch.no_grad():
    output = policy_module(TensorDict({'observation': X_test}, batch_size=[]))
    predictions = output['logits'].argmax(dim=-1)
```

## Important Implementation Details

### Environment Reward Mechanism
- **RewardTP**: `10.0 * (1.0 + step/total_frames) * ratio` - Increases over time, weighted by class imbalance
- **RewardTN**: `10.0 * (1.1 - step/total_frames)` - Decreases over time
- **RewardFN**: `-10.0 * (1.0 + step/total_frames) * ratio` - Negative, increases penalty
- **RewardFP**: `-10.0 * (0.1 + step/total_frames)` - Negative, light penalty

This balanced approach handles class imbalance and dynamic training progression.

### Data Format Notes
- **CSV Features**: Row 0 is header, columns 1+ contain numeric features
- **FASTA Sequences**: Standard FASTA format, extracted for sequence-based features
- **Labels (y)**: 0 = negative sites, 1 = positive (succinylation) sites
- **Class Ratio**: Computed for each feature type to account for imbalance

## Common Tasks

### Adding a New Feature Extractor
1. Create new file in `Feature/` directory with `Get<FeatureName>_4()` function
2. Function signature: `Get<FeatureName>_4(train_neg, train_pos, test_neg, test_pos) -> (X_train, y_train, X_test, y_test, ratio)`
3. Update `Features.py` to import and expose the new function
4. Create corresponding training script in new folder

### Troubleshooting Model Training
- Check `total_frames` calculation matches dataset size
- Verify `frames_per_batch` is smaller than `total_frames`
- Monitor reward scaling - if rewards are too large, training may be unstable
- Ensure device settings (`cpu` vs `cuda`) are consistent

### Evaluating Models
- Use `sklearn.metrics`: confusion_matrix, f1_score, roc_auc_score, recall_score, average_precision_score
- Key metrics: Accuracy (ACC), Matthews Correlation Coefficient (MCC), Sensitivity (SN), Specificity (SP)
- Model naming convention: `<FeatureName>_ACC<acc>_MCC<mcc>_SN<sn>_SP<sp>.pth`

## Dependencies & Versions
- torch==2.1.1
- torchrl==0.2.1 (Reinforcement Learning)
- tensordict==0.2.1 (Efficient dict-based tensor operations)
- numpy==1.24.4
- pandas==2.0.3
- Bio==1.4.0 (Biopython)
- protlearn==0.0.3 (Protein learning utilities)
- sklearn (via requirements)

## References
- PyTorch PPO tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html#
- TorchRL documentation: https://pytorch.org/rl/
- Project uses TensorDict for efficient data handling in RL environments

## Code Style Notes
- Comments are primarily in Chinese (original project language)
- Variable naming: `X_train`, `y_train` (features and labels), `model_name` for experiment tracking
- Metrics are printed/saved with naming: `ACC`, `MCC`, `SN`, `SP` (Sensitivity, Specificity)
