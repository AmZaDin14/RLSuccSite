import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from sklearn.preprocessing import StandardScaler
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torch.distributions import OneHotCategorical
from torchrl.data import DiscreteTensorSpec
from torchrl.modules import ProbabilisticActor
import torch.nn.functional as F

# Import feature extractors
from Feature.CKSAAP import extract_cksAAP_from_fasta
from Feature.CTDC import extract_ctdc_from_fasta
from Feature.PAAC import extract_pse_aac_from_fasta
from Feature.TPEMPPS import ZccF_LiHua, ZccF_alltoK

device = "cpu"
BASE_DIR = Path(__file__).resolve().parent.parent

# Training data paths for fitting scalers
train_negative_fasta = BASE_DIR / "Dataset/train/fasta/train_negative_sites.fasta"
train_positive_fasta = BASE_DIR / "Dataset/train/fasta/train_positive_sites.fasta"


def extract_tpempps_features(fasta_file):
    """Extract TPEMPPS features (ZccF_LiHua + ZccF_alltoK) from FASTA."""
    features_lihua = ZccF_LiHua(fasta_file)
    features_alltok = ZccF_alltoK(fasta_file)
    return np.hstack((features_lihua, features_alltok))


def extract_ccp_features(fasta_file):
    """Extract CCP features (CKSAAP + CTDC + PAAC) from FASTA."""
    cks_features = extract_cksAAP_from_fasta(fasta_file)
    ctd_features = extract_ctdc_from_fasta(fasta_file)
    paac_features = extract_pse_aac_from_fasta(fasta_file)
    return np.hstack((cks_features, ctd_features, paac_features))


def main():
    parser = argparse.ArgumentParser(
        description="Predict succinylation sites using RLSuccSite ensemble model."
    )
    parser.add_argument(
        "--prott5_features",
        required=True,
        help="Path to the ProtT5 features CSV file for the new data.",
    )
    parser.add_argument(
        "--fragments_fasta",
        required=True,
        help="Path to the FASTA file with 33-residue fragments for the new data.",
    )
    parser.add_argument(
        "--output", required=True, help="Path to save the prediction results CSV file."
    )
    args = parser.parse_args()

    print("Loading and processing new data...")

    # Get sequence identifiers
    sequence_ids = [record.id for record in SeqIO.parse(args.fragments_fasta, "fasta")]
    num_samples = len(sequence_ids)

    # -----------------------------------------------------Fit Scalers on Training Data-----------------------------------------------------
    print("Extracting training features to fit scalers...")
    # Extract TPEMPPS features from training data
    X_train_tpempps = np.vstack(
        (
            extract_tpempps_features(train_negative_fasta),
            extract_tpempps_features(train_positive_fasta),
        )
    )

    # Extract CCP features from training data
    X_train_ccp = np.vstack(
        (
            extract_ccp_features(train_negative_fasta),
            extract_ccp_features(train_positive_fasta),
        )
    )

    # Fit separate scalers for TPEMPPS and CCP, as done during training
    scaler_tpempps = StandardScaler()
    scaler_tpempps.fit(X_train_tpempps)

    scaler_ccp = StandardScaler()
    scaler_ccp.fit(X_train_ccp)

    print(
        f"Scalers fitted. TPEMPPS dim: {X_train_tpempps.shape[1]}, CCP dim: {X_train_ccp.shape[1]}"
    )

    # -----------------------------------------------------Process New Data-----------------------------------------------------
    print("Processing new data...")
    # Load ProtT5 features (assumes first column is ID, rest are features)
    X_ProtT5_new = (
        pd.read_csv(args.prott5_features, header=None)
        .iloc[:, 1:]
        .values.astype(np.float32)
    )

    # Extract TPEMPPS features from new data and scale
    X_new_tpempps = extract_tpempps_features(args.fragments_fasta)
    X_new_tpempps_scaled = scaler_tpempps.transform(X_new_tpempps)

    # Extract CCP features from new data and scale
    X_new_ccp = extract_ccp_features(args.fragments_fasta)
    X_new_ccp_scaled = scaler_ccp.transform(X_new_ccp)

    # Combine in same order as training: [TPEMPPS, CCP]
    X_ZccFCCP_new = np.hstack((X_new_tpempps_scaled, X_new_ccp_scaled)).astype(
        np.float32
    )

    print(
        f"New data - ProtT5: {X_ProtT5_new.shape}, TPEMPPS+CCP: {X_ZccFCCP_new.shape}"
    )

    # Validate dimensions
    if X_ProtT5_new.shape[0] != X_ZccFCCP_new.shape[0]:
        print(f"Error: Mismatch in number of samples.")
        print(f"ProtT5 features: {X_ProtT5_new.shape[0]} samples")
        print(f"TPEMPPS+CCP features: {X_ZccFCCP_new.shape[0]} samples")
        sys.exit(1)

    # -----------------------------------------------------Load Models-----------------------------------------------------
    # ProtT5 model (1024 features)
    actor_net_ProtT5 = nn.Sequential(
        nn.LazyLinear(1024, device=device),
        nn.ReLU(),
        nn.LazyLinear(2, device=device),
    )
    policy_module_ProtT5 = ProbabilisticActor(
        module=TensorDictModule(
            actor_net_ProtT5, in_keys=["observation"], out_keys=["logits"]
        ),
        spec=DiscreteTensorSpec(2),
        in_keys=["logits"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )
    model_path_ProtT5 = (
        BASE_DIR / "Models" / "ProtT5_N10_ACC7142_MCC3513_SN7191_SP7130.pth"
    )
    policy_module_ProtT5.load_state_dict(
        torch.load(model_path_ProtT5, map_location=device)
    )
    policy_module_ProtT5.eval()

    # TPEMPPS_CCP model (feature size from training data)
    # Determine TPEMPPS_CCP model input dimension from training data dimensions
    input_dim = X_train_tpempps.shape[1] + X_train_ccp.shape[1]
    actor_net_ZccFCCP = nn.Sequential(
        nn.LazyLinear(input_dim, device=device),
        nn.ReLU(),
        nn.LazyLinear(2, device=device),
    )
    policy_module_ZccFCCP = ProbabilisticActor(
        module=TensorDictModule(
            actor_net_ZccFCCP, in_keys=["observation"], out_keys=["logits"]
        ),
        spec=DiscreteTensorSpec(2),
        in_keys=["logits"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )
    model_path_ZccFCCP = (
        BASE_DIR / "Models" / "TPEMPPS_CCP_ACC7083_MCC3307_SN6943_SP7116.pth"
    )
    policy_module_ZccFCCP.load_state_dict(
        torch.load(model_path_ZccFCCP, map_location=device)
    )
    policy_module_ZccFCCP.eval()

    # -----------------------------------------------------Ensemble Prediction-----------------------------------------------------
    print(f"Performing prediction on {num_samples} sites...")
    predicted_labels = []
    positive_probabilities = []

    with torch.no_grad():
        weight1 = 0.5  # ProtT5
        weight2 = 0.5  # TPEMPPS_CCP

        for i in range(X_ProtT5_new.shape[0]):
            # ProtT5 prediction
            td1 = TensorDict({"observation": X_ProtT5_new[i]}, batch_size=[])
            logits1 = policy_module_ProtT5(td1)["logits"]

            # TPEMPPS_CCP prediction
            td2 = TensorDict({"observation": X_ZccFCCP_new[i]}, batch_size=[])
            logits2 = policy_module_ZccFCCP(td2)["logits"]

            # Weighted average of logits
            avg_logits = (logits1 * weight1) + (logits2 * weight2)
            avg_probs = F.softmax(avg_logits, dim=-1)

            positive_probabilities.append(avg_probs[1].item())
            prediction = avg_probs.argmax(dim=-1)
            predicted_labels.append(prediction.item())

    # ---------------------------------------------------Save Results----------------------------------------------------
    df_predictions = pd.DataFrame(
        {
            "SequenceID": sequence_ids,
            "PositiveProbability": positive_probabilities,
            "PredictedLabel": predicted_labels,
        }
    )
    df_predictions.to_csv(args.output, index=False)
    print(f"Predictions saved successfully to {args.output}")
    print(f"Total predictions: {len(predicted_labels)}")
    print(f"Positive sites predicted: {sum(predicted_labels)}")


if __name__ == "__main__":
    main()
