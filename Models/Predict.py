import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import joblib
import tempfile
import argparse
import numpy as np
import torch
import pandas as pd
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torch.distributions import OneHotCategorical
from torchrl.data import DiscreteTensorSpec
from torchrl.modules import ProbabilisticActor
import torch.nn.functional as F

# Feature extractors
from Feature.CKSAAP import extract_cksAAP_from_fasta
from Feature.CTDC import extract_ctdc_from_fasta
from Feature.PAAC import extract_pse_aac_from_fasta
from Feature.TPEMPPS import ZccF_LiHua, ZccF_alltoK


device = torch.device("cpu")
BASE_DIR = Path(__file__).resolve().parent.parent

train_negative_fasta = BASE_DIR / 'Dataset/train/fasta/train_negative_sites.fasta'
train_positive_fasta = BASE_DIR / 'Dataset/train/fasta/train_positive_sites.fasta'

# -------------------- FASTA streaming --------------------
def stream_fasta_batches(fasta_path, batch_size):
    batch_ids, batch_seqs = [], []
    with open(fasta_path, "r") as f:
        while True:
            title = f.readline().strip()
            if not title:
                break
            seq = f.readline().strip()

            batch_ids.append(title)
            batch_seqs.append(seq)

            if len(batch_seqs) == batch_size:
                yield batch_ids, batch_seqs
                batch_ids, batch_seqs = [], []

    if batch_seqs:
        yield batch_ids, batch_seqs

# -------------------- Parallel worker --------------------
def process_chunk(args):
    indices, seqs = args

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        for i, s in enumerate(seqs):
            tmp.write(f">seq{i}\n{s}\n")
        tmp_path = tmp.name

    tpempps = np.hstack((ZccF_LiHua(tmp_path), ZccF_alltoK(tmp_path)))
    ccp = np.hstack((
        extract_cksAAP_from_fasta(tmp_path),
        extract_ctdc_from_fasta(tmp_path),
        extract_pse_aac_from_fasta(tmp_path)
    ))

    os.remove(tmp_path)
    return indices, tpempps, ccp

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prott5_features_pt', required=True)
    parser.add_argument('--fragments_fasta', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=2048)
    args = parser.parse_args()

    print("Loading ProtT5 features...")
    data = torch.load(args.prott5_features_pt, map_location='cpu')
    sequence_ids = list(data['ids'])
    X_ProtT5_new = data['features'].cpu().numpy().astype(np.float32)

    # -------------------- Load or fit scalers --------------------
    scaler_tpempps_path = BASE_DIR / "Models/scaler_tpempps.pkl"
    scaler_ccp_path = BASE_DIR / "Models/scaler_ccp.pkl"

    if scaler_tpempps_path.exists() and scaler_ccp_path.exists():
        scaler_tpempps = joblib.load(scaler_tpempps_path)
        scaler_ccp = joblib.load(scaler_ccp_path)
    else:
        X_train_tpempps = np.vstack((
            np.hstack((ZccF_LiHua(train_negative_fasta), ZccF_alltoK(train_negative_fasta))),
            np.hstack((ZccF_LiHua(train_positive_fasta), ZccF_alltoK(train_positive_fasta)))
        ))
        X_train_ccp = np.vstack((
            np.hstack((
                extract_cksAAP_from_fasta(train_negative_fasta),
                extract_ctdc_from_fasta(train_negative_fasta),
                extract_pse_aac_from_fasta(train_negative_fasta)
            )),
            np.hstack((
                extract_cksAAP_from_fasta(train_positive_fasta),
                extract_ctdc_from_fasta(train_positive_fasta),
                extract_pse_aac_from_fasta(train_positive_fasta)
            ))
        ))

        scaler_tpempps = StandardScaler().fit(X_train_tpempps)
        scaler_ccp = StandardScaler().fit(X_train_ccp)

        joblib.dump(scaler_tpempps, scaler_tpempps_path)
        joblib.dump(scaler_ccp, scaler_ccp_path)

    # -------------------- Load models --------------------
    actor_net_ProtT5 = nn.Sequential(nn.LazyLinear(1024), nn.ReLU(), nn.LazyLinear(2)).to(device)
    policy_module_ProtT5 = ProbabilisticActor(
        module=TensorDictModule(actor_net_ProtT5, in_keys=['observation'], out_keys=['logits']),
        spec=DiscreteTensorSpec(2),
        in_keys=['logits'],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    ).to(device)

    policy_module_ProtT5.load_state_dict(torch.load(BASE_DIR / 'Models/ProtT5_N10_ACC7142_MCC3513_SN7191_SP7130.pth', map_location=device))
    policy_module_ProtT5.eval()

    input_dim = scaler_tpempps.mean_.shape[0] + scaler_ccp.mean_.shape[0]
    actor_net_ZccFCCP = nn.Sequential(nn.LazyLinear(input_dim), nn.ReLU(), nn.LazyLinear(2)).to(device)
    policy_module_ZccFCCP = ProbabilisticActor(
        module=TensorDictModule(actor_net_ZccFCCP, in_keys=['observation'], out_keys=['logits']),
        spec=DiscreteTensorSpec(2),
        in_keys=['logits'],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    ).to(device)

    policy_module_ZccFCCP.load_state_dict(torch.load(BASE_DIR / 'Models/TPEMPPS_CCP_ACC7083_MCC3307_SN6943_SP7116.pth', map_location=device))
    policy_module_ZccFCCP.eval()

    # -------------------- Prediction --------------------
    print("Streaming + Parallel prediction...")

    predicted_labels = []
    positive_probabilities = []
    all_sequences = []

    idx_offset = 0

    pool = Pool(processes=args.num_workers)

    with torch.no_grad():
        for batch_ids, batch_seqs in stream_fasta_batches(args.fragments_fasta, args.batch_size):

            all_sequences.extend(batch_seqs)

            # Split batch into chunks for workers
            chunk_size = max(1, len(batch_seqs) // args.num_workers)
            chunks = []

            for i in range(0, len(batch_seqs), chunk_size):
                indices = list(range(idx_offset + i, idx_offset + i + len(batch_seqs[i:i+chunk_size])))
                seqs = batch_seqs[i:i+chunk_size]
                chunks.append((indices, seqs))

            results = pool.map(process_chunk, chunks)

            # Merge results (order preserved by chunk order)
            tp_list, cc_list = [], []
            for indices, tp, cc in results:
                tp_list.append(tp)
                cc_list.append(cc)

            X_tpempps = np.vstack(tp_list)
            X_ccp = np.vstack(cc_list)

            X_tpempps = scaler_tpempps.transform(X_tpempps)
            X_ccp = scaler_ccp.transform(X_ccp)

            X_batch = np.hstack((X_tpempps, X_ccp)).astype(np.float32)
            X_prot_batch = X_ProtT5_new[idx_offset: idx_offset + len(batch_seqs)]

            # GPU batch inference
            X_prot_tensor = torch.tensor(X_prot_batch, device=device)
            X_hand_tensor = torch.tensor(X_batch, device=device)

            td1 = TensorDict({'observation': X_prot_tensor}, batch_size=[len(batch_seqs)])
            logits1 = policy_module_ProtT5(td1)['logits']

            td2 = TensorDict({'observation': X_hand_tensor}, batch_size=[len(batch_seqs)])
            logits2 = policy_module_ZccFCCP(td2)['logits']

            avg_logits = (logits1 * 0.5) + (logits2 * 0.5)
            probs = F.softmax(avg_logits, dim=-1)

            positive_probabilities.extend(probs[:, 1].cpu().tolist())
            predicted_labels.extend(probs.argmax(dim=-1).cpu().tolist())

            idx_offset += len(batch_seqs)
            print(f"Processed {idx_offset} samples", end="\r")
    print()

    pool.close()
    pool.join()

    df = pd.DataFrame({
        'SequenceID': sequence_ids,
        'Sequence': all_sequences,
        'PositiveProbability': positive_probabilities,
        'PredictedLabel': predicted_labels,
    })

    df.to_csv(args.output, index=False)
    print(f"Saved → {args.output}")


if __name__ == '__main__':
    main()
