import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
from tqdm import tqdm
import csv
import numpy as np

tokenizer = T5Tokenizer.from_pretrained("../ProtT5", do_lower_case=False)
model = T5EncoderModel.from_pretrained("../ProtT5", local_files_only=True)

device = torch.device("cpu")
model = model.to(device)
model = model.eval()


def find_features_full_seq(sequence):
    sequence = re.sub(r"[UZOB]", "X", sequence)
    sequence = " ".join(sequence)
    ids = tokenizer.encode_plus(
        sequence, add_special_tokens=True, padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        embedding = model(**ids)[0]
    embedding = embedding.squeeze(0).cpu().numpy()
    seq_len = (ids["attention_mask"][0] == 1).sum().item()
    seq_emd = embedding[: seq_len - 1]
    return seq_emd


def process_sequences_file(input_file, output_file, start_index=0):
    batch_size = 8
    with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
        csv_writer = csv.writer(outfile)
        sequences_batch = []
        total_sequences = sum(1 for line in infile) // 2
        infile.seek(0)
        for _ in range(start_index):
            next(infile)
            next(infile)
        with tqdm(total=total_sequences - start_index) as pbar:
            while True:
                title = infile.readline().strip()
                if not title:
                    break
                sequence = infile.readline().strip()
                sequences_batch.append((title, sequence))

                if len(sequences_batch) == batch_size:
                    features_batch = [
                        find_features_full_seq(seq[1]) for seq in sequences_batch
                    ]
                    for title_seq, features in zip(sequences_batch, features_batch):
                        features = features.reshape(-1, 1024)
                        center_features = features[16]
                        csv_writer.writerow([title_seq[0]] + center_features.tolist())
                    sequences_batch = []
                    pbar.update(batch_size)

            if len(sequences_batch) > 0:
                features_batch = [
                    find_features_full_seq(seq[1]) for seq in sequences_batch
                ]
                for title_seq, features in zip(sequences_batch, features_batch):
                    features = features.reshape(-1, 1024)
                    center_features = features[16]
                    csv_writer.writerow([title_seq[0]] + center_features.tolist())
                pbar.update(len(sequences_batch))


input_file_path = "../Dataset/d.catenatum/extracted_sites.fasta"
output_file_path = "../Dataset/d.catenatum/extracted_sites_ProtT5_features_K.csv"

start_index = 0

process_sequences_file(input_file_path, output_file_path, start_index)
