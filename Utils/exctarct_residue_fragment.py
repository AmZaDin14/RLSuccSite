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
    extract_fragments("KC771275.1.fasta", "extracted_sites.fasta")
