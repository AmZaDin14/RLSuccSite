from Bio import SeqIO
import os

def extract_fragments(input_fasta, output_fasta, window_size=33):
    half_window = window_size // 2
    count = 0
    with open(output_fasta, "w") as out_f:
        for record in SeqIO.parse(input_fasta, "fasta"):
            seq = str(record.seq).upper()
            # Check if it looks like DNA
            if all(c in "ATGCN" for c in seq[:100]) and len(seq) > 100:
                print(f"Warning: {record.id} looks like DNA. This model requires Protein sequences.")
                continue
                
            for i, res in enumerate(seq):
                if res == 'K':
                    left = max(0, i - half_window)
                    right = min(len(seq), i + half_window + 1)
                    fragment = seq[left:right]
                    
                    # Apply 'X' padding
                    if i < half_window:
                        fragment = 'X' * (half_window - i) + fragment
                    if (len(seq) - 1 - i) < half_window:
                        fragment = fragment + 'X' * (half_window - (len(seq) - 1 - i))
                    
                    out_f.write(f">{record.id}|pos_{i+1}\n{fragment}\n")
                    count += 1
    print(f"Extraction complete. Found {count} lysine (K) sites.")

if __name__ == "__main__":
    # Update these filenames as needed
    input_file = "ncbi_dataset/ncbi_dataset/data/GCF_001605985.2/protein.faa" 
    output_file = "extracted_sites.fasta"
    
    if os.path.exists(input_file):
        extract_fragments(input_file, output_file)
    else:
        print(f"Error: File '{input_file}' not found.")
