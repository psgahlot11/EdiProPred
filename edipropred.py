#########################################################################
# EDIpropred
# Prediction of Edible / Non-Edible proteins from sequence
# Developed by Prof. G. P. S. Raghava's group
# https://webs.iiitd.edu.in/raghava/EDIpropred/
#########################################################################

import argparse
import os
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =========================================================
# Device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Paths
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model", "saved_model_t33")

MODEL_PATH = os.path.join(MODEL_DIR, "final_full_model_object.pth")
ALPHABET_PATH = os.path.join(MODEL_DIR, "esm_alphabet.pth")

# =========================================================
# Model definition (MUST match trained model)
# =========================================================
class ProteinClassifier(nn.Module):
    def __init__(self, esm_model, embedding_dim, num_classes):
        super().__init__()
        self.esm_model = esm_model
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, tokens):
        with torch.no_grad():
            out = self.esm_model(tokens, repr_layers=[33])
        emb = out["representations"][33].mean(1)
        return self.fc(emb)

# =========================================================
# Load model + alphabet
# =========================================================
print("🔄 Loading trained ESM2-t33 model...")

alphabet = torch.load(ALPHABET_PATH, map_location="cpu", weights_only=False)
batch_converter = alphabet.get_batch_converter()

classifier = torch.load(MODEL_PATH, map_location=device, weights_only=False)
classifier = classifier.to(device)
classifier.eval()

print("✅ Model loaded successfully")

# =========================================================
# Sequence reader
# =========================================================
def readseq(file):
    seq_ids, seqs = [], []

    with open(file) as f:
        content = f.read().strip()

    if content.startswith(">"):  # FASTA
        records = content.split(">")[1:]
        for r in records:
            lines = r.splitlines()
            sid = lines[0].split()[0]
            seq = "".join(lines[1:]).upper()
            seq = re.sub("[^ACDEFGHIKLMNPQRSTVWY-]", "", seq)
            seq_ids.append(f">{sid}")
            seqs.append(seq)
    else:  # Plain text
        for i, s in enumerate(content.splitlines(), 1):
            seq_ids.append(f">Seq_{i}")
            seqs.append(re.sub("[^ACDEFGHIKLMNPQRSTVWY-]", "", s.upper()))

    return pd.DataFrame({"SeqID": seq_ids, "Sequence": seqs})

# =========================================================
# Prediction
# =========================================================
def predict_from_dataframe(df, threshold, batch_size=4):
    data = list(zip(df["SeqID"], df["Sequence"]))
    results = []

    for i in tqdm(range(0, len(data), batch_size), desc="Predicting"):
        batch = data[i:i + batch_size]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)

        with torch.no_grad():
            logits = classifier(tokens)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        for (sid, seq), score in zip(batch, probs):
            results.append({
                "SeqID": sid.replace(">", ""),
                "Sequence": seq,
                "ESM Score": round(float(score), 4),
                "Prediction": "Edible" if score >= threshold else "Non-Edible"
            })

    return pd.DataFrame(results)

# =========================================================
# Design module
# =========================================================
STD_AA = "ACDEFGHIKLMNPQRSTVWY"

def generate_mutant(seq, residues, position):
    position -= 1  # 1-based → 0-based

    if position < 0 or position >= len(seq):
        raise ValueError("Mutation position out of range")

    if any(r not in STD_AA for r in residues):
        raise ValueError("Invalid amino acid in residues")

    if len(residues) == 1:
        return seq[:position] + residues + seq[position + 1:]

    if len(residues) == 2:
        return seq[:position] + residues + seq[position + 2:]

    raise ValueError("Residues must be 1 or 2 amino acids")


def design_module(df, residues, position, threshold):
    if len(df) != 1:
        raise ValueError("Design module accepts ONLY ONE sequence")

    original_seq = df.iloc[0]["Sequence"]
    seq_id = df.iloc[0]["SeqID"]

    if len(original_seq) > 400:
        raise ValueError("Sequence length must be ≤ 400 aa")

    mutant_seq = generate_mutant(original_seq, residues, position)

    design_df = pd.DataFrame({
        "SeqID": [seq_id, seq_id],
        "Sequence": [original_seq, mutant_seq],
        "Type": ["Original", "Mutant"]
    })

    pred_df = predict_from_dataframe(
        design_df[["SeqID", "Sequence"]],
        threshold=threshold
    )

    final_df = pd.concat([design_df["Type"], pred_df], axis=1)
    return final_df

# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="EDIpropred – Edible Protein Prediction (ESM2-t33)"
    )

    parser.add_argument("-i", "--input", required=True, help="Input FASTA / text file")
    parser.add_argument("-o", "--output", default="output.csv", help="Output CSV")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Threshold (0–1)")
    parser.add_argument(
        "-j", "--job",
        type=int,
        choices=[1, 3],
        default=1,
        help="Job: 1 Prediction | 3 Design"
    )
    parser.add_argument("-p", "--position", type=int, help="Mutation position (design)")
    parser.add_argument("-r", "--residues", type=str, help="Mutant residues (design)")
    parser.add_argument("-wd", "--working", default=os.getcwd(), help="Working directory")

    args = parser.parse_args()
    os.makedirs(args.working, exist_ok=True)

    print("\n===== EDIpropred =====")
    print(f"Input     : {args.input}")
    print(f"Job       : {args.job}")
    print(f"Threshold : {args.threshold}")
    print(f"Output    : {args.output}\n")

    df = readseq(args.input)

    if args.job == 1:
        result = predict_from_dataframe(df, args.threshold)

    elif args.job == 3:
        if args.position is None or args.residues is None:
            raise ValueError("Design job requires --position and --residues")

        result = design_module(
            df,
            residues=args.residues.upper(),
            position=args.position,
            threshold=args.threshold
        )

    out_path = os.path.join(args.working, args.output)
    result.to_csv(out_path, index=False)
    print(f"✅ Results saved to {out_path}")

# =========================================================
if __name__ == "__main__":
    main()
