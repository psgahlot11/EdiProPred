#########################################################################
# EDIpropred
# Prediction of Edible / Non-Edible proteins from sequence
# Developed by Prof. G. P. S. Raghava's group
# https://webs.iiitd.edu.in/raghava/edipropred/
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
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

# =========================================================
# Device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Hugging Face repository (MODEL SOURCE)
# =========================================================
HF_REPO_ID = "raghavagps-group/edipropred"
HF_MODEL_FILE = "final_full_model_object.pth"
HF_ALPHABET_FILE = "esm_alphabet.pth"

# Local cache directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

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
# Download model files from Hugging Face
# =========================================================
print("🔄 Checking / downloading model files from Hugging Face...")

MODEL_PATH = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=HF_MODEL_FILE,
    cache_dir=CACHE_DIR
)

ALPHABET_PATH = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=HF_ALPHABET_FILE,
    cache_dir=CACHE_DIR
)

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
    position -= 1
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

    return pd.concat([design_df["Type"], pred_df], axis=1)

# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="EDIpropred – Edible Protein Prediction (ESM2-t33)"
    )

    parser.add_argument("-i", "--input", required=True, help="Input FASTA / text file")
    parser.add_argument("-o", "--output", default="output.csv", help="Output CSV")
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    parser.add_argument("-j", "--job", type=int, choices=[1, 3], default=1,
                        help="1 Prediction | 3 Design")
    parser.add_argument("-p", "--position", type=int, help="Mutation position")
    parser.add_argument("-r", "--residues", type=str, help="Mutant residues")
    parser.add_argument("-wd", "--working", default=os.getcwd())

    args = parser.parse_args()
    os.makedirs(args.working, exist_ok=True)

    df = readseq(args.input)

    if args.job == 1:
        result = predict_from_dataframe(df, args.threshold)
    else:
        if args.position is None or args.residues is None:
            raise ValueError("Design job requires --position and --residues")
        result = design_module(df, args.residues.upper(), args.position, args.threshold)

    out_path = os.path.join(args.working, args.output)
    result.to_csv(out_path, index=False)
    print(f"✅ Results saved to {out_path}")

# =========================================================
if __name__ == "__main__":
    main()
