#########################################################################
# EDIpropred is developed for predicting Edible and Non-Edible proteins 
# from their primary sequence. It is developed by  #
# Prof G. P. S. Raghava's group. Please cite : EDIpropred                 #
# Available at: https://webs.iiitd.edu.in/raghava/EDIpropred/             #
#########################################################################
import argparse
import warnings
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import shutil

warnings.filterwarnings("ignore")

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR_T33 = os.path.join(BASE_DIR, "Model", "saved_model_t33")

MODEL_PATH_T33 = os.path.join(MODEL_DIR_T33, "final_full_model_object.pth")
ALPHABET_PATH_T33 = os.path.join(MODEL_DIR_T33, "esm_alphabet.pth")

# =========================
# REQUIRED CLASS DEFINITION
# =========================
class ProteinClassifier(nn.Module):
    def __init__(self, esm_model, embedding_dim, num_classes):
        super().__init__()
        self.esm_model = esm_model
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, tokens):
        with torch.no_grad():
            results = self.esm_model(tokens, repr_layers=[33])
        emb = results["representations"][33].mean(1)
        return self.fc(emb)

# =========================
# Load TRAINED model + alphabet
# =========================
print("🔄 Loading trained ESM2-t33 model...")

alphabet = torch.load(ALPHABET_PATH_T33, map_location="cpu", weights_only=False)
batch_converter = alphabet.get_batch_converter()

classifier = torch.load(MODEL_PATH_T33, map_location=device, weights_only=False)
classifier = classifier.to(device)
classifier.eval()

print("✅ Trained ESM2-t33 model loaded successfully")

# =========================
# Sequence reader
# =========================
def readseq(file):
    seqid, seq = [], []
    with open(file) as f:
        content = f.read().strip()

    if content.startswith(">"):
        records = content.split(">")[1:]
        for r in records:
            lines = r.splitlines()
            seqid.append(">" + lines[0].split()[0])
            seq.append(re.sub("[^ACDEFGHIKLMNPQRSTVWY-]", "", "".join(lines[1:]).upper()))
    else:
        for i, s in enumerate(content.splitlines(), 1):
            seqid.append(f">Seq_{i}")
            seq.append(re.sub("[^ACDEFGHIKLMNPQRSTVWY-]", "", s.upper()))

    return pd.DataFrame(seqid, columns=["Seq_id"]), pd.DataFrame(seq, columns=["Sequence"])

# =========================
# Prediction function
# =========================
def predict_from_dataframe(df, threshold, output_path, batch_size=4):
    data = list(zip(df["Seq_id"], df["Sequence"]))
    all_ids, all_seq, all_scores, all_preds = [], [], [], []

    for i in tqdm(range(0, len(data), batch_size), desc="Prediction"):
        batch = data[i:i+batch_size]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)

        with torch.no_grad():
            logits = classifier(tokens)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        for (sid, seq), p in zip(batch, probs):
            all_ids.append(sid.replace(">", ""))
            all_seq.append(seq)
            all_scores.append(p)
            all_preds.append("Edible" if p >= threshold else "Non-Edible")

    out_df = pd.DataFrame({
        "SeqID": all_ids,
        "Sequence": all_seq,
        "ESM Score": np.round(all_scores, 4),
        "Prediction": all_preds
    })

    out_df.to_csv(output_path, index=False)
    print(f"✅ Results saved to {output_path}")
    return out_df

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="EDIpropred (ESM2-t33 trained model)")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA / text file")
    parser.add_argument("-o", "--output", default="output.csv", help="Output CSV")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Threshold (0–1)")
    parser.add_argument("-wd", "--working", default=os.getcwd(), help="Working directory")

    args = parser.parse_args()

    wd = args.working
    os.makedirs(wd, exist_ok=True)

    print("\n===== EDIpropred Prediction =====")
    print(f"Input     : {args.input}")
    print(f"Threshold : {args.threshold}")
    print(f"Output    : {args.output}")

    # Read sequences
    seqid, seq = readseq(args.input)
    df = pd.concat([seqid, seq], axis=1)

    # Predict
    predict_from_dataframe(
        df=df,
        threshold=args.threshold,
        output_path=os.path.join(wd, args.output)
    )

if __name__ == "__main__":
    main()
