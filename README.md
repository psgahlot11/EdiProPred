# EdiProPred
A machine learning–based platform for predicting edible and non-edible proteins from primary amino acid sequences.

📌 Introduction

EdiProPred is designed to support researchers in food science, nutrition biology, crop genomics, and allergen-free protein engineering by offering accurate prediction of edible vs. non-edible proteins using large-scale protein language models.

The tool integrates:

ESM2 protein language models

Evolutionary embeddings

Machine Learning classifiers (Extra Trees)

EdiProPred offers prediction and Protein scanning features to enhance understanding of protein edibility, identify potential safety concerns, and inform nutritional design.

🔗 Visit the web server (placeholder):
http://webs.iiitd.edu.in/raghava/edipropred

📖 Please cite our work when using EdiProPred.

📚 Reference

Coming soon


🧪 Quick Start for Reproducibility
# 1. Clone the repository
git clone https://github.com/raghavagps/edipropred.git
cd edipropred

# 2. Create and activate environment
conda env create -f environment.yml
conda activate EdiProPred

# 3. Download pre-trained model (ESM + Classifier)
# Place the model folder inside the root directory

# 4. Run prediction on sample FASTA
python edipropred.py -i example.fasta -o results.csv -m 1 -j 1 -wd working_path

🛠️ Installation Options
🔹 PIP Installation
pip install edipropred


Check options:

edipropred -h

🔹 Standalone Installation Requirements
Python Version
python >= 3.10

Required Libraries
python=3.12.5 
torch==2.6.0+cu124
scikit-learn==1.3.2.
transformers==4.51.3 
biopython==1.84 
fair-esm==2.0.0 
pandas==2.1.4


Install manually:

pip install numpy pandas scikit-learn torch transformers biopython tqdm joblib 

🔹 Installation using environment.yml
conda env create -f environment.yml
conda activate EdiProPred

⚠️ Important Notes

The ESM2 model and trained classifier are large files.

Download the zipped model from the Download Page (placeholder):
https://webs.iiitd.edu.in/raghava/edipropred/download.html

Extract before running predictions.

🔬 Classification

EdiProPred classifies input sequences into:

✔ Edible Proteins

Likely safe and suitable for consumption.

✔ Non-Edible Proteins

May be toxic, allergenic, antinutritional, or structurally unsuitable.

🔹 Model Options
ESM2-t33 Direct	Fast prediction using fine-tuned ESM2 model
ESM2 + Extra Trees	Embeddings + ET classifier

Default Model: ESM2-t33 (Direct Model)

🚀 Usage
🔹 Minimum usage
edipropred.py -h


Predict from FASTA:

edipropred.py -i example.fasta

🔹 Full Usage
usage: edipropred.py [-h]
                     [-i INPUT]
                     [-o OUTPUT]
                     [-t THRESHOLD]
                     [-j {1,2,3,4}]
                     [-m {1,2}]
                     [-wd WORKING DIRECTORY]

Required Arguments
Argument	Description
-i INPUT	Input FASTA or simple format file
-o OUTPUT	Output CSV (default: outfile.csv)
-t THRESHOLD	Decision threshold (0–1, default: 0.5)
-j	Job type: 1-Predict, 2-Scan, 3-Design, 4-All Mutants
-m {1,2,3}	Model selection: 1-ESM2, 2-ET, 3-Combined
-wd	Working directory path
📂 Input & Output Formats
Input:

FASTA

Simple Sequence Format (one per line)

Output:

CSV with score, prediction, probability, and details.

🔍 Job Types & Features
Job	Function
1️⃣ Prediction	Predict edible vs. non-edible
2️⃣ Protein Scanning	Sliding-window analysis to find edible segments
📑 Package Contents
File	Description
INSTALLATION	Installation instructions
LICENSE	License
README.md	This documentation
edipropred.py	Main prediction script
example.fasta	Sample input
📦 PIP Installation (Reference)
pip install edipropred

🚀 Start predicting edible proteins with EdiProPred today!
