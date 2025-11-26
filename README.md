# EdiProPred

A machine learning–based platform for predicting **edible and non-edible proteins** from primary amino acid sequences.

---

## 📌 Introduction

**EdiProPred** supports food science, nutrition biology, crop genomics, and protein engineering by predicting edible vs. non-edible proteins using modern protein language models.

It integrates:

- ESM2 protein language models  
- Embedding-based Extra Trees classifier  
- Prediction, Protein Scanning, and Mutant Design modules  
- FASTA and simple sequence formats  

Webserver (placeholder):  
http://webs.iiitd.edu.in/raghava/edipropred

---

## 📚 Reference

coming soon 


---



# 🧪 Quick Start for Reproducibility

```bash
# Clone repository
git clone https://github.com/raghavagps/edipropred.git
cd edipropred

# Create environment
conda env create -f environment.yml
conda activate EdiProPred

# Download model and extract into project root

# Run prediction
python edipropred.py -i example.fasta -o output.csv -m 1 -j 1 -wd workdir
```

---

# 🛠 Installation

## PIP Installation

```bash
pip install edipropred
```

Check help:

```bash
edipropred -h
```

---

## Manual Installation

### Python Version

```
python >= 3.10
```

### Required Libraries

```
python=3.12.5 
torch==2.6.0+cu124
scikit-learn==1.3.2.
transformers==4.51.3 
biopython==1.84 
fair-esm==2.0.0 
pandas==2.1.4
```

Install manually:

```bash
pip install numpy pandas scikit-learn torch transformers biopython fair-esm joblib 
```

---

## Install using environment.yml

```bash
conda env create -f environment.yml
conda activate EdiProPred
```

---

# ⚠️ Model Download

Download the ESM2 model and trained classifier from:

https://webs.iiitd.edu.in/raghava/edipropred/download.html

Extract the zip file into the project root directory before running predictions.

---

# 🔬 Classification Overview

EdiProPred predicts:

- **Edible proteins**  
- **Non-edible proteins**

---

## Model Options

| Model | Description |
|-------|-------------|
| ESM2-t33 Direct | Fine-tuned model for fast predictions |
| ESM2 + Extra Trees | Embeddings + Extra Trees classifier |
| Combined Model | Automatically selects best method |

**Default model:** ESM2-t33 Direct

---

# 🚀 Usage

```bash
edipropred.py -h
```

Run basic prediction:

```bash
edipropred.py -i example.fasta
```

---

## Full Usage

```bash
usage: edipropred.py [-h]
                     [-i INPUT]
                     [-o OUTPUT]
                     [-t THRESHOLD]
                     [-j {1,2,3,4}]
                     [-m {1,2,3}]
                     [-wd WORKING_DIRECTORY]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `-i` | Input sequence file |
| `-o` | Output CSV file |
| `-t` | Threshold (0–1) |
| `-j` | Job type |
| `-m` | Model selection |
| `-wd` | Working directory |

---

# 📂 Input & Output

### Input Formats

- FASTA  
- One sequence per line  

### Output

- CSV file with predictions, probabilities, and model scores

---

# 🔍 Job Types

| Job | Description |
|-----|-------------|
| 1 | Prediction |
| 2 | Protein Scanning |
| 3 | Mutant Design |
| 4 | All Possible Mutants |

---

# 📑 Package Contents

| File | Description |
|------|-------------|
| INSTALLATION | Installation instructions |
| LICENSE | License info |
| README.md | This file |
| edipropred.py | Main script |
| example.fasta | Sample FASTA input |

---

# 📦 Install via PIP

```bash
pip install edipropred
```

---

# 🚀 Start predicting edible proteins with EdiProPred today!

