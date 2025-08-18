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
import sklearn
from transformers import AutoTokenizer, EsmForSequenceClassification, EsmModel, EsmTokenizer

from Bio import SeqIO
import joblib
import torch
import pandas as pd
import torch
import pathlib
import shutil
import zipfile
import urllib.request
from tqdm.auto import tqdm
import tqdm
warnings.filterwarnings('ignore')


nf_path = os.path.dirname(__file__)

print('\n')
print('#####################################################################################')
print('# The program EDIpropred is developed for predicting Edible and Non-Edible    #')
print('# proteins from their primary sequence, developed by Prof G. P. S.     #')
print("# Raghava's group. Available at: https://webs.iiitd.edu.in/raghava/EDIpropred/        #")
print('#####################################################################################')


################################### Model Calling ##########################################
import argparse
import os
import zipfile
import urllib.request
from tqdm.auto import tqdm
import warnings
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import esm
from esm.data import Alphabet

torch.serialization.add_safe_globals([Alphabet]) 

# Suppress warnings
warnings.filterwarnings('ignore')

# Get the absolute path of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "Model")
ZIP_PATH = os.path.join(SCRIPT_DIR, "Model.zip")
MODEL_URL = "https://webs.iiitd.edu.in/raghava/EDIpropred/download/Model.zip"

# Check if the Model folder exists
if not os.path.exists(MODEL_DIR):
    print('##############################')
    print("Downloading the model files...")
    print('##############################')

    try:
        # Download the ZIP file with the progress bar
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(MODEL_URL, ZIP_PATH, reporthook=lambda block_num, block_size, total_size: t.update(block_size))

        print("Download complete. Extracting files...")

        # Extract the ZIP file
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(SCRIPT_DIR)

        print("Extraction complete. Removing ZIP file...")

        # Remove the ZIP file after extraction
        os.remove(ZIP_PATH)
        print("Model setup completed successfully.")

    except urllib.error.URLError as e:
        print(f"Network error: {e}. Please check your internet connection.")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is corrupted. Please try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
else:
    print('#################################################################')
    print("Model folder already exists. Skipping download.")
    print('#################################################################')




##################################################################################################################
# Function to check the sequence residue
import pandas as pd
import re
import os

def readseq(file):
    non_standard_detected = False  # Flag for non-standard amino acids
    _, ext = os.path.splitext(file)
    
    # Case 1: CSV format (first col: header with >, second col: sequence)
    if ext.lower() == ".csv":
        df = pd.read_csv(file)
        if df.shape[1] < 2:
            raise ValueError("CSV must have at least two columns: header and sequence.")
        
        seqid = df.iloc[:, 0].astype(str).tolist()  # Already has '>'
        seq_raw = df.iloc[:, 1].astype(str).str.upper().tolist()
        
        seq = []
        for s in seq_raw:
            filtered = re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '', s)
            if filtered != s:
                non_standard_detected = True
            seq.append(filtered)
    
    # Case 2: FASTA or plain text
    else:
        with open(file) as f:
            content = f.read().strip()
        
        if content.startswith(">"):  # FASTA
            records = content.split('>')[1:]
            seqid, seq = [], []
            for fasta in records:
                array = fasta.split('\n')
                name, sequence = array[0].split()[0], ''.join(array[1:]).upper()
                
                filtered_sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '', sequence)
                if filtered_sequence != sequence:
                    non_standard_detected = True
                
                seqid.append('>' + name)
                seq.append(filtered_sequence)
        
        else:  # Plain text: one sequence per line
            seq_raw = [line.strip().upper() for line in content.splitlines() if line.strip()]
            seqid = [">Seq_" + str(i) for i in range(1, len(seq_raw) + 1)]
            
            seq = []
            for s in seq_raw:
                filtered = re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '', s)
                if filtered != s:
                    non_standard_detected = True
                seq.append(filtered)
    
    # Messages
    if non_standard_detected:
        print("Non-standard amino acids were detected. Processed sequences have been saved and used for further prediction.")
    else:
        print("No non-standard amino acids were detected.")
    
    # Return DataFrames
    return pd.DataFrame(seqid), pd.DataFrame(seq)

# def readseq(file):
#     with open(file) as f:
#         records = f.read()
#     records = records.split('>')[1:]
#     seqid = []
#     seq = []
#     non_standard_detected = False  # Flag to track non-standard amino acids

#     for fasta in records:
#         array = fasta.split('\n')
#         name, sequence = array[0].split()[0], ''.join(array[1:]).upper()
        
#         # Check for non-standard amino acids
#         filtered_sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '', sequence)
#         if filtered_sequence != sequence:
#             non_standard_detected = True
        
#         seqid.append('>' + name)
#         seq.append(filtered_sequence)
    
#     if len(seqid) == 0:
#         f = open(file, "r")
#         data1 = f.readlines()
#         for each in data1:
#             seq.append(each.replace('\n', ''))
#         for i in range(1, len(seq) + 1):
#             seqid.append(">Seq_" + str(i))
    
#     if non_standard_detected:
#         print("Non-standard amino acids were detected. Processed sequences have been saved and used for further prediction.")
#     else:
#         print("No non-standard amino acids were detected.")
    
#     df1 = pd.DataFrame(seqid)
#     df2 = pd.DataFrame(seq)
#     return df1, df2


class ProteinClassifier(nn.Module):
    def __init__(self, esm_model, embedding_dim, num_classes):
        super(ProteinClassifier, self).__init__()
        self.esm_model = esm_model
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, tokens):
        with torch.no_grad():
            results = self.esm_model(tokens, repr_layers=[30])
        embeddings = results["representations"][30].mean(1)
        output = self.fc(embeddings)
        return output

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import os

def predict_from_dataframe(df, classifier, batch_converter, device, threshold=0.5, output_path=None, batch_size=32):
    df = df.dropna(subset=['Sequence']).copy()
    df['Sequence'] = df['Sequence'].astype(str).str.strip()
    data = list(zip(df['Seq_id'], df['Sequence']))

    all_ids = []
    all_sequences = []
    all_scores = []
    all_predictions = []

    classifier.eval()
    num_batches = (len(data) + batch_size - 1) // batch_size

    print(f"[INFO] Predicting {len(data)} sequences in {num_batches} batches...")
    for i in tqdm(range(0, len(data), batch_size), total=num_batches, desc="Prediction"):
        batch_data = data[i:i+batch_size]
        _, _, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)

        with torch.no_grad():
            outputs = classifier(tokens)
            probs = torch.softmax(outputs, dim=1)
            esm_scores = probs[:, 1].cpu().numpy()

        predictions = np.where(esm_scores >= threshold, 'Edible', 'Non-Edible')

        all_ids.extend([id_ for id_, _ in batch_data])
        all_sequences.extend([seq for _, seq in batch_data])
        all_scores.extend(esm_scores)
        all_predictions.extend(predictions)

    result_df = pd.DataFrame({
        'SeqID': all_ids,
        'Sequence': all_sequences,
        'ESM Score': all_scores,
        'Prediction': all_predictions
    })

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_csv(output_path, index=False)
        print(f"[INFO] Results saved to: {output_path}")

    return result_df


from tqdm import tqdm

def extract_embeddings(input_csv, classifier, batch_converter, device, output_csv, batch_size=32):
   

    if isinstance(input_csv, pd.DataFrame):
        df = input_csv.copy()
    else:
        df = pd.read_csv(input_csv)

    df = df.dropna(subset=['Sequence']).copy()
    df['Sequence'] = df['Sequence'].astype(str)

    sequences = df['Sequence'].tolist()
    seq_ids = df['Seq_id'].astype(str).tolist()

    all_embeddings = []
    all_ids = []
    all_sequences = []

    # Progress bar for batches
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    print(f"[INFO] Extracting embeddings for {len(sequences)} sequences in {num_batches} batches...")

    for batch_idx in tqdm(range(0, len(sequences), batch_size), total=num_batches, desc="Embedding extraction"):
        batch_seqs = sequences[batch_idx:batch_idx+batch_size]
        batch_ids = seq_ids[batch_idx:batch_idx+batch_size]
        batch_data = list(zip(batch_ids, batch_seqs))

        _, _, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = classifier.esm_model(tokens, repr_layers=[30])
            embeddings = results["representations"][30].mean(1).cpu().numpy()

        all_embeddings.extend(embeddings)
        all_ids.extend(batch_ids)
        all_sequences.extend(batch_seqs)

    # Create DataFrame
    embeddings_df = pd.DataFrame(all_embeddings)
    embeddings_df.insert(0, "Seq_id", all_ids)
    embeddings_df.insert(1, "Sequence", all_sequences)

    # Save
    embeddings_df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved embeddings for {len(embeddings_df)} sequences to {output_csv}")


def pred_prot_emb(input_csv, model_path, output_csv):
    """ Predict protein embeddings using a trained ET model and selected features."""
  
    df = input_csv

    clf = joblib.load(model_path)

    drop_cols = ['Seq_id','ID', 'Sequence', 'Pattern ID', 'Start', 'End']
    X_test = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Predict probabilities
    y_p_score = clf.predict_proba(X_test)

    # Extract probability of positive class
    df_pred = pd.DataFrame(y_p_score[:, -1])

    # Save predictions
    df_pred.to_csv(output_csv, index=False, header=False)


def class_assignment(file1,thr,out,wd):
    df1 = pd.read_csv(file1, header=None)
    df1.columns = ['Pred score']
    cc = []
    for i in range(0,len(df1)):
        if df1['Pred score'][i]>=float(thr):
            cc.append('Edible')
        else:
            cc.append('Non-Edible')
    df1['Prediction'] = cc
    df1 =  df1.round(3)
    output_file = os.path.join(wd, out)
    df1.to_csv(output_file, index=None)



def seq_pattern(file1, file2, num):
    df1 = file1.copy()
    df1.columns = ['Seq']
    df2 = file2.copy()
    df2.columns = ['Name']

    total_iterations = sum(len(seq) for seq in df1['Seq'])  # total number of j-loops
    print(f"[INFO] Processing {len(df1)} sequences with total {total_iterations} sliding windows.")

    cc, dd, ee, ff, gg = [], [], [], [], []

    with tqdm(total=total_iterations, desc="Scanning patterns", unit="pattern") as pbar:
        for i in range(len(df1)):
            seq_str = str(df1['Seq'][i])  # make sure it's string
            for j in range(len(seq_str)):
                xx = seq_str[j:j+num]
                if len(xx) == num:
                    cc.append(df2['Name'][i])
                    dd.append(f'Pattern_{j + 1}')
                    ee.append(xx)
                    ff.append(j + 1)        # start position (1-based)
                    gg.append(j + num)      # end position (1-based)
                pbar.update(1)

    if not cc:  # no matches found
        print("[WARNING] No patterns found matching the given window length.")
        return pd.DataFrame(columns=['SeqID', 'Pattern ID', 'Start', 'End', 'Seq'])

    df3 = pd.concat([
        pd.DataFrame(cc),
        pd.DataFrame(dd),
        pd.DataFrame(ff),
        pd.DataFrame(gg),
        pd.DataFrame(ee)
    ], axis=1)

    df3.columns = ['SeqID', 'Pattern ID', 'Start', 'End', 'Seq']
    print(df3)
    return df3

    
def generate_mutant(original_seq, residues, position):
    std = "ACDEFGHIKLMNPQRSTVWY"
    if all(residue.upper() in std for residue in residues):
        if len(residues) == 1:
            mutated_seq = original_seq[:position-1] + residues.upper() + original_seq[position:]
        elif len(residues) == 2:
            mutated_seq = original_seq[:position-1] + residues[0].upper() + residues[1].upper() + original_seq[position+1:]
        else:
            print("Invalid residues. Please enter one or two of the 20 essential amino acids.")
            return None
    else:
        print("Invalid residues. Please enter one or two of the 20 essential amino acids.")
        return None
    return mutated_seq


def generate_mutants_from_dataframe(df, residues, position):
    mutants = []
    for index, row in df.iterrows():
        original_seq = row[0]
        mutant_seq = generate_mutant(original_seq, residues, position)
        # print('mutant_seq',mutant_seq)
        if mutant_seq is not None:
            mutants.append((original_seq, mutant_seq, position))
    return mutants

def all_mutants(file1,file2):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    cc = []
    dd = []
    ee = []
    df2 = pd.DataFrame(file2)
    df2.columns = ['Name']
    df1 = pd.DataFrame(file1)
    df1.columns = ['Seq']
    for k in range(len(df1)):
        cc.append(df1['Seq'][k])
        dd.append('Original_'+'Seq'+str(k+1))
        ee.append(df2['Name'][k])
        for i in range(0,len(df1['Seq'][k])):
            for j in std:
                if df1['Seq'][k][i]!=j:
                    #dd.append('Mutant_'+df1['Seq'][k][i]+str(i+1)+j+'_Seq'+str(k+1))
                    dd.append('Mutant_'+df1['Seq'][k][i]+str(i+1)+j)
                    cc.append(df1['Seq'][k][:i] + j + df1['Seq'][k][i + 1:])
                    ee.append(df2['Name'][k])
    xx = pd.concat([pd.DataFrame(ee),pd.DataFrame(dd),pd.DataFrame(cc)],axis=1)
    xx.columns = ['SeqID','Mutant_ID','Seq']
    return xx


##################################################################################################################



def main():
    
    parser = argparse.ArgumentParser(description='Please provide the following arguments. Please make the suitable changes in the envfile provided in the folder.') 

    
    parser.add_argument("-i", "--input", type=str, required=True, help="Input: protein sequence in FASTA format or single sequence per line in single letter code")
    parser.add_argument("-o", "--output",type=str, default="outfile.csv", help="Output: File for saving results by default outfile.csv")
    parser.add_argument("-t","--threshold", type=float,  help="Threshold: Value between 0 to 1 by default 0.5. ")
    parser.add_argument("-j", "--job", type=int, choices=[1, 2, 3, 4], default=1, help="Job Type: 1: Prediction, 2: Protein Scanning, 3: Design, 4: Design all possible mutants")
    parser.add_argument("-m", "--model", type=int, default=1, choices=[1, 2], help="Model selection: Model 1: ESM2-t30, For proteins  Model 2: ET, For combined")
    parser.add_argument("-p", "--Position", type=int, help="Position of mutation (1-indexed)")
    parser.add_argument("-r", "--Residues", type=str, help="Mutated residues (one or two of the 20 essential amino acids in upper case)")
    parser.add_argument("-w","--winleng", type=int, choices =range(8, 21), help="Window Length: 8 to 20 (scan mode only), by default 12")
    parser.add_argument("-wd", "--working", type=str, default=os.getcwd(),required=True, help="Working Directory: Location for writing results")
    parser.add_argument("-d","--display", type=int, choices = [1,2], default=2, help="Display: 1:Edible, 2: All peptides, by default 2")


    args = parser.parse_args()

    # Parameter initialization or assigning a variable for command-level arguments

    Sequence= args.input        # Input variable 
    
    # Output file 
    if args.output is None:
        result_filename = "output.csv"
    else:
        result_filename = args.output
            
    # Threshold
    if args.threshold is None:
        Threshold = {1: 0.45, 2: 0.55, 3: 0.5}.get(args.model, 0.5)
    else:
        Threshold = float(args.threshold)


    # Model
    if args.model is None:
        Model = 1
    else:
        Model = int(args.model)

    # Display
    if args.display is None:
        dplay = 2
    else:
        dplay = int(args.display)

    # Job Type
    if args.job is None:
        Job = 1
    else:
        Job = args.job

    # # Window Length 
    if args.winleng == None:
        Win_len = int(12)
    else:
        Win_len = int(args.winleng)


    if args.Position is None:
        position = 1
    else:
        position = args.Position

    if args.Residues is None:
        residues = "AA"
    else:
        residues = args.Residues

    # Working Directory
    wd = args.working

    print('\nSummary of Parameters:')
    print(f"Input File: {args.input} ; Model: {args.model} ; Job: {args.job} ; Threshold: {Threshold}")
    print(f"Output File: {args.output} ; Display: {args.display}")

    #------------------ Read input file ---------------------
    f=open(Sequence,"r")
    len1 = f.read().count('>')
    f.close()

    # Use the `readseq` function to process the input file
    seqid, seq = readseq(Sequence)


    # Combine sequence IDs and sequences into a single DataFrame
    seqid_list = seqid.iloc[:, 0].tolist()
    seq_list = seq.iloc[:, 0].tolist()
    CM = pd.DataFrame({"Seq_id": seqid_list, "Sequence": seq_list})

    # Save to CSV file
    CM.to_csv(f"{wd}/Sequence_1.csv", index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = f"{nf_path}/Model/saved_model_t30"
            
    # Load ESM model and alphabet
    esm_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    esm_model = esm_model.to(device)
    alphabet = torch.load(os.path.join(model_save_path, "esm_alphabet.pth"))
    batch_converter = alphabet.get_batch_converter()
    
    classifier = ProteinClassifier(esm_model, embedding_dim=640, num_classes=2)
    classifier.load_state_dict(torch.load(os.path.join(model_save_path, "classifier_state.pth"), map_location=device))
    classifier = classifier.to(device)
                
                #======================= Prediction Module starts from here =====================
    if Job == 1:

              
        if Model == 1:
                print('\n======= You are using the Prediction module of EDIpropred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
                print('==== Prediction using ESM2-t30 model: Processing sequences please wait ...')
                df = pd.read_csv(f"{wd}/Sequence_1.csv")
                predict_from_dataframe(df, classifier, batch_converter, device, Threshold, f"{wd}/{result_filename}")
            
                df13 = pd.read_csv(f"{wd}/{result_filename}")
                
                df13.columns = ['SeqID', 'Sequence', 'ESM Score', "Prediction"]
                df13['SeqID'] = df13['SeqID'].astype(str).str.replace('>', '', regex=False)
                df13 = round(df13, 3)
                df13.to_csv(f"{wd}/{result_filename}", index=None)

                if dplay == 1:
                    df13 = df13.loc[df13.Prediction == "Edible"]
                    print(df13)
                elif dplay == 2:
                    df13=df13
                    print(df13)
                
                # Clean up temporary files 
                # os.remove(f'{wd}/Sequence_1.csv')  

             
        if Model == 2:
                print('\n======= You are using the Prediction module of EDIpropred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
                print('==== Predicting using ET model with ESM2-t30 embeddings as features: Processing sequences please wait ...')
                #seq = seq.iloc[:, 0].tolist()  # Converts the first column to a list
                extract_embeddings(f'{wd}/Sequence_1.csv',classifier, batch_converter, device, f'{wd}/esm12_embeddings.csv')
            
                df1 = pd.read_csv(f'{wd}/esm2_embeddings.csv')
            
                pred_prot_emb(df1, f"{nf_path}/Model/extra_trees_model.pkl", f'{wd}/seq.pred')
            
                class_assignment(f'{wd}/seq.pred', Threshold, 'seq.out', wd)
            
                df2 = df1[['Seq_id','Sequence']]
                df3 = pd.read_csv(f'{wd}/seq.out')
                df3 = round(df3,3)
                df4 = pd.concat([df2,df3],axis=1)
                df4.columns = ['SeqID','Sequence','Pred Score','Prediction']
                df4 = round(df4,3)
                df4.to_csv(f"{wd}/{result_filename}", index=None)
                if dplay == 1:
                    df4 = df4.loc[df4.Prediction=="Edible"]
                    print(df4)
                elif dplay == 2:
                    df4=df4
                    print(df4)
            
                # Clean up temporary files 
                os.remove(f'{wd}/Sequence_1.csv') 
                os.remove(f'{wd}/seq.out')
                os.remove(f'{wd}/seq.pred')
                os.remove(f'{wd}/esm2_embeddings.csv')
                shutil.rmtree(f'{wd}/esm2-t30_embeddings', ignore_errors=True)   

         
            
                #======================= Protein Scanning Module starts from here =====================                            
    elif Job == 2:
                  
        if Model == 1:
            print('\n======= You are using the Protein Scanning module of EDIpropred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Scanning through ESM2-t30 model: Processing sequences please wait ...')
            print(seq)
            df_1 = seq_pattern(seq,seqid,Win_len)
            df_1.columns = ['Seq_id','Pattern ID', 'Start', 'End', 'Sequence']

            predict_from_dataframe(df_1, classifier, batch_converter, device, Threshold, f"{wd}/{result_filename}")
            
            df13 = pd.read_csv(f"{wd}/{result_filename}")
            
            df13 = df13.drop(columns=["SeqID", "Sequence"])
            df21 = pd.concat([df_1, df13], axis=1)
            df21["Seq_id"] = df21["Seq_id"].str.lstrip(">")
            df21.columns = ['SeqID','Pattern ID', 'Start', 'End', 'Sequence', 'ESM Score', "Prediction"]
            df21 = round(df21, 3)
            df21.to_csv(f"{wd}/{result_filename}", index=None)

            if dplay == 1:
                df21 = df21.loc[df21.Prediction == "Edible"]
                print(df21)
            elif dplay == 2:
                df21=df21
                print(df21)

            # Clean up temporary files 
            os.remove(f'{wd}/Sequence_1.csv')

               
        if Model == 2:
                print('\n======= You are using the Protein Scanning module of EDIpropred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
                print('==== Prediction using ET model with ESM2-t30 embeddings as features: Processing sequences please wait ...')                        
                # print(seq)
                df_1 = seq_pattern(seq,seqid,Win_len)
                df_1.columns = ['Seq_id','Pattern ID', 'Start', 'End', 'Sequence']
                # print(df_1)
                extract_embeddings(df_1,classifier, batch_converter, device, f'{wd}/esm22_embeddings.csv')
    
                df = pd.read_csv(f'{wd}/esm22_embeddings.csv')
                # df = df.rename(columns={"ID": "Seq_id"})
                # print(df_1['Seq_id'])
                # print(df['Seq_id'])
                df11 = df_1.merge(df, on=['Seq_id','Sequence'], how="inner")
                df11.to_csv(f'{wd}/pattern_embeddings_in_order.csv')
            
                pred_prot_emb(df11, f"{nf_path}/Model/extra_trees_model.pkl",f'{wd}/seq.pred')
            
                class_assignment(f'{wd}/seq.pred', Threshold, 'seq.out', wd)
            
                df3 = pd.read_csv(f'{wd}/seq.out')
            
                df4 = pd.concat([df_1,df3],axis=1)
            
                df4["Seq_id"] = df4["Seq_id"].str.lstrip(">")
                df4.columns = ['SeqID','Pattern ID', 'Start', 'End', 'Sequence', 'ML Score', "Prediction"]
                df4 = round(df4,3)
                df4.to_csv(f"{wd}/{result_filename}", index=None)
                if dplay == 1:
                    df4 = df4.loc[df4.Prediction=="Edible"]
                    print(df4)
                elif dplay == 2:
                    df4=df4
                    print(df4)
                
                # Clean up temporary files 
                os.remove(f'{wd}/Sequence_1.csv')
                os.remove(f'{wd}/seq.out')
                os.remove(f'{wd}/seq.pred')  
                os.remove(f'{wd}/pattern_embeddings_in_order.csv')  
                os.remove(f'{wd}/esm22_embeddings.csv')
                shutil.rmtree(f'{wd}/esm2-t30_embeddings', ignore_errors=True)
        

            
            #======================= Design Module starts from here =====================
    elif Job == 3:
                      
        if Model == 1:
            print('\n======= You are using the Design Module of EDIpropred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Prediction using ESM2-t30 model: Processing sequences please wait ...')
          
            mutants = generate_mutants_from_dataframe(seq, residues, position)
            result_df = pd.DataFrame(mutants, columns=['Original Sequence','seq', 'Position'])
            
            out_len_mut = pd.DataFrame(result_df['seq'])
            
            df = pd.read_csv(f"{wd}/Sequence_1.csv")
            predict_from_dataframe(df, classifier, batch_converter, device, Threshold, f"{wd}/{result_filename}")
            
            out_len_mut.columns = ['Sequence']
            mut_df = pd.concat([df['Seq_id'], out_len_mut],axis=1)
            
            predict_from_dataframe(mut_df, classifier, batch_converter, device, Threshold, f"{wd}/out_m.csv")
            
            df13 = pd.read_csv(f"{wd}/{result_filename}")
            df14 = pd.read_csv(f"{wd}/out_m.csv")
            df_id = df14['SeqID']
            df14 = df14.drop(columns=['SeqID'])

            df15 = pd.concat([df13, df14], axis=1)
            
            seqid_1 = pd.Series(df_id, name="SeqID")
            
            df15 = pd.concat([seqid_1, result_df['Original Sequence'], df13['ESM Score'], df13['Prediction'], 
                                result_df['seq'], result_df['Position'], df14['ESM Score'], df14['Prediction']], axis=1)
            df15.columns = ['SeqID', 'Original Sequence', 'ESM Score', 'Prediction', 'Mutant Sequence', 'Position', 'Mutant ESM Score', 'Mutant Prediction']
            df15['SeqID'] = df15['SeqID'].str.replace('>', '')
            df15 = round(df15, 3)
            df15.to_csv(f"{wd}/{result_filename}", index=None)
            if dplay == 1:
                df15 = df15.loc[df15['Mutant Prediction'] == "Edible"]
                print(df15)
            elif dplay == 2:
                df15 = df15
                print(df15)
        
            # Clean up temporary files
            os.remove(f'{wd}/out_m.csv')
            os.remove(f'{wd}/Sequence_1.csv')
            

        if Model == 2: 
            print('\n======= You are using the Design Module of EDIpropred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Prediction using ET model with ESM2-t30 embeddings as features: Processing sequences please wait ...')
            mutants = generate_mutants_from_dataframe(seq, residues, position)
            result_df = pd.DataFrame(mutants, columns=['Original Sequence', 'Mutant Sequence', 'Position'])
            result_df['Mutant Sequence'].to_csv(f'{wd}/out_len_mut.csv', index=None, header=None)
            
            out_len_mut = pd.DataFrame(result_df['Mutant Sequence'])
            ori_Sequence = pd.DataFrame(result_df["Original Sequence"])
            
            
            #prediction for original sequence
            seqid.columns = ['Seq_id']
            ori_Sequence = ori_Sequence.rename(columns={"Original Sequence": "Sequence"})
            df =pd.concat([seqid,ori_Sequence],axis=1)
            extract_embeddings(df, classifier, batch_converter, device, f'{wd}/esm32_embeddings.csv')
            ori_Sequence_embeddings = pd.read_csv(f'{wd}/esm32_embeddings.csv')
            
            # ori_Sequence_embeddings = ori_Sequence_embeddings.rename(columns={"ID": "Seq"})
            
            df11 = ori_Sequence_embeddings
            df11.to_csv(f'{wd}/ori_Sequence_embeddings_in_order.csv',  index=False)
            
            pred_prot_emb(df11, f"{nf_path}/Model/extra_trees_model.pkl",f'{wd}/seq.pred')
            class_assignment(f'{wd}/seq.pred', Threshold, 'seq.out', wd)
            
            df = pd.read_csv(f'{wd}/seq.out')
            df3 = round(df,3)
            
            df4 = pd.concat([ori_Sequence,df3],axis=1)
            df4.columns = ['Original Sequence','ML Score','Prediction']

            ##prediction for mutant sequence
            Mut_Sequence = pd.DataFrame(result_df["Mutant Sequence"])
            Mut_Sequence = Mut_Sequence.rename(columns={"Mutant Sequence": "Sequence"})
            df =pd.concat([seqid,Mut_Sequence],axis=1)
            extract_embeddings(df, classifier, batch_converter, device, f'{wd}/esm33_embeddings.csv')
            Mut_Sequence_embeddings  = pd.read_csv(f'{wd}/esm33_embeddings.csv')
            
            df12 = Mut_Sequence_embeddings
            df12.to_csv(f'{wd}/Mut_Sequence_embeddings_in_order.csv',  index=False)
            
            pred_prot_emb(df12, f"{nf_path}/Model/extra_trees_model.pkl",f'{wd}/seq.pred')
            class_assignment(f'{wd}/seq.pred',Threshold, 'seq.out', wd)
            
            df = pd.read_csv(f'{wd}/seq.out')
            df33 = round(df,3)
            
            df44 = pd.concat([Mut_Sequence,df33],axis=1)
            df44.columns = ['Mutant Sequence','ML Score','Prediction']

            ##prediction of original sequence + mutant sequence
            df55 = pd.concat([seqid,df4,df44],axis=1)
            df55.columns = ['SeqID','Original Sequence','ML Score','Prediction','Mutant Sequence','Mutant ML Score','Mutant Prediction']
            df55["SeqID"] = df55["SeqID"].str.lstrip(">")
            df55.to_csv(f"{wd}/{result_filename}", index=None)

            if dplay == 1:
                df55 = df55.loc[df55['Mutant Prediction'] == "Edible"]
                print(df55)
            elif dplay == 2:
                df55 = df55
                print(df55)

        
            # Clean up temporary files 
            os.remove(f'{wd}/Sequence_1.csv')
            os.remove(f'{wd}/seq.out')
            os.remove(f'{wd}/seq.pred')  
            # os.remove(f'{wd}/Mut_Sequence.fasta') 
            # os.remove(f'{wd}/ori_Sequence.fasta')
            os.remove(f'{wd}/out_len_mut.csv')
            os.remove(f'{wd}/Mut_Sequence_embeddings_in_order.csv') 
            os.remove(f'{wd}/ori_Sequence_embeddings_in_order.csv')  
            os.remove(f'{wd}/esm32_embeddings.csv')
            os.remove(f'{wd}/esm33_embeddings.csv')
            shutil.rmtree(f'{wd}/esm2-t30_embeddings', ignore_errors=True) 

            


    #======================= Design Module for all possible mutants starts from here =====================      
    elif Job == 4:
                   
        if Model == 1:
                print('\n======= You are using the Design Module for all possible mutants of EDIpropred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
                print('==== Prediction using ESM2-t30 model: Processing sequences please wait ...')
                muts = all_mutants(seq, seqid)
                muts.to_csv(f'{wd}/muts.csv', index=None, header=None)  
                seq=muts['Seq'].tolist()
                seqid_1=muts['Mutant_ID'].tolist()
                combined_df = pd.DataFrame({
                            "Seq_id": seqid_1,
                            "Sequence": seq
                        })

                predict_from_dataframe(combined_df, classifier, batch_converter, device, Threshold, f"{wd}/{result_filename}")
                df13 = pd.read_csv(f"{wd}/{result_filename}")
            
                df13 = df13[["ESM Score", "Prediction"]]
                df14 = pd.concat([muts,df13],axis=1)
            
                df14.columns = [ 'SeqID', 'MutantID', 'Sequence', 'ESM Score', "Prediction"]
                df14["SeqID"] = df14["SeqID"].apply(lambda x: str(x).lstrip(">") if pd.notnull(x) else "")
                df14.to_csv(f"{wd}/{result_filename}", index=None)
                if dplay == 1:
                    df14 = df14.loc[df14.Prediction == "Edible"]
                    print(df14)
                elif dplay == 2:
                    df14=df14
                    print(df14)   

                # Clean up temporary files 
                os.remove(f'{wd}/Sequence_1.csv') 
                os.remove(f'{wd}/muts.csv')    

            
                 
        if Model == 2:
                print('\n======= You are using the Design module for all possible mutants of EDIpropred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
                print('==== Prediction using ET model with ESM2-t30 embeddings as features: Processing sequences please wait ...')
                            
                muts = all_mutants(seq, seqid)
                muts.to_csv(f'{wd}/muts.csv', index=None, header=None) 
                df = muts[['SeqID','Seq']]
                df.columns = ['Seq_id','Sequence']
                extract_embeddings(df, classifier, batch_converter, device, f'{wd}/esm42_embeddings.csv')
                muts_embeddings  = pd.read_csv(f'{wd}/esm42_embeddings.csv')
                
                df12 = muts_embeddings
                df12.to_csv(f'{wd}/muts_embeddings_in_order.csv',  index=False)
            
                pred_prot_emb(df12, f"{nf_path}/Model/extra_trees_model.pkl",f'{wd}/seq.pred')
                class_assignment(f'{wd}/seq.pred', Threshold, 'seq.out', wd)
                df = pd.read_csv(f'{wd}/seq.out')
                df33 = round(df,3)
            
                df44 = pd.concat([muts,df33],axis=1)
                df44.columns = ['SeqID','MutantID','Sequence','ML Score','Prediction']
                df44['SeqID'] = df44['SeqID'].str.replace('>','')

                df44.to_csv(f"{wd}/{result_filename}", index=None)
                if dplay == 1:
                    df44 = df44.loc[df44.Prediction=="Edible"]
                    print(df44)
                elif dplay == 2:
                    df44 = df44
                    print(df44) 

                # Clean up temporary files 
                os.remove(f'{wd}/Sequence_1.csv')
                os.remove(f'{wd}/seq.out')
                os.remove(f'{wd}/seq.pred')  
                # os.remove(f'{wd}/muts.fasta') 
                os.remove(f'{wd}/muts.csv')
                os.remove(f'{wd}/muts_embeddings_in_order.csv')  
                os.remove(f'{wd}/esm42_embeddings.csv')
                shutil.rmtree(f'{wd}/esm2-t30_embeddings', ignore_errors=True)  
      
if __name__ == "__main__":
    main()
