import gdown
import zipfile
import pandas as pd
from pathlib import Path
import os
from .avaliacao import * 
import numpy as np
import re
import sys
import csv

output_dir = Path(__file__).parent.parent.parent / "data/raw/foldsDeTreinamentoUllysses"

document_id = "1AstJwYZhiPUx-wKahwOxrTuJ7td5rLMi"

url = f"https://drive.google.com/uc?export=download&id={document_id}"

zip_filename = "Ulysses-RFCorpus.zip"

dataset_name = "bills_dataset.csv"
queries_name = "fold_teste.csv"

def dir_exists(path=output_dir):
    Path(path).mkdir(parents=True, exist_ok=True)

def get_ulysses_data():

    dir_exists()
    zip_path = os.path.join(output_dir, zip_filename)
    gdown.download(url, zip_path, quiet=False)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)


def get_bills():
    # First set the CSV field size limit using Python's csv module
    import csv
    csv.field_size_limit(2147483647)  # maximum value for 32-bit systems
    
    if not os.path.exists(os.path.join(output_dir, dataset_name)):
        get_ulysses_data()
        
    try:
        # First verify the file can be read with csv module
        with open(os.path.join(output_dir, dataset_name), 'r', encoding='utf-8') as f:
            # Test reading first line
            reader = csv.reader(f)
            try:
                next(reader)
            except csv.Error as e:
                print(f"CSV read error: {str(e)}")
                return pd.DataFrame()
            
            # Reset file pointer
            f.seek(0)
            
            # Now read with pandas
            chunks = pd.read_csv(
                f,
                encoding="utf-8", 
                quotechar='"', 
                on_bad_lines="warn", 
                usecols=['name', 'text', 'code'], 
                chunksize=10000, 
                delimiter=',', 
                engine='python'
            )
            dataset = pd.concat(chunks, ignore_index=True)
            print(f"Current working directory bills inside: {os.getcwd()}")
            print("Successfully loaded bills dataset")
            return dataset
            
    except Exception as e:
        print(f"Failed at retrieving bills dataset: {str(e)}")
        return pd.DataFrame()

def get_queries_data():

    queries_path = os.path.join(output_dir, queries_name)

    if(not os.path.exists(queries_path)):
        get_ulysses_data()

    try:
        print(f"Current working directory getting queries: {os.getcwd()}")
        read_results = qrel_from_ulysses(queries_path)
        return read_results
    except:
        print(f"Failed at retrieving {queries_name}")
        return None
    
def clean_text(text):
    # Remove quebras de linha seguidas de letras minúsculas (indicando continuação da frase)
    text = re.sub(r'(?<=[a-z,])\n(?=[a-z])', ' ', text)
    # Remove quebras de linha seguidas de pontuação
    text = re.sub(r'\n(?=[.,;:])', '', text)
    # Remove múltiplos espaços em branco
    text = re.sub(r'\s+', ' ', text)
    # Remove espaços no início e no final do texto
    text = text.strip()
    return text

def bbla():
    print(output_dir)
    print(output_dir / dataset_name)
    print(output_dir / queries_name)
