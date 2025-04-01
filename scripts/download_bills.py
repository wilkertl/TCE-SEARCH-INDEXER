import gdown
import os

url = "https://drive.google.com/uc?id=1yuRCNKLphpk6FPhAfBD4_yygnOPFilHj"
output = "./bills_dataset.csv"

os.makedirs(os.path.dirname(output), exist_ok=True)
gdown.download(url, output, quiet=False)
