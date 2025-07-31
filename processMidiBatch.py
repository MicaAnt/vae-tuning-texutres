import os
import sys
from tqdm import tqdm

from utilProcessing import GenDataSet

BATCH_DIR = "./COMMUDataset/batches/"  # Diretório fixo dos batches

def load_batch_ids(batch_filename):
    batch_path = os.path.join(BATCH_DIR, batch_filename)
    with open(batch_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def process_batch(batch_filename="batch_000.txt",
                  dataSetPath="./COMMUDataset/midiFiles/",
                  csv_path="./COMMUDataset/CommuVAEDataset.csv",
                  output_dir="./COMMUDataset/npzFiles/"):

    track_ids = load_batch_ids(batch_filename)

    for trackId in tqdm(track_ids, desc=f"Processing {batch_filename}"):
        GenDataSet(trackId=trackId,
                   dataSetPath=dataSetPath,
                   csv_path=csv_path,
                   output_dir=output_dir)

if __name__ == "__main__":
    # Só o nome do arquivo do batch, ex: batch_003.txt
    batch_filename = sys.argv[1] if len(sys.argv) > 1 else "batch_000.txt"
    process_batch(batch_filename=batch_filename)
