import os
import sys
from tqdm import tqdm

from utilProcessing import GenDataSet

def process_folder(folder_path, dataSetPath="./midiDataTest/", csv_path="./midiDataTest/commu_meta.csv", output_dir="./commuTestNPZ"):
    # List all .mid files in the folder
    mid_files = [f for f in os.listdir(folder_path) if f.endswith(".mid")]

    # Process each file with a progress bar
    for filename in tqdm(mid_files, desc="Processing .mid files"):
        # Remove the .mid extension to get the track ID
        trackId = filename.replace(".mid", "")
        # Call your dataset generation function
        GenDataSet(trackId=trackId, dataSetPath=dataSetPath, csv_path=csv_path, output_dir=output_dir)

if __name__ == "__main__":
   # Set default values
    default_input_folder = "./midiDataTest"
    default_output_dir = "./commuTestNPZ"

    # Use provided arguments or fall back to defaults
    folder_to_process = sys.argv[1] if len(sys.argv) > 1 else default_input_folder
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output_dir

    process_folder(folder_to_process, output_dir=output_dir)