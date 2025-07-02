import numpy as np
import sys

def inspect_npz(path):
    data = np.load(path, allow_pickle=True)
    print(f"Chaves encontradas no arquivo '{path}':\n")
    for key in data.files[:]:  # Mostra só as 3 primeiras chaves
        print(f"– {key}: shape = {data[key].shape}, dtype = {data[key].dtype}")
        print(f" valor:\n{data[key]}\n")

if __name__ == "__main__":
    path = sys.argv[1]
    inspect_npz(path)