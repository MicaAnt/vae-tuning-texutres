import numpy as np
import sys

#np.set_printoptions(threshold=np.inf, linewidth=np.inf)

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def inspect_npz(path= "./dataSet/POP09-PIANOROLL-4-bin-quantization/001.npz"):
    data = np.load(path, allow_pickle=True)
    print(f"Chaves encontradas no arquivo '{path}':\n")
    for key in data.files[:]:  # Mostra só as 3 primeiras chaves
        #print(f"– {key}: shape = {data[key].shape}, dtype = {data[key].dtype}")
        #print(f" valor:\n{data[key]}\n")
        arr = data[key]

        print(f"– {key}: shape = {arr.shape}, dtype = {arr.dtype}")

        if arr.ndim == 0:
            print(f"  Value: {arr.item()}\n")
        else:
            print(f"  First 20 rows:\n{arr[:20]}\n")
        #print(f"– {key}: shape = {arr.shape}, dtype = {arr.dtype}")
        #print(f"20 primeiras linhas:\n{arr[:20]}\n")

if __name__ == "__main__":
    #path = "./commu00002.npz"
    #path = "./dataSet/POP09-PIANOROLL-4-bin-quantization/347.npz" #
    path = sys.argv[1]
    inspect_npz(path)