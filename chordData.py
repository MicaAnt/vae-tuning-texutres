import numpy as np
import sys

def inspect_chord(path):
    data = np.load(path, allow_pickle=True)
    
    if "chord" not in data.files:
        print("A chave 'chord' não foi encontrada no arquivo.")
        return

    chord = data["chord"]

    print(f"Chave 'chord' encontrada:")
    print(f"  shape = {chord.shape}")
    print(f"  dtype = {chord.dtype}\n")

    # Mostrar o array completo sem truncamento
    np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)
    print("Valores:\n")
    print(chord)

if __name__ == "__main__":
    path = sys.argv[1]
    inspect_chord(path)
