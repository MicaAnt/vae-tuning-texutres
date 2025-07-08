# hello funds
import csv
import ast

import sys

sys.path.append("../")

from utilProcessing import get_fund

def test_get_fund(csv_path="../midiDataTest/commu_meta.csv", track_id="commu00002"):
    # Executa a função e imprime o vetor de fundamentais
    fundamentals = get_fund(csv_path=csv_path, track_id=track_id)
    print("\n🎯 Resultados de get_fund:")
    print(fundamentals)

    # Lê o CSV e mostra a linha correspondente
    #print("\n📄 Linha completa do CSV:")
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        row = next((r for r in reader if r.get("id") == track_id), None)

    if row is None:
        print(f"Track id {track_id} not found.")
        return

    #for key, value in row.items():
     #   print(f"{key}: {value}")

    # Mostra a progressão de acordes de forma legível
    print("\n🎼 Conteúdo de chord_progressions:")
    prog = ast.literal_eval(row["chord_progressions"])
    print(prog[0])  # imprime a sequência de acordes como lista


# Executar teste (caso use __main__)
if __name__ == "__main__":
    test_get_fund()