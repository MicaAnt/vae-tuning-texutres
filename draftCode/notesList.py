#import sys
#import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#import numpy as np
#np.set_printoptions(threshold=np.inf, linewidth=np.inf)

#from utilProcessing import parseCOMU, midiFileTo4bin

#midi_path = "../midiDataTest/commu00002.mid"

#rawData = parseCOMU(midi_path)
#print("Raw data", rawData)

#quantiData = midiFileTo4bin(midi_path)
#print("Quanti data", quantiData)

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Adiciona o diretório pai ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilProcessing import parseCOMU, midiFileTo4bin

def format_note(note):
    return f"({note.start:.3f}, {note.end:.3f}, {note.pitch}, {note.velocity})"

def format_quanti(row):
    return f"{row.tolist()}"

def make_raw_quanti_table(raw_data, quanti_data, midi_path):
    max_len = max(len(raw_data), len(quanti_data))
    rows = []

    for i in range(max_len):
        raw_str = format_note(raw_data[i]) if i < len(raw_data) else ""
        quanti_str = format_quanti(quanti_data[i]) if i < len(quanti_data) else ""
        rows.append((raw_str, quanti_str))

    fig, ax = plt.subplots(figsize=(12, min(0.4 * max_len + 1, 20)))
    ax.axis('off')

    table_data = [("RawData (start, end, pitch, velocity)", "QuantiData [sM, sP, r, eM, eP, r, pitch, velocity]")] + rows
    table = ax.table(cellText=table_data, loc="center", cellLoc="left", colWidths=[0.45, 0.5])
    table.scale(1, 1.2)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.tight_layout()

    out_path = f"tabela_{os.path.splitext(os.path.basename(midi_path))[0]}.png"
    plt.savefig(out_path, dpi=300)
    print(f"✅ Tabela salva como imagem em: {out_path}")

if __name__ == "__main__":
    midi_path = "../midiDataTest/commu00001.mid"
    raw_data = parseCOMU(midi_path)
    quanti_data = midiFileTo4bin(midi_path)
    make_raw_quanti_table(raw_data, quanti_data, midi_path)
