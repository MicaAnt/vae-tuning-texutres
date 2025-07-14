import numpy as np
import matplotlib.pyplot as plt
import pretty_midi as pm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

from utilProcessing import parseCOMU, midiFileTo4bin

def notes_to_roll(notes, fs=100):
    """Convert a list of PrettyMIDI notes to a piano-roll array."""
    if not notes:
        return np.zeros((128, 0))  # Return empty piano roll if no notes found

    max_time = max(n.end for n in notes)  # Get the end time of the last note
    num_frames = int(np.ceil(max_time * fs))  # Total number of time frames
    roll = np.zeros((128, num_frames))  # Initialize empty piano roll (pitch x time)

    for note in notes:
        start = int(note.start * fs)  # Convert note start time to frame index
        end = int(note.end * fs)      # Convert note end time to frame index
        roll[note.pitch, start:end] = note.velocity  # Fill in the note with its velocity

    return roll

def npz_to_roll(npz_path):
    """Convert quantized npz note data to a piano-roll array."""
    data = np.load(npz_path)  # Load the .npz file containing quantized note data
    arr = data[data.files[0]]  # Take the first array stored in the file

    q = int(arr[0, 2])  # Quantization factor (subdivisions per beat)
    max_index = np.max(arr[:, 3] * q + arr[:, 4])  # Last index in time (beat * q + sub)
    roll = np.zeros((128, int(max_index)))  # Initialize empty piano roll

    for row in arr:
        start = int(row[0] * q + row[1])  # Compute quantized start index
        end = int(row[3] * q + row[4])    # Compute quantized end index
        pitch = int(row[6])               # MIDI pitch
        velocity = row[7]                 # MIDI velocity
        roll[pitch, start:end] = velocity  # Fill piano roll segment

    return roll, q  # Return the roll and the quantization factor

def viewPR_novo():
    # Caminho para o MIDI
    midi_path = "../midiDataTest/commu00003.mid"

    # Gera dados quantizados diretamente
    arr = midiFileTo4bin(midi_path)  # Retorna np.ndarray diretamente
    print("Notas quantizadas:", arr)

    q = int(arr[0, 2])  # Quantização
    max_index = np.max(arr[:, 3] * q + arr[:, 4])
    roll_npz = np.zeros((128, int(max_index)))

    for row in arr:
        start = int(row[0] * q + row[1])
        end = int(row[3] * q + row[4])
        pitch = int(row[6])
        velocity = row[7]
        roll_npz[pitch, start:end] = velocity

    # Roll do MIDI original (sem quantização)
    notes = parseCOMU(midi_path)
    print("Notas sem quantização:", notes)
    roll_midi = notes_to_roll(notes)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(roll_midi, aspect="auto", origin="lower", cmap="gray_r")
    plt.title("MIDI sem quantização")
    plt.ylabel("Pitch")
    plt.xlabel("Tempo (frames)")

    plt.subplot(2, 1, 2)
    plt.imshow(roll_npz, aspect="auto", origin="lower", cmap="gray_r")
    plt.title("NPZ quantizado")
    plt.ylabel("Pitch")
    plt.xlabel(f"Posição quantizada ({q} subdivisões)")

    plt.tight_layout()
    plt.savefig(f"pianoRoll_{os.path.splitext(os.path.basename(midi_path))[0]}.png")
   
if __name__ == "__main__":
    viewPR_novo()