import numpy as np
import matplotlib.pyplot as plt
import pretty_midi as pm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilProcessing import parseCOMU

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

def viewPR():
    # File paths for the original MIDI and the quantized version
    midi_path = "../midiDataTest/commu00001.mid"
    npz_path = "../commu00001_4bin.npz"

    notes = parseCOMU(midi_path)        # Extract notes from MIDI file
    roll_midi = notes_to_roll(notes)    # Convert to piano roll (no quantization)

    roll_npz, q = npz_to_roll(npz_path) # Convert quantized data to piano roll

    print(roll_midi, roll_npz)

    plt.figure(figsize=(12, 8))  # Create a new figure

    # Plot the original MIDI piano roll
    plt.subplot(2, 1, 1)
    plt.imshow(roll_midi, aspect="auto", origin="lower", cmap="gray_r")
    plt.title("MIDI sem quantização")
    plt.ylabel("Pitch")
    plt.xlabel("Tempo (frames)")

    # Plot the quantized piano roll from NPZ
    plt.subplot(2, 1, 2)
    plt.imshow(roll_npz, aspect="auto", origin="lower", cmap="gray_r")
    plt.title("NPZ quantizado")
    plt.ylabel("Pitch")
    plt.xlabel(f"Posição quantizada ({q} subdivisões)")

    plt.tight_layout()  # Adjust subplot layout
    #plt.savefig("pianorolls")          # Display the figure (only works in GUI environments)
    plt.savefig(f"pianoRoll_{os.path.splitext(os.path.basename(midi_path))[0]}.png")

if __name__ == "__main__":
    viewPR()  # Run the piano roll viewer if script is executed directly