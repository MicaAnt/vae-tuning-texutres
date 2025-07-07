import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilProcessing import parseCOMU

def testPitchVelocity(midi_path="../midiDataTest/commu00001.mid", npz_path="../commu00001_4bin.npz"):
    """Check that all pitch/velocity pairs from the MIDI file are present in the
    quantized NPZ data.

    Parameters
    ----------
    midi_path : str
        Path to the original MIDI file without quantization.
    npz_path : str
        Path to the quantized NPZ file.

    Returns
    -------
    str
        English message describing whether the data matches or what is missing.
    """
    # Parse original MIDI notes (unquantized)
    notes = parseCOMU(midi_path)
    midi_pairs = {(note.pitch, note.velocity) for note in notes}

    # Load quantized notes from npz (use first array if key not known)
    data = np.load(npz_path, allow_pickle=True)
    if 'quantized_notes' in data.files:
        quantized = data['quantized_notes']
    else:
        quantized = data[data.files[0]]
    quantized_pairs = {(int(row[6]), int(row[7])) for row in quantized}

    # Determine if any pairs from the original MIDI are missing
    missing = midi_pairs - quantized_pairs
    if not missing:
        return "All pitch and velocity pairs from the MIDI file are present in the quantized data."
    return f"Missing pitch/velocity pairs in the quantized data: {sorted(missing)}"


if __name__ == "__main__":
    message = testPitchVelocity()
    print(message)
