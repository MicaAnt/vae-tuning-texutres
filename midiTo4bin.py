import argparse
import os
import numpy as np

from utilProcessing import parseCOMU, pitchDataProcessing


def midi_to_4bin(midi_path: str) -> np.ndarray:
    """Convert MIDI to quantized note matrix with 4 subdivisions per beat.

    Parameters
    ----------
    midi_path : str
        Path to MIDI file.

    Returns
    -------
    np.ndarray
        Matrix of shape (n, 8) with quantized note data.
    """
    notes = parseCOMU(midi_path)
    quantized = pitchDataProcessing(midi_path, notes, quantization=4)
    return quantized


def save_npz(data: np.ndarray, out_path: str) -> None:
    """Save the quantized notes to a compressed npz file."""
    np.savez_compressed(out_path, quantized_notes=data)


def main():
    parser = argparse.ArgumentParser(description="Convert MIDI to 4-bin quantized NPZ")
    parser.add_argument("midi_path", help="Path to MIDI file")
    parser.add_argument("out_path", nargs="?", help="Output npz file")
    args = parser.parse_args()

    out_path = args.out_path
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.midi_path))[0]
        out_path = base + "_4bin.npz"

    data = midi_to_4bin(args.midi_path)
    save_npz(data, out_path)
    print(f"Saved quantized notes to {out_path}")


if __name__ == "__main__":
    main()
