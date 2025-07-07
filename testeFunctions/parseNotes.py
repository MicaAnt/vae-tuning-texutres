import numpy as np
import matplotlib.pyplot as plt
import pretty_midi as pm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilProcessing import parseCOMU

def parse():

    midi_path = "../midiDataTest/commu00001.mid"

    print(parseCOMU(midi_path))

if __name__ == "__main__":
    parse()  # Run visualisation if script is executed directly