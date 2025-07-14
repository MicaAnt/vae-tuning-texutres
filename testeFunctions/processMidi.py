import sys

sys.path.append("../")

from utilProcessing import midiFileTo4bin

dados = midiFileTo4bin(midi_path = "../midiDataTest/commu00002.mid")

print(dados)