### FUNCOES SECUNDARIAS PARA O PITCH PROCESSING ###

import pretty_midi as pm
import numpy as np
# for harmonic processing
import csv
import ast
import numpy as np
from pychord import Chord, utils

def midi_to_timeBeats(midi_file):
    """
    Processes a MIDI file and returns a table with beats and downbeats as a numpy array.

    Parameters:
        midi_file (str): Path to the MIDI file.

    Returns:
        np.ndarray: Array with two columns:
                    - 1st column ("timeBeats"): beat times in seconds.
                    - 2nd column ("downbeats"): 1.0 if the beat is a downbeat, otherwise 0.0.
    """

    midiObject = pm.PrettyMIDI(midi_file) # Load the MIDI file
    firstOnset = midiObject.get_onsets()[0] # Get the first Onset
    timeBeats = midiObject.get_beats(firstOnset) # Get beat times
    downbeats = midiObject.get_downbeats() # Get downbeats
    is_downbeat = np.isin(timeBeats, downbeats).astype(float) # Create a vector indicating if the beat is a downbeat
    timeBeats_array = np.column_stack((timeBeats, is_downbeat)) # Combine into a NumPy array

    return timeBeats_array

def beat_position_list(input_array):
    """
    Assigns a beat position index to each beat in the input array, starting from 0.

    Parameters:
        input_array (np.ndarray): A 2D array with two columns:
            - 1st column ("timeBeats"): Beat times in seconds.
            - 2nd column ("downbeats"): 1.0 if the beat is a downbeat, otherwise 0.0.

    Returns:
        np.ndarray: A 2D array with three columns:
            - 1st column ("beat position"): Beat position index, starting at 0.
            - 2nd column ("timeBeats"): Beat times in seconds.
            - 3rd column ("downbeats"): 1.0 if the beat is a downbeat, otherwise 0.0.
    """
    # Validate the input is a 2D array with two columns
    if not isinstance(input_array, np.ndarray) or input_array.shape[1] != 2:
        raise ValueError("Input must be a 2D numpy array with two columns.")

    # Number of beats in the input
    num_beats = input_array.shape[0]

    # Generate the beat position indices, starting from 0
    #beat_positions = np.arange(num_beats)
    beat_positions = np.arange(1, num_beats + 1)

    # Combine the beat positions with the original array
    result = np.column_stack((beat_positions, input_array))

    return result

def quantize_time(input_array, n_quantization):
    """
    Generates a quantized table with beat subdivisions based on the input beats and quantization value.

    Parameters:
        input_array (np.ndarray): A 2D array with three columns:
            - 1st column ("beat position"): Beat position index, starting at 0.
            - 2nd column ("timeBeats"): Beat times in seconds.
            - 3rd column ("downbeats"): 1.0 if the beat is a downbeat, otherwise 0.0.
        n_quantization (int): Number of subdivisions for each beat.

    Returns:
        np.ndarray: A 2D array with four columns:
            - 1st column ("beat position"): Beat position index.
            - 2nd column ("beatSubdivision"): Subdivision index for each beat (0 to n_quantization-1).
            - 3rd column ("qTimeBeats"): Quantized beat times in seconds.
            - 4th column ("quantization"): Value of n_quantization (repeated for all rows).
    """
    # Validate the input is a 2D array with three columns
    if not isinstance(input_array, np.ndarray) or input_array.shape[1] != 3:
        raise ValueError("Input must be a 2D numpy array with three columns.")

    # Extract columns from the input array
    beat_positions = input_array[:, 0]  # Beat position index
    time_beats = input_array[:, 1]      # Beat times in seconds

    # Initialize lists to store the output data
    quantized_positions = []
    subdivisions = []
    quantized_times = []
    quantization_values = []

    # Iterate over each beat
    for i in range(len(beat_positions) - 1): # Iterate over each beat, except the last one, because there is no next beat to calculate the interval-
        # Current and next beat times
        current_time = time_beats[i]
        next_time = time_beats[i + 1]

        # Compute the quantized time steps for the current beat
        step = (next_time - current_time) / n_quantization
        for subdivision in range(n_quantization):
            quantized_positions.append(beat_positions[i]) # Append the current beat position index to the list of quantized positions
            subdivisions.append(subdivision) # Append the current subdivision index (0 to n_quantization - 1) to the list of subdivisions
            quantized_times.append(current_time + subdivision * step) # Compute the quantized time for the current subdivision and append it to the quantized times list
            quantization_values.append(n_quantization) # Append the quantization value (n_quantization) to the quantization values list

    # Handle the last beat (no next time to interpolate)
    quantized_positions.append(beat_positions[-1])
    subdivisions.append(0)
    quantized_times.append(time_beats[-1])
    quantization_values.append(n_quantization)

    # Combine results into a single numpy array
    result = np.column_stack((
        quantized_positions,
        subdivisions,
        quantized_times,
        quantization_values
    ))

    return result

def processNoteList(note_list, quantization_data):
    """
    Process a list of MIDI notes and quantize their start and end times based on quantization data.

    Parameters:
    ----------
    note_list : list
        A list of MIDI notes, where each note is represented as a named tuple or similar object with:
        - start (float): Start time of the note in seconds.
        - end (float): End time of the note in seconds.
        - pitch (int): Pitch of the note.
        - velocity (int): Velocity of the note.

    quantization_data : np.ndarray
        A 2D array with four columns:
        - 1st column ("beat position"): Beat position index.
        - 2nd column ("beatSubdivision"): Beat subdivision index (starting from 0).
        - 3rd column ("qTimeBeats"): Quantized beat times in seconds.
        - 4th column ("quantization"): Number of quantizations.

    Returns:
    -------
    np.ndarray
        A 2D array with eight columns:
        - 1st column ("beatStart position"): Beat position index of the start time.
        - 2nd column ("beatStartSubdivision"): Beat subdivision index of the start time.
        - 3rd column ("quantization"): N of the quantization for the start.
        - 4th column ("beatEndPosition"): Beat position index of the end time.
        - 5th column ("beatEndSubdivision"): Beat subdivision index of the end time.
        - 6th column ("quantization"): N of the quantization for the end.
        - 7th column ("pitch"): Pitch of the note.
        - 8th column ("velocity"): Velocity of the note.
    """
    quantized_notes = []

    # Extract relevant columns from quantization data for easier access
    q_times = quantization_data[:, 2]  # Quantized times in seconds
    beat_positions = quantization_data[:, 0]  # Beat position indices
    beat_subdivisions = quantization_data[:, 1]  # Beat subdivision indices
    quantizations = quantization_data[:, 3]  # Quantization levels

    # Process each note in the list
    for note in note_list:
        # Find the closest quantized time for the start time of the note
        start_index = np.argmin(np.abs(q_times - note.start)) # Find the index of the quantized time in q_times closest to the note's start time
        start_beat_position = beat_positions[start_index]
        start_beat_subdivision = beat_subdivisions[start_index]
        start_quantization = quantizations[start_index]

        # Find the closest quantized time for the end time of the note
        end_index = np.argmin(np.abs(q_times - note.end))
        end_beat_position = beat_positions[end_index]
        end_beat_subdivision = beat_subdivisions[end_index]
        end_quantization = quantizations[end_index]

        # Append the quantized note information to the list
        quantized_notes.append([
            start_beat_position, start_beat_subdivision, start_quantization,
            end_beat_position, end_beat_subdivision, end_quantization,
            note.pitch, note.velocity
        ])

    # Convert the quantized notes list to a NumPy array for output
    return np.array(quantized_notes, dtype=np.int32)

def parseCOMU(file_path):

  # Carrega o arquivo MIDI
  midi_obj = pm.PrettyMIDI(file_path)

  # Obtendo as notas

  all_notes =  midi_obj.instruments[0].notes

  return sorted(all_notes, key=lambda note: note.start)

### MAIN FUNCTION PRO PITCH PROCESSING ###

def pitchDataProcessing(midi_file, notes_list, quantization):

  timeBeats_array = midi_to_timeBeats(midi_file)
  beatPosition = beat_position_list(timeBeats_array)
  qTime = quantize_time(beatPosition, quantization)

  return processNoteList(notes_list, qTime)

#----------- tratamento de harmonia

def get_fund(csv_path="./midiDataTest/commu_meta.csv", track_id="commu00001"):
    """Return a vector of chord roots for a given track.

    Parameters
    ----------
    csv_path : str
        Path to the metadata CSV file.
    track_id : str
        ID of the track to parse.

    Returns
    -------
    numpy.ndarray
        An ``n x 1`` array with the root index of each chord.

    Raises
    ------
    ValueError
        If the time signature is not suitable or there aren't exactly
        two chords per beat.
    """
    # Read metadata for the desired track
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        row = next((r for r in reader if r.get("id") == track_id), None)

    if row is None:
        raise ValueError(f"Track id {track_id} not found")

    # Time signature check
    input_signature = row["time_signature"]
    numerator = int(input_signature.split("/")[0])
    if numerator in [2, 4]:
        time_signature = numerator
    else:
        raise ValueError("The time signature is not suitable for the model")

    # Number of chords per beat check
    prog = ast.literal_eval(row["chord_progressions"])
    chord_list = prog[0]
    num_measures = int(row["num_measures"])
    n_chords = len(chord_list)
    if n_chords / time_signature / num_measures != 2:
        raise ValueError("there is not 2 chords per beat")

    # Reduce to one chord per beat
    reduced = [chord_list[i] for i in range(0, n_chords, 2)]

    # Convert chord roots to numeric values
    fundamentals = [utils.note_to_val(Chord(ch).root) for ch in reduced]
    return np.array(fundamentals, dtype=np.int32)