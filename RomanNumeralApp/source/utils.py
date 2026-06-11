import torch
import numpy as np
from typing import Dict, Any
from music21.note import Note

from .constants import (
    TASKS,
    WINDOW_SIZE,
    LABEL_DOMAINS,
    CHROMATIC_SCALE,
    LABEL_PADDING_VALUE
)

get_note_name = lambda x: Note(x).name
get_note_pc = lambda x: Note(x).pitch.pitchClass

def has_valid_tags(theorytab: Dict[str, Any]) -> bool:
    if any([tag in theorytab['tags'] for tag in ['HAS_METER_CHANGE', 'HAS_TEMPO_CHANGE', 'HAS_SWING_TEMPO']]):
        return False
    
    return all([tag in theorytab['tags'] for tag in ['HAS_AUDIO', 'HAS_HARMONY', 'ONLY_COMMON_TIME', 'ONLY_MAJMIN_KEYS']])


def encode_labels(theorytab: Dict[str, Any]) -> np.ndarray:
    num_beats = theorytab['num_beats'] if theorytab['num_beats'] > WINDOW_SIZE else WINDOW_SIZE
    labels = np.full((len(TASKS), num_beats), fill_value=LABEL_PADDING_VALUE, dtype=np.int64)

    for key in theorytab['keys']:
        onset = key['onset']
        offset = key['offset']

        tonic = CHROMATIC_SCALE[key['tonic_pitch_class']]
        labels[0, onset:offset] = LABEL_DOMAINS['global_key'].index(f"{tonic} {key['scale']}")

    chord_tasks = [task for task in TASKS if task != 'global_key']
    for chord in theorytab['chords']:
        onset = chord['onset']
        offset = chord['offset']
        
        for i, task in enumerate(chord_tasks, start=1):
            if task != 'global_key':
                labels[i, onset:offset] = LABEL_DOMAINS[task].index(chord[task])

    return labels
