import json

# Update the following paths as necessary
PARC_FILEPATH = 'parc.json'
AUDIOS_FILEPATH = 'custom_audios.h5'
SPLITS_FILEPATH = 'dataset/splits.json'
PCSETS_FILEPATH = 'dataset/pcsets.json'

LABELS_FILEPATH = 'segments/labels.h5'
FEATURES_FILEPATH = 'segments/features.h5'

LABEL_SIZES_FILEPATH = 'dataset/metadata/label_sizes.json'
LABEL_DOMAINS_FILEPATH = 'dataset/metadata/label_domains.json'
GENRE_THEORYTAB_IDS_FILEPATH = 'dataset/metadata/genre_theorytab_ids.json'
COMPLEXITY_THEORYTAB_IDS_FILEPATH = 'dataset/metadata/complexity_theorytab_ids.json'

STEP_SIZE = 32
WINDOW_SIZE = 256

SAMPLING_RATE = 44100
VAMP_FEATURE_STEP = 2048 / SAMPLING_RATE

LABEL_PADDING_VALUE = -1
FEATURE_PADDING_VALUE = 0
HALF_DIMINISHED_SYMBOL = 'ø'  # unicode U+00F8

CHROMATIC_SCALE = 'C C# D D# E F F# G G# A A# B'.split()
COMPLEXITIES = ['Beginner', 'Intermediate', 'Advanced I', 'Advanced II']
GENRES = [
    'Alt-Country', 'Alternative', 'Blues', "Children's", 'Classical', 'Country',
    'Dance', 'Disney', 'Electronic', 'Experimental', 'Folk', 'Hip-Hop/Rap', 'Holiday',
    'House', 'Indie', 'J-Pop', 'Jazz', 'K-pop', 'Latin', 'Metal', 'Pop', 'Punk', 'R & B',
    'Reggae', 'Rock', 'Singer-Songwriter', 'Soul', 'Soundtrack', 'Techno', 'Video Game',
    'Vocal', 'World', 'Worship'
]

TASKS = [
    'global_key', 'tonicization', 'root_scale_degree',
    'quality', 'inversion', 'root_pitch_class', 'bass_pitch_class',
    'tonicized_pitch_class', 'roman_numeral'
]

MODE_INTERVALS = {
    'major': [2, 2, 1, 2, 2, 2],
    'minor': [2, 1, 2, 2, 1, 2],
    'dorian': [2, 1, 2, 2, 2, 1],
    'phrygian': [1, 2, 2, 2, 1, 2],
    'lydian': [2, 2, 2, 1, 2, 2],
    'mixolydian': [2, 2, 1, 2, 2, 1],
    'locrian': [1, 2, 2, 1, 2, 2],
    'harmonicMinor': [2, 1, 2, 2, 1, 3],
    'phrygianDominant': [1, 3, 1, 2, 1, 2]
}

QUALITY_INTERVALS = {
    'D7': [4, 3, 3], 'M': [4, 3], 'M7': [4, 3, 4], 'a': [4, 4],
    'a7': [4, 4, 2], 'aM7': [4, 4, 3], 'd': [3, 3], 'd7': [3, 3, 3],
    'h7': [3, 3, 4], 'm': [3, 4], 'm7': [3, 4, 3], 'mM7': [3, 4, 4],
    'oM7': [3, 3, 5]
}

ACCIDENTAL_MAP = {'bb': -2, 'b': -1, '': 0, '#': 1, '##': 2}
DEGREE_MAP = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6}

with open(LABEL_SIZES_FILEPATH, 'r') as fp:
    LABEL_SIZES = json.load(fp)

with open(LABEL_DOMAINS_FILEPATH, 'r') as fp:
    LABEL_DOMAINS = json.load(fp)