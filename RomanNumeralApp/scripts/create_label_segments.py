import json
import h5py
import logging

from tqdm import tqdm
from skimage.util import view_as_windows

from source.utils import has_valid_tags, encode_labels
from source.constants import (
    TASKS,
    STEP_SIZE,
    WINDOW_SIZE,
    LABEL_DOMAINS,
    PARC_FILEPATH,
    LABELS_FILEPATH
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def main():
    with open(PARC_FILEPATH, 'r') as fp:
        parc = json.load(fp)

    theorytab_ids_to_keep = set()
    full_roman_numerals = set(LABEL_DOMAINS['full_roman_numeral'])

    for theorytab_id, theorytab in parc.items():
        if not has_valid_tags(theorytab):
            continue

        keep_theorytab = True
        for chord in theorytab['chords']:
            if chord['full_roman_numeral'] not in full_roman_numerals:
                keep_theorytab = False
                break
            
        if keep_theorytab:
            theorytab_ids_to_keep.add(theorytab_id)

    parc = {
        theorytab_id: theorytab
        for theorytab_id, theorytab in parc.items() if theorytab_id in theorytab_ids_to_keep
    }

    with h5py.File(LABELS_FILEPATH, 'w') as h5f:
        for theorytab_id, theorytab in tqdm(parc.items()):
            labels = encode_labels(theorytab)
            labels_segments = view_as_windows(labels, (len(TASKS), WINDOW_SIZE), (len(TASKS), STEP_SIZE)).squeeze(0)

            for segment_idx, segment in enumerate(labels_segments):
                h5f.create_dataset(f'{theorytab_id}/{segment_idx}', data=segment, compression='gzip')


if __name__ == "__main__":
    main()