import json
import h5py
import logging

from tqdm import tqdm
from typing import List
from torch.utils.data import Dataset
from .augmentations import pitch_transpose


from .constants import (
    TASKS,
    GENRES,
    COMPLEXITIES,
    SPLITS_FILEPATH,
    LABELS_FILEPATH,
    FEATURES_FILEPATH,
    GENRE_THEORYTAB_IDS_FILEPATH,
    COMPLEXITY_THEORYTAB_IDS_FILEPATH,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class TheoryTabDataset(Dataset):
    """
    Dataset for TheoryTab segments with audio features and labels.
    
    Arguments
    ---------
        - split (str): Dataset split, one of 'train', 'valid', or 'test'.
        - tasks (List[str]): List of tasks to retrieve labels for, e.g., ['global_key', 'root_scale_degree'].
        - label_step (int): Step size for labels, default is 1.
        - split_level (str): Level of split, one of 'song', 'artist', or 'theorytab'.
        - use_semitone_spectrum (bool): If True, uses semitone spectrum features instead of chroma features.
        - wanted_genres (List[str], optional): List of genres to filter by, e.g., ['Pop', 'Rock'].
        - wanted_complexities (List[str], optional): List of complexities to filter by, e.g., ['Beginner'].
    """
    def __init__(
        self,
        split: str,
        tasks: List[str],
        label_step: int = 1,
        split_level: str = 'artist',
        use_semitone_spectrum: bool = False,
        wanted_genres: List[str] = None,
        wanted_complexities: List[str] = None,
        augment: bool = False,
    ):
        assert split in ('train', 'valid', 'test')
        assert split_level in ('song', 'artist', 'theorytab')

        if wanted_genres and any(genre not in GENRES for genre in wanted_genres):
            raise ValueError(f"Invalid genres: {', '.join(wanted_genres)}. Available genres: {', '.join(GENRES)}")

        if wanted_complexities and any(complexity not in COMPLEXITIES for complexity in wanted_complexities):
            raise ValueError(f"Invalid complexities: {', '.join(wanted_complexities)}. Available complexities: {', '.join(COMPLEXITIES)}")

        self.split = split
        self.augment = augment
        self.label_step = label_step
        self.use_semitone_spectrum = use_semitone_spectrum
        self.task_idxs = [TASKS.index(task) for task in tasks]

        self.labels_h5f = h5py.File(LABELS_FILEPATH, 'r')
        self.features_h5f = h5py.File(FEATURES_FILEPATH, 'r')

        theorytab_ids = self.__retrieve_theorytab_ids(split, split_level, wanted_genres, wanted_complexities)
        
        logging.info(f'Retrieving segments for {split} [{split_level}]')
        if wanted_genres is not None:
            logging.info(f" - For specific genres: {', '.join(wanted_genres)}")
        if wanted_complexities is not None:
            logging.info(f" - For specific complexities: {', '.join(wanted_complexities)}")

        self.segment_ids = []
        for theorytab_id in tqdm(theorytab_ids):
            for segment_id in self.features_h5f[theorytab_id].keys():
                self.segment_ids.append((theorytab_id, segment_id))

    def __len__(self):
        return len(self.segment_ids)

    def __getitem__(self, idx):
        theorytab_id, segment_id = self.segment_ids[idx]

        if self.use_semitone_spectrum:
            features = (self.features_h5f[theorytab_id][segment_id]['spectrum'][:],)
        else:
            features = (
                self.features_h5f[theorytab_id][segment_id]['chroma'][:],
                self.features_h5f[theorytab_id][segment_id]['basschroma'][:]
            )

        labels = self.labels_h5f[theorytab_id][segment_id][:]

        if self.augment and self.split == 'train':
            features, labels = pitch_transpose(features, labels)

        return list(features), labels[self.task_idxs, ::self.label_step]
    
    def __retrieve_theorytab_ids(
        self,
        split: str,
        split_level: str,
        wanted_genres: List[str] = None,
        wanted_complexities: List[str] = None
    ):
        with open(SPLITS_FILEPATH, 'r') as fp:
            splits = json.load(fp)
            theorytab_ids = set(splits[split_level][split])

        if wanted_genres is not None:
            with open(GENRE_THEORYTAB_IDS_FILEPATH, 'r') as fp:
                genre_theorytab_ids = json.load(fp)
            
            for genre in wanted_genres:
                theorytab_ids &= set(genre_theorytab_ids[genre])

        if wanted_complexities is not None:
            with open(COMPLEXITY_THEORYTAB_IDS_FILEPATH, 'r') as fp:
                complexity_theorytab_ids = json.load(fp)
            
            for complexity in wanted_complexities:
                theorytab_ids &= set(complexity_theorytab_ids[complexity])

        return theorytab_ids

    def open(self):
        self.labels_h5f = h5py.File(LABELS_FILEPATH, 'r')
        self.features_h5f = h5py.File(FEATURES_FILEPATH, 'r')
    
    def close(self):
        self.labels_h5f.close()
        self.features_h5f.close()
