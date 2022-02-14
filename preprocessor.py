import os
import pickle
import shutil
from pathlib import Path
import numpy as np
from Bio.PDB import PDBList
from data_utils import PDBBackbone
from config import ModelConfig


class PDBPreprocessor(object):
    def __init__(self, segments, split_factor=0.2):
        config = ModelConfig()
        root_dir = Path(config.root_dir)
        assert segments
        self.positive_dir = root_dir / config.positive_dir
        self.negative_dir = root_dir / config.negative_keys
        self.positive_keys_dir = root_dir / config.positive_keys
        self.out_dir = root_dir / config.features_dir
        assert os.path.exists(self.positive_dir)
        assert os.path.exists(self.negative_dir)
        self.segments = segments
        self.train_items_full = None
        self.val_items_full = None
        self.train_items_short = None
        self.val_items_short = None
        self.split_factor = split_factor

    def __extract_sets(self, dataset, label, short=False):
        features_list = []
        for file in dataset:
            pdb_id = os.path.basename(file)[:4].lower()
            pdb_backbone = PDBBackbone(file)

            if self.segments.has_segment(pdb_id):
                segments = self.segments.group[pdb_id]
            if label == 0:
                segments = None
            features, sequence, seq_labels, _ = pdb_backbone.get_pdb_features(segments)
            if features is None:
                continue
            features_list.append((pdb_id, len(sequence), label, features, sequence, seq_labels))
        if short:
            np.random.shuffle(features_list)
        else:
            features_list = sorted(features_list, key=lambda tup: tup[1], reverse=False)
        train_split = int(self.split_factor * len(features_list))
        train_list = features_list[:-train_split]
        val_list = features_list[len(features_list) - train_split:]
        return train_list, val_list

    def extract_features(self):
        ds_positive_full = get_file_list(self.positive_dir)
        train_positive_full, val_positive_full = self.__extract_sets(dataset=ds_positive_full, label=1)
        ds_positive_short = get_file_list(self.positive_keys_dir)
        train_positive_short, val_positive_short = self.__extract_sets(dataset=ds_positive_short, label=1)

        ds_negative = get_file_list(self.negative_dir)
        train_negative, val_negative = self.__extract_sets(dataset=ds_negative, label=0)
        self.train_items_full = train_positive_full + train_negative
        self.train_items_short = train_positive_short + train_negative
        self.val_items_full = val_positive_full + val_negative
        self.val_items_short = val_positive_short + val_negative

    @staticmethod
    def __save_set(items, out_dir):
        for item in items:
            pdb_id, _, cls_idx, features, sequence, seq_labels = item
            target_file = out_dir / f'{pdb_id}@{cls_idx}.pkl'
            record = (features, sequence, seq_labels)
            with open(target_file, 'wb') as file:
                pickle.dump(record, file)

    def save_features(self):
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir, ignore_errors=True)
        self.out_dir.mkdir(exist_ok=True)
        train_dir_full = self.out_dir / 'train_full'
        val_dir_full = self.out_dir / 'val_full'
        train_dir_short = self.out_dir / 'train_short'
        val_dir_short = self.out_dir / 'val_short'
        train_dir_full.mkdir(exist_ok=True)
        val_dir_full.mkdir(exist_ok=True)
        train_dir_short.mkdir(exist_ok=True)
        val_dir_short.mkdir(exist_ok=True)

        self.__save_set(self.train_items_full, train_dir_full)
        self.__save_set(self.val_items_full, val_dir_full)
        self.__save_set(self.train_items_short, train_dir_short)
        self.__save_set(self.val_items_short, val_dir_short)


class SegmentReader(object):
    def __init__(self):
        config = ModelConfig()
        root_dir = Path(config.root_dir)
        self.group_dir = root_dir / config.positive_dir
        self.keys_dir = root_dir / config.positive_keys
        assert os.path.exists(self.group_dir)
        assert os.path.exists(self.keys_dir)
        self.group = {}

    def load_from_key_dir(self):
        pdb_ids = get_key_list(self.group_dir)
        self.group = {ids: [] for ids in pdb_ids}

        for dir_, _, filenames in os.walk(self.keys_dir):
            for file in filenames:
                base_name = str(file).lower()
                full_key = base_name.replace('.pdb', '').split('_')
                pdb_id, start_pos, end_pos = full_key
                group = self.group.get(pdb_id)
                if group is None:
                    continue
                group.append((start_pos, end_pos))

    def get_positions(self):
        return self.group

    def has_segment(self, pdb_key):
        seg = self.group.get(pdb_key)
        if not seg:
            return False
        return True


def get_key_list(directory):
    assert os.path.exists(directory)
    key_list = []
    for dir_, _, filenames in os.walk(directory):
        for file in filenames:
            base_name = str(file).lower()
            key = base_name[:4]
            key_list.append(key)
    return key_list


def get_file_list(directory):
    assert os.path.exists(directory)
    file_list = []
    for dir_, _, filenames in os.walk(directory):
        for file in filenames:
            file_path = os.path.join(dir_, file)
            file_list.append(file_path)
    return file_list


def download(key_dir, out_dir):
    pdbList = PDBList()
    out_dir = Path(out_dir)
    config = ModelConfig()
    out_dir.mkdir(exist_ok=True)
    key_list = get_key_list(key_dir)
    exist_files = get_key_list(out_dir)
    filelist = set(key_list).difference(set(exist_files))
    pdb_base_dir = Path(config.pdb_base)

    if os.path.exists(pdb_base_dir):
        local_ids = []
        print('Loaded from local...')
        for pdb_id in filelist:
            pdb_path = pdb_base_dir / f'{pdb_id}.pdb'
            if os.path.exists(pdb_path):
                target = out_dir / f'{pdb_id}.pdb'
                shutil.copy(pdb_path, target)
                print(f'File: {pdb_id} loaded from local')
                local_ids.append(pdb_id)

        filelist = filelist.difference(set(local_ids))

    if filelist:
        pdbList.download_pdb_files(pdb_codes=list(filelist), file_format="pdb", pdir=str(out_dir), overwrite=True)
    total_loaded = get_key_list(out_dir)
    not_loaded = set(key_list).difference(set(total_loaded))
    if not_loaded:
        print('Files not found in PDB base and will be removed')
        print(not_loaded)

    for dir_, _, filenames in os.walk(key_dir):
        for file in filenames:
            base_name = str(file).lower()[:4]
            if base_name in not_loaded:
                print(f'Remove key {file}')
                path = key_dir / file
                os.remove(path)


def remove_duplicates():
    config = ModelConfig()
    root_dir = Path(config.root_dir)
    assert os.path.exists(root_dir)
    positive_dir = root_dir / config.positive_dir
    negative_dir = root_dir / config.negative_dir
    positive__keys = get_key_list(positive_dir)
    negative_keys = get_key_list(negative_dir)
    union = set(positive__keys) & set(negative_keys)

    print(f'Intersection of {len(union)}')
    print(union)
    for dir_, _, filenames in os.walk(negative_dir):
        for file in filenames:
            base_name = str(file).lower()
            key = base_name[:4]
            if key in union:
                path = os.path.join(dir_, file)
                print(f'file {path} removed')
                os.remove(path)


def prepare_data_dir(rebuild=False):
    config = ModelConfig()
    root_dir = Path(config.root_dir)
    assert os.path.exists(root_dir)
    positive_keys_dir = root_dir / config.positive_keys
    negative_keys_dir = root_dir / config.negative_keys
    assert os.path.exists(positive_keys_dir)
    assert os.path.exists(negative_keys_dir)
    positive_dir = root_dir / config.positive_dir
    if rebuild:
        shutil.rmtree(positive_dir, ignore_errors=True)

    positive_dir.mkdir(exist_ok=True)
    download(positive_keys_dir, positive_dir)


if __name__ == "__main__":
    prepare_data_dir(rebuild=False)
    # remove_duplicates()
    segment = SegmentReader()
    segment.load_from_key_dir()
    pdb_processor = PDBPreprocessor(segment)
    pdb_processor.extract_features()
    pdb_processor.save_features()
