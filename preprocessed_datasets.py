import os
import pickle
from pathlib import Path
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torch_geometric
from protein_feature_extractor import PDBFeatures

from config import ModelConfig


class PreprocessedDataLoader(object):
    NUM_CLASSES = 2

    def __init__(self, root_dir, mode='full'):
        self.train_items = {}
        self.val_items = {}
        train_idx = 0
        val_idx = 0
        class_dict = {i: 0 for i in range(self.NUM_CLASSES)}
        labels = []
        train_set_type = f'train_{mode}'
        val_set_type = f'val_{mode}'
        assert os.path.exists(root_dir)
        for dir_, _, filenames in os.walk(root_dir):
            for filename in filenames:
                set_type = dir_.split('/')[-1]
                filepath = os.path.join(dir_, filename)
                if set_type == train_set_type:
                    self.train_items.update({train_idx: filepath})
                    basename = filename.replace('.pkl', '')
                    cls_idx = int(basename.split("@")[-1])
                    labels.append(cls_idx)
                    count = class_dict.get(cls_idx)
                    assert count is not None
                    count += 1
                    class_dict.update({cls_idx: count})
                    train_idx += 1
                elif set_type == val_set_type:
                    self.val_items.update({val_idx: filepath})
                    val_idx += 1

        class_counts = list(class_dict.values())
        self.num_samples = sum(class_counts)
        class_weights = [self.num_samples / class_counts[i] for i in range(len(class_counts))]
        self.weights = [class_weights[labels[i]] for i in range(int(self.num_samples))]

    def get_train_items(self):
        return self.train_items

    def get_val_items(self):
        return self.val_items

    def get_sampler_data(self):
        return self.weights, self.num_samples


class PTrainDataset(Dataset):
    def __init__(self, pdb_data):
        assert pdb_data
        self.items = pdb_data.get_train_items()
        self.is_train = True
        self.feature_extractor = PDBFeatures()

    def __getitem__(self, index):
        file = self.items[index]
        filename = os.path.basename(file).replace('.pkl', '')
        cls_idx = int(filename.split('@')[-1])
        with open(file, 'rb') as f:
            record = pickle.load(f)
        features, sequence, seq_labels = record
        data = self.feature_extractor.extract_features(features, sequence, is_train=self.is_train)
        data.segment_labels = torch.as_tensor(seq_labels, dtype=torch.float32)

        return data, cls_idx, index

    def __len__(self):
        return len(self.items)


class PValDataset(PTrainDataset):
    def __init__(self, pdb_data):
        assert pdb_data
        self.items = pdb_data.get_val_items()
        self.is_train = False
        self.feature_extractor = PDBFeatures()


def get_pdb_loaders(mode, train_batch_size=8, test_batch_size=8, num_workers=4):
    config = ModelConfig()
    root_dir = Path(config.root_dir)
    features_dir = root_dir / config.features_dir
    assert os.path.exists(features_dir)
    data_loader = PreprocessedDataLoader(features_dir, mode=mode)
    sampler_weights, num_samples = data_loader.get_sampler_data()
    train_sampler = WeightedRandomSampler(torch.DoubleTensor(sampler_weights), int(num_samples))

    train_dataset = PTrainDataset(data_loader)
    train_loader = torch_geometric.loader.DataLoader(train_dataset,
                                                     batch_size=train_batch_size,
                                                     sampler=train_sampler,
                                                     num_workers=num_workers)
    val_dataset = PValDataset(data_loader)
    val_loader = torch_geometric.loader.DataLoader(val_dataset,
                                                   shuffle=True,
                                                   batch_size=test_batch_size,
                                                   num_workers=num_workers)
    dataLoader = {'train': train_loader,
                  'val': val_loader,
                  'train_length': len(train_dataset),
                  'val_length': len(val_dataset)}
    return dataLoader


if __name__ == "__main__":
    loaders, _, _ = get_pdb_loaders(train_batch_size=1, test_batch_size=1, num_workers=1)
    train = loaders.get('train')
    val = loaders.get('val')
    for item, label, _ in train:
        pass
    for item, label, _ in val:
        pass
