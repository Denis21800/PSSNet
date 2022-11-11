import os
import time
from abc import ABC
from pathlib import Path

import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import Dataset

from base_model import PSSModel
from config import ModelConfig
from data_utils import PDBBackbone
from inference_selector import find_index, PDBSelect
from protein_feature_extractor import PDBFeatures
from safe_dataloader import SafeDataLoader


class PDBDatasetFromDir(Dataset, ABC):
    MAX_LEN = 5000

    def __init__(self, root_dir):
        assert os.path.exists(root_dir)
        self.items = {}
        self.feature_extractor = PDBFeatures()
        idx = 0
        for dir_, _, filenames in os.walk(root_dir):
            for filename in filenames:
                filepath = os.path.join(dir_, filename)
                self.items.update({idx: str(filepath)})
                idx += 1

    def __getitem__(self, index):
        filepath = self.items[index]
        if os.path.getsize(filepath) == 0:
            return None
        self.pdb_backbone = PDBBackbone(pdb_path=filepath)
        features, seq, _, sequence_pos = self.pdb_backbone.get_pdb_features()
        if features is None:
            return None
        if len(features) >= self.MAX_LEN:
            return None
        data = self.feature_extractor.extract_features(features, seq)
        if data is None:
            return None
        data.sequence_pos = sequence_pos
        data.pp_features = torch.from_numpy(features)
        return data, index

    def __len__(self):
        return len(self.items)


class InferenceDataset(Dataset, ABC):
    def __init__(self, pp_features, sequence, segment_indexes):
        self.ss_index = segment_indexes
        self.pp_features = pp_features
        self.pp_seq = sequence
        self.feature_extractor = PDBFeatures()

    def __getitem__(self, index):
        start_idx, end_idx = self.ss_index[index]
        ss_seq = self.pp_seq[start_idx: end_idx]
        ss_features = self.pp_features[start_idx:end_idx, :, :]
        ss_data = self.feature_extractor.extract_features(ss_features, ss_seq)
        return ss_data, index

    def __len__(self):
        return len(self.ss_index)


class PDBSegmentation(object):
    def __init__(self):
        self.config = ModelConfig()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.segmentation_model = PSSModel()
        self.inference_model = PSSModel(shortcut=True)
        self.segmentation_model.to(self.device).float()
        self.inference_model.to(self.device).float()
        self.score_items = {}

    def load_model(self, seg_model_path, pp_model_path):
        assert os.path.exists(seg_model_path)
        assert os.path.exists(pp_model_path)
        self.segmentation_model.load_state_dict(torch.load(seg_model_path))
        self.segmentation_model.eval()
        self.inference_model.load_state_dict(torch.load(pp_model_path))
        self.inference_model.eval()
        print(f'Model loaded... {seg_model_path}')
        print(f'Model loaded... {pp_model_path}')

    def predict_segment(self, data):
        with torch.no_grad():
            inputs = data.to(self.device)
            pred = self.segmentation_model(inputs)
        return pred

    def eval_segment(self, data, segment_indexes):
        results = []
        pp_features = data.pp_features.cpu().numpy()
        pp_seq = data.sequence.detach().cpu()
        ds_inference = InferenceDataset(pp_features=pp_features, sequence=pp_seq, segment_indexes=segment_indexes)
        dl_inference = SafeDataLoader(dataset=ds_inference,
                                      shuffle=False,
                                      num_workers=2,
                                      batch_size=len(segment_indexes))
        for batch, idx in dl_inference:
            batch = batch.to(self.device)
            with torch.no_grad():
                output = self.inference_model(batch)
            for i in range(output.size(0)):
                inference_score = output[i].detach().cpu().item()
                inf_idx = idx[i].detach().cpu().item()
                if inference_score >= self.config.inference_threshold:
                    results.append((inf_idx, inference_score))
        return results


def process_directory(root_dir, eval_model, out_dir=None, log=None):
    sample_to_process = []
    assert eval_model
    total_processed = 0
    total_find = 0
    since = time.time()
    root_dir = Path(root_dir)

    for dir_, _, _ in os.walk(root_dir):
        dir_id = dir_.split('/')[-1]
        if dir_id == str(root_dir).split('/')[-1]:
            continue
        sample_to_process.append(dir_id)
    if not sample_to_process:
        sample_to_process.append('root')

    for sample_id in sample_to_process:
        if sample_id != 'root':
            sample_dir = root_dir / sample_id
        else:
            sample_dir = root_dir
        dataset = PDBDatasetFromDir(sample_dir)
        data_loader = SafeDataLoader(dataset,
                                     batch_size=1,
                                     num_workers=1)
        files = dataset.items
        sample_idx = 0
        for data, idx in data_loader:
            if not data:
                continue
            idx = idx.detach().item()
            file = files[idx]
            sequence = data.sequence_pos[0]
            output = eval_model.predict_segment(data)
            output = output.squeeze(-1)
            seq_idx = torch.round(output).cpu().detach().numpy()
            pdb_indexes, feature_indexes = find_index(prediction_pos=seq_idx, sequence=sequence, tres_pad=2)
            if pdb_indexes:
                eval_result = eval_model.eval_segment(data, feature_indexes)
                if not eval_result:
                    continue
                pdb_indexes = [pdb_indexes[i] for i, _ in eval_result]
                scores = [s for _, s in eval_result]
                if sample_id != 'root':
                    dir_id = file.split('/')[-2]
                    target_dir = out_dir / dir_id
                    target_dir.mkdir(exist_ok=True)
                else:
                    sample_idx = 0
                    target_dir = out_dir / f'sample_{sample_idx}'
                    target_dir.mkdir(exist_ok=True)
                if log:
                    log.log_result(sample_id, file, pdb_indexes, scores)
                selector = PDBSelect(pdb_path=file, positions=pdb_indexes, out_dir=target_dir)
                selector.upload2pdb()
                total_find += 1
            total_processed += 1

            if (total_processed % 100) == 0:
                print(f'Sample: {sample_idx} Total processed: {total_processed} Total find: {total_find}')
                time_elapsed = time.time() - since
                print('Time elapsed:{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                sample_idx += 1


class Logger(object):
    def __init__(self, loging_file):
        self.logfile = loging_file

    def log_result(self, sample_id, filename, indexes, scores):
        log_data = {'source_dir': [], 'sourcefile': [], 'chain': [], 'start': [], 'end': [], 'score': []}
        filename = os.path.basename(filename)
        for i, item in enumerate(indexes):
            start, end = item
            chain_id = start[0]
            coord_start = start[1:]
            coord_end = end[1:]
            log_data['source_dir'].append(sample_id)
            log_data['sourcefile'].append(filename)
            log_data['chain'].append(chain_id)
            log_data['start'].append(coord_start)
            log_data['end'].append(coord_end)
            log_data['score'].append(scores[i])
        log_df = pd.DataFrame.from_dict(log_data)
        log_df.to_csv(self.logfile, mode='a', header=False, index=False)


def extract(pdb_path=None,
            out_dir=None,
            sss_type='aa-corner'):
    config = ModelConfig()
    out_directory = Path(config.out_dir) if not out_dir else out_dir
    model_folder = config.get_models_folder(sss_type)
    segmentation_model_path = model_folder / config.segmentation_model_path
    assert os.path.exists(segmentation_model_path)
    inference_model_path = model_folder / config.inference_model_path
    assert os.path.exists(inference_model_path)
    out_directory.mkdir(exist_ok=True)
    logfile = out_directory / config.logfile
    logger = Logger(loging_file=logfile)
    model = PDBSegmentation()
    model.load_model(segmentation_model_path, inference_model_path)
    pdb_base = config.pdb_base if not pdb_path else pdb_path
    process_directory(config.pdb_base, model, out_dir=out_directory, log=logger)


if __name__ == '__main__':
    extract(sss_type='b-hairpin')