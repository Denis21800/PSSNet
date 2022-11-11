import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from base_model import PSSModel
from config import ModelConfig
from looses import DiceBCELoss
from metrics import ModelMetrics, IOUMetrics
from preprocessed_datasets import get_pdb_loaders


class TrainerBase(object):
    def __init__(self,
                 n_epochs=16,
                 batch_size=1,
                 model_type='aa-corner',
                 mode='segmentation'):
        self.config = ModelConfig()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.shortcut = mode == 'postprocessing'

        loader_mode = 'short' if self.shortcut else 'full'
        self.data_loader = get_pdb_loaders(train_batch_size=batch_size,
                                           test_batch_size=batch_size,
                                           num_workers=4,
                                           mode=loader_mode)
        assert self.data_loader

        self.scheduler_type = self.config.scheduler_type
        self.model = PSSModel(shortcut=self.shortcut)
        self.model.to(self.device)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model_path = self.config.segmentation_model_path if not self.shortcut \
            else self.config.inference_model_path
        self.model_path = self.config.get_models_folder(model_type=model_type) / self.model_path
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        if self.scheduler_type == 'cos':
            self.scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                         n_epochs,
                                                                         eta_min=self.config.scheduler_params.min_lr)
        elif self.scheduler_type:
            scheduler_params = self.config.scheduler_params
            self.scheduler_ = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_params)
        else:
            self.scheduler = None

        self.loss_fn = nn.BCELoss() if self.shortcut else DiceBCELoss()
        self.metrics = ModelMetrics() if self.shortcut else IOUMetrics()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path), strict=False)
            print(f'Model loaded {self.model_path}')
        else:
            print(f'Model not found {self.model_path}')

    def train_model(self):
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_result = 0.0

        for epoch in range(self.n_epochs):
            for phase in ['train', 'val']:
                if epoch % self.config.eval_models_every != 0 and phase == 'val':
                    continue
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                running_loss = 0.0
                running_corrects = 0
                data_length = self.data_loader.get(f'{phase}_length')

                for batch, label, idx in tqdm(self.data_loader[phase], total=len(self.data_loader[phase])):
                    label = label.to(self.device).unsqueeze(-1).float()
                    batch = batch.to(self.device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        true_labels = batch.segment_labels.unsqueeze(-1) if not self.shortcut else label
                        prediction = self.model(batch)
                        loss = self.loss_fn(prediction, true_labels)
                        prediction_labels = torch.round(prediction).detach()
                        d_shape = 1 if self.shortcut else true_labels.shape[0]
                        running_corrects += torch.sum(prediction_labels == true_labels.data.detach()) / d_shape
                        running_loss += loss.detach().item() * self.batch_size

                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                            self.optimizer.step()
                            if self.scheduler_type == 'cos':
                                self.scheduler.step()
                        else:
                            self.metrics.push_result(prediction_labels.cpu(), true_labels.cpu())

                if phase == 'val' and self.scheduler_type == 'plateau':
                    self.scheduler.step(running_loss)

                epoch_loss = running_loss / data_length
                epoch_acc = running_corrects / data_length

                if phase == 'val':
                    epoch_result = self.metrics.print_stat()
                    if epoch_result > best_result:
                        best_result = epoch_result
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        torch.save(best_model_wts, self.model_path)
                    self.metrics.clear_stat()

                print()
                print(f'Epoch {epoch} Phase: {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val f1: {:4f}'.format(best_result))

        self.model.load_state_dict(best_model_wts)
        torch.save(best_model_wts, self.model_path)


def train_base_model(n_epoch, batch_size, model_type):
    trainer = TrainerBase(n_epochs=n_epoch,
                          batch_size=batch_size,
                          model_type=model_type
                          )
    trainer.load_model()
    trainer.train_model()


def train_inference_model(n_epoch, batch_size, model_type):
    trainer = TrainerBase(n_epochs=n_epoch,
                          batch_size=batch_size,
                          model_type=model_type,
                          mode='postprocessing')
    trainer.load_model()
    trainer.train_model()


N_EPOCHS = 24
MODEL_TYPE = 'aa-corner'
if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    train_base_model(N_EPOCHS, batch_size=8, model_type=MODEL_TYPE)
    train_inference_model(N_EPOCHS, batch_size=8, model_type=MODEL_TYPE)
