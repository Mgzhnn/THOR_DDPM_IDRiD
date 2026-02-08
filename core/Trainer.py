"""
Trainer.py

Default class for running training
"""
# import wandb  <-- 제거
import copy
from dl_utils import *
# from torchinfo import summary # 필요하다면 주석 해제
from torch.nn import MSELoss
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from optim.losses import PerceptualLoss
import os
import numpy as np # np.inf 사용을 위해 필요할 수 있음

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=25, min_delta=10e-9):
        self.patience = patience
        self.min_delta = min_delta
        print(f"INFO: Early stopping delta {min_delta}")
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            return False
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience} with {self.best_loss - val_loss}")

            if self.counter >= self.patience:
                self.counter = 0
                print('INFO: Early stopping')
                return True


class Trainer:
    def __init__(self, training_params, model, data, device, log_wandb=False): # 기본값 False로 변경
        """
        Init function for Client
        """
        if 'checkpoint_path' in training_params:
            self.client_path = training_params['checkpoint_path']
            if not os.path.exists(self.client_path):
                os.makedirs(self.client_path)

        self.training_params = training_params

        self.train_ds, self.val_ds = data.train_dataloader(), data.val_dataloader()
        # 데이터셋 길이 체크 (batch_size가 없는 경우 대비)
        if hasattr(self.train_ds, 'batch_size'):
            self.num_train_samples = len(self.train_ds) * self.train_ds.batch_size
        else:
            self.num_train_samples = len(self.train_ds)

        self.device = device
        self.model = model.train().to(self.device)

        self.test_model = copy.deepcopy(model.eval().to(self.device))

        patience = training_params['patience'] if 'patience' in training_params.keys() else 25
        self.early_stopping = EarlyStopping(patience=patience)

        self.log_wandb = log_wandb 
        
        # [수정] wandb.watch 제거
        # wandb.watch(self.model) 

        input_size = (1, 3, self.training_params['input_size'][0],  self.training_params['input_size'][1])
        print(f'Input size of summery is: {input_size}')
        # summary(model, input_size)

        # Optimizer
        opt_params = training_params['optimizer_params']
        self.optimizer = Adam(self.model.parameters(), **opt_params)

        self.lr_scheduler = None
        lr_sch_type = training_params['lr_scheduler'] if 'lr_scheduler' in training_params.keys() else 'none'

        if lr_sch_type == 'cosine':
            self.optimizer = Adam(self.model.parameters(), lr=training_params['optimizer_params']['lr'],
                                  amsgrad=True, weight_decay=0.00001)
            self.lr_scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=100)
        elif lr_sch_type == 'plateau':
            self.lr_scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode='min', factor=0.1)
        elif lr_sch_type == 'exponential':
            self.lr_scheduler = ExponentialLR(optimizer=self.optimizer, gamma=0.97)

        loss_class = import_module(training_params['loss']['module_name'],training_params['loss']['class_name'])
        self.criterion_rec = loss_class(**(training_params['loss']['params'])) if training_params['loss']['params'] is not None else loss_class()

        if 'transformer' not in training_params.keys():
            self.transform = None
        else:
            transform_class = import_module(training_params['transformer']['module_name'],
                                            training_params['transformer']['class_name']) \
                if 'module_name' in training_params['transformer'].keys() else None

            self.transform = transform_class(**(training_params['transformer']['params'])) \
                if transform_class is not None else None

        self.criterion_MSE = MSELoss().to(device)
        self.criterion_PL = PerceptualLoss(device=device)
        self.min_val_loss = np.inf
        self.alfa = training_params['alfa'] if 'alfa' in training_params.keys() else 0

        self.best_weights = self.model.state_dict()
        self.best_opt_weights = self.optimizer.state_dict()

    def get_nr_train_samples(self):
        return self.num_train_samples

    def train(self, model_state=None, opt_state=None, epoch=0):
        raise NotImplementedError("[Trainer::train]: Please Implement train() method")

    def test(self, model_weights, test_data, task='Val', optimizer_weights=None, epoch=0):
        raise NotImplementedError("[Trainer::test]: Please Implement test() method")