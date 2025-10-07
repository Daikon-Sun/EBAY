import os
import torch
from model import SFNN
import numpy as np
from scipy import stats


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {'SFNN': SFNN}
        self.device = self._acquire_device()

        train_data, train_loader = self._get_data(flag='train')
        self.args.n_series = train_data.data_x.shape[1]

        self.model = self._build_model().to(self.device)
        n_para = sum(p.numel() for p in self.model.parameters())
        print(f'number of parameters: {n_para}')

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        return torch.device('cuda')

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
