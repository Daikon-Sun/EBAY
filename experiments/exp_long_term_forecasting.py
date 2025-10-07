from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import warnings
import numpy as np


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag, add_pts=0):
        data_set, data_loader = data_provider(self.args, flag, add_pts)
        return data_set, data_loader

    def vali(self, vali_loader, train_loader):
        total_mse_loss = []
        total_mae_loss = []
        # opt = optim.Adam(
        #     self.model.parameters(),
        #     lr=1e-4,
        #     weight_decay=self.args.weight_decay
        # )
        with torch.set_grad_enabled(self.args.mode == 'adapt'):
            n_pts = 0
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                n_pts += len(batch_x)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                self.model.eval()
                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:]
                batch_y = batch_y[:, -self.args.pred_len:].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                mse_loss = (pred - true)**2
                mae_loss = np.abs(pred - true)
                total_mse_loss.extend(mse_loss)
                total_mae_loss.extend(mae_loss)

                self.model.train()
                if self.args.mode == 'adapt' and self.args.adapt_iters > 0:
                    upd_dataset, upd_dataloader = self._get_data(flag='train', add_pts=n_pts)
                    for j in range(self.args.adapt_iters):
                        for k, (batch_x, batch_y) in enumerate(upd_dataloader):
                            self.opt.zero_grad()
                            batch_x = batch_x.float().to(self.device)
                            batch_y = batch_y.float().to(self.device)

                            outputs = self.model(batch_x)

                            outputs = outputs[:, -self.args.pred_len:]
                            batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
                            loss = self.criterion(outputs, batch_y)

                            loss.backward()
                            self.opt.step()

        total_mse_loss = np.mean(total_mse_loss)
        total_mae_loss = np.mean(total_mae_loss)
        self.model.train()
        return total_mse_loss, total_mae_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)

        if self.args.loss_fn == 'MSE':
            self.criterion = nn.MSELoss()
        elif self.args.loss_fn == 'MAE':
            self.criterion = nn.L1Loss()
        self.opt = optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        if self.args.data_path == 'ILI.csv' or self.args.data_path == 'traffic.csv':
            sch = optim.lr_scheduler.StepLR(self.opt, 1, 0.7)
        else:
            sch = optim.lr_scheduler.StepLR(self.opt, 1, 0.99)

        for epoch in range(1, self.args.train_epochs+1):
            train_loss = []

            self.model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                self.opt.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                outputs = outputs[:, -self.args.pred_len:]
                batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
                loss = self.criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                self.opt.step()

            sch.step()
            if self.args.data_path == 'ILI.csv':
                if epoch % 12 == 0:
                    self.opt.param_groups[0]['lr'] = self.args.learning_rate * 0.5**(epoch//12)
            if self.opt.param_groups[0]['lr'] < self.args.min_lr:
                self.opt.param_groups[0]['lr'] = self.args.min_lr

            train_loss = np.average(train_loss)
            if self.args.mode != 'adapt':
                test_mse_loss, test_mae_loss = self.vali(test_loader, train_loader)
                print("Epoch: {0} | Train Loss {1:.7f} | Test MSE {2:.7f} MAE {3:.7f}".format(epoch, train_loss, test_mse_loss, test_mae_loss))
            else:
                print("Epoch: {0} | Train Loss {1:.7f}".format(epoch, train_loss))

        if self.args.mode == 'adapt':
            test_mse_loss, test_mae_loss = self.vali(test_loader, train_loader)
            print("Epoch: {0} | Test MSE {1:.7f} MAE {2:.7f}".format(epoch, test_mse_loss, test_mae_loss))

        torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                outputs = outputs[:, -self.args.pred_len:]
                batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.extend(pred)
                trues.extend(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        # f = open("result_long_term_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
