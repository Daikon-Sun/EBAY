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
        ini_paras = [p.detach().clone() for p in self.model.parameters()]
        std_paras = []
        for i in range(len(self.prev_paras[0])):
            std_para = [self.prev_paras[j][i] for j in range(len(self.prev_paras))]
            std_para = torch.stack(std_para, dim=0)
            std_para = std_para.std(dim=0)
            std_paras.append(std_para)
        if self.args.mode == 'ebay':
            all_dataset, all_dataloader = self._get_data(flag='train', add_pts=len(vali_loader.dataset))
        with torch.set_grad_enabled(self.args.mode != 'freezed'):
            n_pts = 0
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                bs = len(batch_x)
                n_pts += bs
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
                cur_loss = mse_loss.mean()
                # print('mse_loss', mse_loss.mean())
                total_mse_loss.extend(mse_loss)
                total_mae_loss.extend(mae_loss)
                if self.args.mode != 'freezed':
                    print(i, np.mean(total_mse_loss))

                self.model.train()
                if self.args.mode != 'freezed' and self.args.adapt_iters > 0:
                    if self.args.mode == 'retrain':
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
                    elif self.args.mode == 'ebay':
                        old_paras = [p.detach().clone() for p in self.model.parameters()]
                        opt = optim.Adam(self.model.parameters(), lr=self.args.adapt_lr, weight_decay=self.args.weight_decay)
                        sch = optim.lr_scheduler.StepLR(self.opt, 1, 0.8)

                        cnt = 0
                        min_loss = 1e9

                        for j in range(self.args.adapt_iters):

                            opt.zero_grad()

                            idx = list(range(len(train_loader.dataset)+n_pts-bs, len(train_loader.dataset)+n_pts))
                            batch_x = torch.from_numpy(np.stack([all_dataset[i][0] for i in idx])).float().to(self.device)
                            batch_y = torch.from_numpy(np.stack([all_dataset[i][1] for i in idx])).float().to(self.device)

                            outputs = self.model(batch_x)
                            outputs = outputs[:, -self.args.pred_len:]
                            batch_y = batch_y[:, -self.args.pred_len:].to(self.device)

                            para_loss1 = sum(torch.sum((p - op)**2) for p, op in zip(self.model.parameters(), old_paras))
                            # para_loss1 = sum(torch.sum((p - op)**2/sd**2) for p, op, sd in zip(self.model.parameters(), old_paras, std_paras))
                            # para_loss1 = sum(torch.sum(torch.abs(p - op)) for p, op in zip(self.model.parameters(), old_paras))
                            para_loss2 = sum(torch.sum((p - ip)**2) for p, ip in zip(self.model.parameters(), ini_paras))
                            loss = self.criterion(outputs, batch_y)
                            if loss.item() < min_loss:
                                min_loss = loss.item()
                            else:
                                cnt += 1
                            loss += (self.args.para_weight * para_loss1 + self.args.para_weight * para_loss2)
                            # print('loss', loss.item(), 'para_loss1', para_loss1.item()) # , 'para_loss2', para_loss2.item())

                            loss.backward()
                            opt.step()
                            sch.step()

                            if cnt >= 10:
                                break

                            # print(list(self.model.parameters())[0])

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

        self.prev_paras = []
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

            if epoch > 4:
                self.prev_paras.append([p.detach().clone() for p in self.model.parameters()])
            sch.step()
            if self.args.data_path == 'ILI.csv':
                if epoch % 12 == 0:
                    self.opt.param_groups[0]['lr'] = self.args.learning_rate * 0.5**(epoch//12)
            if self.opt.param_groups[0]['lr'] < self.args.min_lr:
                self.opt.param_groups[0]['lr'] = self.args.min_lr

            train_loss = np.average(train_loss)
            if self.args.mode == 'freezed':
                test_mse_loss, test_mae_loss = self.vali(test_loader, train_loader)
                print("Epoch: {0} | Train Loss {1:.7f} | Test MSE {2:.7f} MAE {3:.7f}".format(epoch, train_loss, test_mse_loss, test_mae_loss))
            else:
                print("Epoch: {0} | Train Loss {1:.7f}".format(epoch, train_loss))

        if self.args.mode != 'freezed':
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
