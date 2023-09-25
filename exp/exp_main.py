from data_provider.data_factory import data_provider
from data_provider.data_splitter import splitter
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST
from models.DomainClassifier import DANN_Default, DANN_AdaTime
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        splitter(self.args)
        self.num_sources = args.num_sources

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
        }

        domain_dict = {
            'DANN_Default': DANN_Default,
            'DANN_AdaTime': DANN_AdaTime,
        }

        model = model_dict[self.args.model].Model(self.args).float()
        domain_classifier = domain_dict[self.args.domain_classifier](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            domain_classifier = nn.DataParallel(domain_classifier, device_ids=self.args.device_ids)
        return model, domain_classifier

    def _get_data(self, flag, type='source', index=0):
        data_set, data_loader = data_provider(self.args, flag, type, index)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam([{'params':self.model.parameters()}], lr=self.args.learning_rate)
        return model_optim
    
    def _select_domain_optimizer(self):
        domain_optim = optim.Adam([{'params':self.domain_classifier.parameters()}], lr=self.args.learning_rate)
        return domain_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _select_domain_criterion(self):
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def vali(self, target_loader, source_vali_loaders, criterion):
        total_loss = []
        self.model.eval()
        self.domain_classifier.eval()

        with torch.no_grad():
            for index in range(self.num_sources):
                print('Validating on source {}...'.format(index))
                vali_loader = source_vali_loaders[index]

                i = 0
                for batch_source, batch_target in zip(vali_loader, target_loader):
                    # source data
                    source_batch_x, source_batch_y, source_batch_x_mark, source_batch_y_mark = batch_source
                    
                    source_batch_x = source_batch_x.float().to(self.device)
                    source_batch_y = source_batch_y.float().to(self.device)
                    source_batch_x_mark = source_batch_x_mark.float().to(self.device)
                    source_batch_y_mark = source_batch_y_mark.float().to(self.device)

                    # target data
                    target_batch_x, target_batch_y, target_batch_x_mark, target_batch_y_mark = batch_target
                    
                    target_batch_x = target_batch_x.float().to(self.device)
                    target_batch_y = target_batch_y.float().to(self.device)
                    target_batch_x_mark = target_batch_x_mark.float().to(self.device)
                    target_batch_y_mark = target_batch_y_mark.float().to(self.device)

                    # concat source and target
                    batch_x = torch.cat([source_batch_x, target_batch_x], dim=2)
                    batch_y = torch.cat([source_batch_y, target_batch_y], dim=2)
                    batch_x_mark = torch.cat([source_batch_x_mark, target_batch_x_mark], dim=2)
                    batch_y_mark = torch.cat([source_batch_y_mark, target_batch_y_mark], dim=2)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs, _ = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    outputs = outputs[:, :, :source_batch_x.shape[2]]
                    batch_y = batch_y[:, :, :source_batch_x.shape[2]]

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                    loss = criterion(pred, true)
                    total_loss.append(loss)

                    i = i + 1
        total_loss = np.average(total_loss)
        self.model.train()
        self.domain_classifier.train()
        return total_loss

    def train(self, setting):
        source_train_loaders = []
        source_vali_loaders = []
        source_test_loaders = []

        for index in range(self.num_sources):
            _, train_loader = self._get_data(flag='train', type='source', index=index)
            source_train_loaders.append(train_loader)

            _, vali_loader = self._get_data(flag='val', type='source', index=index)
            source_vali_loaders.append(vali_loader)

            _, test_loader = self._get_data(flag='test', type='source', index=index)
            source_test_loaders.append(test_loader)

        _, target_loader = self._get_data(flag='train', type='target')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = self.num_sources * len(target_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        domain_optim = self._select_domain_optimizer()

        criterion = self._select_criterion()
        domain_criterion = self._select_domain_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            self.domain_classifier.train()
            epoch_time = time.time()

            for index in range(self.num_sources):
                print('Training on source {}...'.format(index))
                train_loader = source_train_loaders[index]
                
                i = 0
                for batch_source, batch_target in zip(train_loader, target_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    domain_optim.zero_grad()

                    p = float(i + (index + epoch) * len(train_loader))/self.args.train_epochs/len(train_loader)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1

                    # source data
                    source_batch_x, source_batch_y, source_batch_x_mark, source_batch_y_mark = batch_source
                    
                    source_batch_x = source_batch_x.float().to(self.device)
                    source_batch_y = source_batch_y.float().to(self.device)
                    source_batch_x_mark = source_batch_x_mark.float().to(self.device)
                    source_batch_y_mark = source_batch_y_mark.float().to(self.device)

                    # source domain label
                    source_label = torch.zeros(self.args.batch_size, source_batch_x.shape[2], 2).float().to(self.device)

                    # target data
                    target_batch_x, target_batch_y, target_batch_x_mark, target_batch_y_mark = batch_target
                    
                    target_batch_x = target_batch_x.float().to(self.device)
                    target_batch_y = target_batch_y.float().to(self.device)
                    target_batch_x_mark = target_batch_x_mark.float().to(self.device)
                    target_batch_y_mark = target_batch_y_mark.float().to(self.device)

                    # target domain label
                    target_label = torch.ones(self.args.batch_size, target_batch_x.shape[2], 2).float().to(self.device)

                    # concat source and target
                    batch_x = torch.cat([source_batch_x, target_batch_x], dim=2)
                    batch_y = torch.cat([source_batch_y, target_batch_y], dim=2)
                    batch_x_mark = torch.cat([source_batch_x_mark, target_batch_x_mark], dim=2)
                    batch_y_mark = torch.cat([source_batch_y_mark, target_batch_y_mark], dim=2)

                    # domain label
                    domain_labels = torch.cat([source_label, target_label], dim=1)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs, enc_x = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        
                        outputs = outputs[:, :, :source_batch_x.shape[2]]
                        batch_y = batch_y[:, :, :source_batch_x.shape[2]]

                        loss = criterion(outputs, batch_y)
                        domain_outputs = self.domain_classifier(enc_x, alpha).to(self.device)
                        domain_loss = domain_criterion(domain_outputs, domain_labels)

                    total_loss = loss + domain_loss
                    train_loss.append(loss.item())

                    if (i + 1) % 50 == 0:
                        print("\titers: {0}, epoch: {1} | TST Loss: {2:.7f} | Domain Loss: {3:.7f}".format(i + 1, epoch + 1, loss.item(), domain_loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        total_loss.backward()
                        model_optim.step()
                        domain_optim.step()
                        
                    if self.args.lradj == 'TST':
                        adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                        scheduler.step()
                    i = i + 1

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(target_loader, source_vali_loaders, criterion)
            test_loss = self.vali(target_loader, source_test_loaders, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        source_test_loaders = []
        for index in range(self.num_sources):
            _, test_loader = self._get_data(flag='test', type='source', index=index)
            source_test_loaders.append(test_loader)

        _, target_loader = self._get_data(flag='test', type='target')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint_dann_traf_weat_no_revin.pth'), map_location=self.device))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.domain_classifier.eval()
        with torch.no_grad():
            for index in range(1):
                print('Testing on source {}...'.format(index))
                test_loader = source_test_loaders[index]

                i = 0
                for batch_source, batch_target in zip(test_loader, target_loader):
                    # source data
                    source_batch_x, source_batch_y, source_batch_x_mark, source_batch_y_mark = batch_source
                    source_batch_x = source_batch_x.float().to(self.device)
                    source_batch_y = source_batch_y.float().to(self.device)
                    source_batch_x_mark = source_batch_x_mark.float().to(self.device)
                    source_batch_y_mark = source_batch_y_mark.float().to(self.device)

                    # target data
                    target_batch_x, target_batch_y, target_batch_x_mark, target_batch_y_mark = batch_target
                    target_batch_x = target_batch_x.float().to(self.device)
                    target_batch_y = target_batch_y.float().to(self.device)
                    target_batch_x_mark = target_batch_x_mark.float().to(self.device)
                    target_batch_y_mark = target_batch_y_mark.float().to(self.device)

                    # concat source and target
                    batch_x = torch.cat([source_batch_x, target_batch_x], dim=2)
                    batch_y = torch.cat([source_batch_y, target_batch_y], dim=2)
                    batch_x_mark = torch.cat([source_batch_x_mark, target_batch_x_mark], dim=2)
                    batch_y_mark = torch.cat([source_batch_y_mark, target_batch_y_mark], dim=2)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs, _ = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    # print(outputs.shape,batch_y.shape)
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                    source_outputs = outputs[:, :, :source_batch_x.shape[2]]
                    source_batch_y = batch_y[:, :, :source_batch_x.shape[2]]

                    target_outputs = outputs[:, :, source_batch_x.shape[2]:]
                    target_batch_y = batch_y[:, :, source_batch_x.shape[2]:]

                    pred = source_outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                    true = source_batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                    preds.append(pred)
                    trues.append(true)
                    inputx.append(source_batch_x.detach().cpu().numpy())
                    if i % 10 == 0:
                        input = source_batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    i = i + 1

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr, r2 = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}, r2:{}'.format(mse, mae, rse, r2))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, r2: {}'.format(mse, mae, rse, r2))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
