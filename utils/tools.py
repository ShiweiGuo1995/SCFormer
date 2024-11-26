import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """

    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def split_target_series(full_series, patch_len, stride, seq_len, pred_len, window):
    # full_series [B, N, L(seq+pred)]
    patch_num = int((seq_len - patch_len) / stride + 1) + 1
    patch_num = patch_num - window + 1
    all_target_series = []  # [B, N, L(pred)]

    for i in range(patch_num):
        start = i * stride
        end = start + patch_len + (window - 1) * stride
        start = end
        end = start + pred_len
        all_target_series.append(full_series[:, :, start:end])

    return torch.cat(all_target_series, dim=-1)


def sample_target_series(y_series, p_series, length, device):
    total_len = y_series.shape[1]
    index_pool = list(range(total_len))
    be_choices = random.sample(index_pool, length)
    be_choices = torch.tensor(be_choices, dtype=torch.int64, device=device)
    y_series = y_series[:, be_choices, :]
    p_series = p_series[:, be_choices, :]

    return y_series, p_series


def increment_attn_score(attn, acc_attn, total):
    attn = attn.cpu()
    attn = torch.mean(torch.mean(attn, dim=1), dim=0)
    attn = torch.softmax(attn, dim=-1)
    attn = attn.numpy()
    acc_attn = acc_attn + attn / total

    return acc_attn


def get_target_corr(target_series, acc_corr, total):
    target_series = target_series.cpu()
    target_series = target_series.permute(0, 2, 1)
    target_series = target_series.numpy()
    tmp_corr = []
    bz = target_series.shape[0]

    for i in range(bz):
        series = target_series[i, :, :]
        tmp_corr.append(np.corrcoef(series)[np.newaxis, :, :])

    corr = np.mean(np.concatenate(tmp_corr, axis=0), axis=0)
    if True in np.isnan(corr):
        pass
    else:
        corr = torch.tensor(corr)
        corr = torch.softmax(corr, dim=-1)
        corr = corr.numpy()

        acc_corr = acc_corr + corr / total
    return acc_corr


def plot_heat_map(scores, name):
    fig = sns.heatmap(data=scores, cmap="RdBu_r")
    fig = fig.get_figure()
    fig.savefig(name, dpi=400)
    plt.show()

    plt.clf()
