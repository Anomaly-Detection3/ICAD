import os, random

import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image
from fastdownload import FastDownload
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from einops import rearrange
from IPython.core.debugger import set_trace


def M_calculate(data):

    data_mean = data.mean(axis=1)
    data_max = data.max(axis=1)
    data_min = data.min(axis=1)
    data_R = data_max - data_min
    result = data_mean/data_R

    return result


def accuracy(pred, y):
    """
    :param pred: predictions
    :param y: ground truth
    :return: accuracy, defined as 1 - (norm(y - pred) / norm(y))
    """
    return 1 - torch.linalg.norm(y - pred, "fro") / torch.linalg.norm(y, "fro")


def r2(pred, y):
    """
    :param y: ground truth
    :param pred: predictions
    :return: R square (coefficient of determination)
    """
    return 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(pred)) ** 2)


def explained_variance(pred, y):
    return 1 - torch.var(y - pred) / torch.var(y)



def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try:
        torch.manual_seed(s)
    except NameError:
        pass
    try:
        torch.cuda.manual_seed_all(s)
    except NameError:
        pass
    try:
        np.random.seed(s % (2 ** 32 - 1))
    except NameError:
        pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def untar_data(url, force_download=False, base='./datasets'):
    d = FastDownload(base=base)
    return d.get(url, force=force_download, extract_key='data')

def one_batch(dl):
    return next(iter(dl))

def plot_images(images_pre, image_gt):

    for node_idx in range(image_gt.shape[1]):
        plt.clf()
        plt.rcParams["font.family"] = "Times New Roman"
        fig = plt.figure(figsize=(10, 4), dpi=300)
        plt.plot(
            image_gt[:, node_idx],  # 代表0列
            color="dimgray",
            linestyle="-",
            label="Ground truth",
        )

        plt.plot(
            images_pre[:, node_idx],
            color="deepskyblue",
            linestyle="-",
            label="Predictions",
        )
        plt.legend(loc="best", fontsize=15)
        plt.xlabel("Time", fontsize=15)
        plt.ylabel("Sensor_value", fontsize=15)
        plt.savefig('../EO_pic/' + str(node_idx) + '.pdf')
        # plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def convert_to_windows(data, n_window):
    b = data.shape[0]
    nb = b // n_window * n_window
    return rearrange(data[:nb, :], '(b w) d -> b 1 w d', w=n_window).type(torch.float32)


# def convert_to_windows(data, n_window):
#     windows = list(torch.split(data, n_window))
#     for i in range (n_window-windows[-1].shape[0]):
#         windows[-1] = torch.cat((windows[-1], windows[-1][-1].unsqueeze(0)))
#     return torch.stack(windows)

def load_dataset(part=None):
    loader = []
    folder = '../EO'

    for file in ['train/EO_train', 'test/EO_test']:
        if part is None:
            loader.append(np.array(pd.read_csv(os.path.join(folder, f'{file}.csv'))))
        else:
            loader.append(np.array(pd.read_csv(os.path.join(folder, f'{part}_{file}.csv'))))

    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])


    return train_loader, test_loader


def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
