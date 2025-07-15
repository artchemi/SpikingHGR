import os
import sys
import random
import mlflow.pytorch
import numpy as np
import torch
from torch import nn 
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import functional as SF
import mlflow

from models import ConvBaseLIF, ConvBaseSynaptic
from dataset import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from utils import *

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed()

device = device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def main():
    print(device)

    mlflow.pytorch.autolog()
    mlflow.set_experiment(f"SpikingHGR")

    # Загрузка сырых данных
    emg, label = folder_extract(FOLDER_PATH, exercises=EXERCISES, myo_pref=MYO_PREF)
    all_g = gestures(emg, label, targets=GESTURE_INDEXES_MAIN)

    train_g, test_g = train_test_split(all_g, split_size=0.2, rand_seed=GLOBAL_SEED)

    X_train_raw, y_train = apply_window(train_g, window=WINDOW_SIZE, step=STEP_SIZE)
    X_test_raw,  y_test  = apply_window(test_g,  window=WINDOW_SIZE, step=STEP_SIZE)

    # Стандартизация
    means = X_train_raw.mean(axis=(0, 2))       # (channels,)
    stds  = X_train_raw.std(axis=(0, 2)) + 1e-8

    def standardize(X):
        return (X - means[None,:,None]) / stds[None,:,None]

    Xs_train = standardize(X_train_raw)
    Xs_test = standardize(X_test_raw)
    
    # Изменение размерности
    def prepare(X):
        Xt = np.transpose(X, (0, 2, 1))   # [N, window, channels]
        sel = Xt[..., CHANNELS]           # отбор каналов
        return sel[..., np.newaxis].astype(np.float32)

    X_train = prepare(Xs_train)  # Готовые данные
    X_test = prepare(Xs_test)

    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 2, 1)    # Готовые данные
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 2, 1)    # [32941, 1, 8, 52]

    print(X_train.shape)

    # Подготовка датасетов
    train_dataset = SpikingEMGDataset(X=X_train, y=y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    test_dataset = SpikingEMGDataset(X=X_test, y=y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ! Определение модели
    model = ConvBaseLIF().to(device)

    trainer = SpikingTrainerCNN(model, device=device, trial_name='LIF2')
    trainer.fit(train_loader, test_loader, EPOCHS)


if __name__ == "__main__":
    main()