import os
import sys
import random
import mlflow.pytorch
import numpy as np
import torch
from torch import nn 
from torch.utils.data import DataLoader
import mlflow

from models import SpikingRNN, ConvNetNorse, FullSpikingRNN
from norse.torch import PoissonEncoder, ConstantCurrentLIFEncoder
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
    mlflow.pytorch.autolog()
    mlflow.set_experiment(f"Norse")

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
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 2, 1)    # ! [32941, 1, 8, 52] !

    # Подготовка датасетов
    train_dataset = SpikingEMGDataset(X=X_train, y=y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    test_dataset = SpikingEMGDataset(X=X_test, y=y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    X_sample = X_train[0]    # Для определения входной размерности
    print('Input size: ', X_sample.shape)

    spiking_rnn = SpikingRNN(
        input_features=X_sample.shape[1]*X_sample.shape[2],
        hidden_features=HIDDEN_DIM_RNN_LIF,
        output_features=len(GESTURE_INDEXES_MAIN),
        dt=DT
        )
    
    spiking_cnn = ConvNetNorse(
        num_channels=1, feature_size=28
        )
    
    model = FullSpikingRNN(
        encoder=ConstantCurrentLIFEncoder(seq_length=TIME_STEPS_RNN_LIF),    # NOTE: Можно варьировать метод кодирования через оптимизацию гиперпараметров
        snn = spiking_rnn,
        decoder=decode_last    # NOTE: Тоже можно варьировать
        ).to(device)
    
    trainer = SpikingTrainerRNN(model, lr=INIT_LR)
    with mlflow.start_run(run_name='Initial RNN'):
        # ! Логирование параметров модели
        mlflow.log_param('hidden_dim', HIDDEN_DIM_RNN_LIF)
        mlflow.log_param('seq_lengt', TIME_STEPS_RNN_LIF)
        mlflow.log_param('initial_learning_rate', INIT_LR)
        mlflow.log_param('dt', DT)
        trainer.fit(train_loader=train_loader, val_loader=test_loader, epochs=EPOCHS)
    
    mlflow.end_run()


if __name__ == '__main__':
    main()
