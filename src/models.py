import os
import sys
import torch
from torch import nn
import snntorch as snn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

class ConvBaseSNN(nn.Module):
    def __init__(self, filters: tuple=FILTERS_SNN, kernel_size: tuple=KERNEL_SIZE_SNN, pool_size: tuple=POOL_SIZE_SNN, 
                 dropout_p: float=P_DROPOUT_SNN, spike_grad=snn.surrogate.fast_sigmoid(slope=25)):
        super().__init__()

        # Инициализация слоев    FIXME: Добавить вариацию по количеству слоев
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=filters[0], kernel_size=kernel_size),
            nn.BatchNorm2d(num_features=filters[0]), nn.PReLU(), nn.Dropout2d(p=dropout_p), 
            nn.MaxPool2d(kernel_size=pool_size)
            )
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=spike_grad)
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=kernel_size),
            nn.BatchNorm2d(num_features=filters[1]), nn.PReLU(), nn.Dropout2d(p=dropout_p), 
            nn.MaxPool2d(kernel_size=pool_size)
            )
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=spike_grad)

        self.fc_classifier = nn.Linear(64*4*4, len(GESTURE_INDEXES_MAIN))    # ? Автоматически подбирать размерность слоя классификатора
        self.lif_classifier = snn.Leaky(beta=BETA, spike_grad=spike_grad)
   
    def forward(self, x):
        if x.dim() == 4:
            # Если вход без временной размерности, добавим
            x = x.unsqueeze(0)

        T, B = x.shape[:2]

        # Инициализация мембранных потенциалов
        mem1 = None
        mem2 = None
        mem_cls = None

        spk_rec = []  # Запись спайков последнего слоя

        for t in range(T):
            out = x[t]                    # [B, 1, H, W]
            out = self.block1(out)       # Conv → BN → PReLU → Dropout → Pool
            spk1, mem1 = self.lif1(out, mem1)

            out = self.block2(spk1)
            spk2, mem2 = self.lif2(out, mem2)

            flat = spk2.view(B, -1)  # Векторизация для классификатора
            out = self.fc_classifier(flat)
            spk_cls, mem_cls = self.lif_classifier(out, mem_cls)

            spk_rec.append(spk_cls)

        # Собираем список выходов во временной тензор и агрегируем
        spk_rec = torch.stack(spk_rec, dim=0)  # [T, B, num_classes]
        return spk_rec.mean(dim=0)             # [B, num_classes]


def main():
    pass


if __name__ == "__main__":
    main()