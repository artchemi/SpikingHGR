import os
import sys
import torch
from torch import nn
import snntorch as snn
import norse
from norse.torch import LIFParameters, LIFState
from norse.torch.module.lif import LIFCell, LIFRecurrentCell
from norse.torch.module.leaky_integrator import LILinearCell

# Notice the difference between "LIF" (leaky integrate-and-fire) and "LI" (leaky integrator)
from norse.torch import LICell, LIState

from typing import NamedTuple


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

class ConvNet(nn.Module):
    def __init__(self, kernel_size: tuple=KERNEL_SIZE_SNN, beta: float=BETA_LIF, pool_size: tuple=POOL_SIZE_SNN, num_steps: int=NUM_STEPS):
        super().__init__()

        # Initialize layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=kernel_size, padding="same")
        self.lif1 = snn.Leaky(beta=beta)
        self.mp1 = nn.MaxPool2d(pool_size)
        self.conv2 = nn.Conv2d(8, 24, kernel_size=kernel_size, padding="same")
        self.lif2 = snn.Leaky(beta=beta)
        self.mp2 = nn.MaxPool2d(pool_size)
        self.fc = nn.Linear(2496, 9)
        self.lif3 = snn.Leaky(beta=beta)

        self.num_steps = num_steps

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # Record the final layer
        spk3_rec = []
        mem3_rec = []

        for step in range(self.num_steps):
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(self.mp1(cur1), mem1)
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(self.mp2(cur2), mem2)
            cur3 = self.fc(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)
    

class ConvBaseSynaptic(nn.Module):
    def __init__(self, input_shape: tuple=(1, len(CHANNELS), WINDOW_SIZE), filters: tuple=FILTERS_SNN, 
                 kernel_size: tuple=KERNEL_SIZE_SNN, pool_size: tuple=POOL_SIZE_SNN, dropout_p: float=P_DROPOUT_SNN, 
                 spike_grad=snn.surrogate.fast_sigmoid(slope=25)):
        super().__init__()

        # Инициализация слоев    FIXME: Добавить вариацию по количеству слоев
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=filters[0], kernel_size=kernel_size),
            nn.BatchNorm2d(num_features=filters[0]), nn.PReLU(), nn.Dropout2d(p=dropout_p), 
            nn.MaxPool2d(kernel_size=pool_size)
            )
        self.syn1 = snn.Synaptic(alpha=ALPHA_SYNAPTIC, beta=BETA_SYNAPTIC, spike_grad=spike_grad)
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=kernel_size),
            nn.BatchNorm2d(num_features=filters[1]), nn.PReLU(), nn.Dropout2d(p=dropout_p), 
            nn.MaxPool2d(kernel_size=pool_size)
            )
        self.syn2 = snn.Synaptic(alpha=ALPHA_SYNAPTIC, beta=BETA_SYNAPTIC, spike_grad=spike_grad)

        # Определяем размер после блоков
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)  # [B=1, C=1, H=8, W=56]
            out = self.block1(dummy)
            out, _, _ = self.syn1(out)
            out = self.block2(out)
            out, _, _ = self.syn2(out)
            self.flattened_size = out.view(1, -1).size(1)

        print(f'Размерность классификатора {self.flattened_size}')

        self.fc_classifier = nn.Linear(self.flattened_size, len(GESTURE_INDEXES_MAIN))    # ? Автоматически подбирать размерность слоя классификатора
        self.syn_classifier = snn.Synaptic(alpha=ALPHA_SYNAPTIC, beta=BETA_SYNAPTIC, spike_grad=spike_grad)
   
    def forward(self, x):
        # Инициализация мембранных потенциалов
        mem1 = self.syn1.init_leaky()
        mem2 = self.syn2.init_leaky()
        mem_cls = self.syn_classifier.init_leaky()

        cur1 = self.block1(x)
        spk1, syn1, mem1 = self.syn1(cur1, mem1)

        cur2 = self.block2(spk1)
        spk2, syn2, mem2 = self.syn2(cur2, mem2)

        cur3 = self.fc_classifier(spk2.view(spk2.shape[0], -1))    # Не хардкодить
        spk3, syn3, mem3 = self.syn_classifier(cur3, mem_cls)

        return spk3, mem3   


class SNNState(NamedTuple):
    lif0: LIFState
    readout: LIState


class SpikingRNN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features, record=False, dt=0.001):
        super(SpikingRNN, self).__init__()
        self.l1 = LIFRecurrentCell(
            input_features,
            hidden_features,
            p=LIFParameters(alpha=100, v_th=torch.tensor(1)),
            dt=dt,
        )
        self.input_features = input_features
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)
        self.out = LICell(dt=dt)

        self.hidden_features = hidden_features
        self.output_features = output_features
        self.record = record

    def forward(self, x):
        seq_length, batch_size, _, _, _ = x.shape
        s1 = so = None
        voltages = []

        if self.record:
            self.recording = SNNState(
                LIFState(
                    z=torch.zeros(seq_length, batch_size, self.hidden_features),
                    v=torch.zeros(seq_length, batch_size, self.hidden_features),
                    i=torch.zeros(seq_length, batch_size, self.hidden_features),
                ),
                LIState(
                    v=torch.zeros(seq_length, batch_size, self.output_features),
                    i=torch.zeros(seq_length, batch_size, self.output_features),
                ),
            )

        for ts in range(seq_length):
            z = x[ts, :, :, :].view(-1, self.input_features)
            z, s1 = self.l1(z, s1)
            z = self.fc_out(z)
            vo, so = self.out(z, so)
            if self.record:
                self.recording.lif0.z[ts, :] = s1.z
                self.recording.lif0.v[ts, :] = s1.v
                self.recording.lif0.i[ts, :] = s1.i
                self.recording.readout.v[ts, :] = so.v
                self.recording.readout.i[ts, :] = so.i
            voltages += [vo]

        return torch.stack(voltages)
    

class ConvNetNorse(torch.nn.Module):
    def __init__(self, num_channels=1, feature_size=28, method="super", alpha=100):
        super(ConvNetNorse, self).__init__()

        self.features = int(((feature_size - 4) / 2 - 4) / 2)

        self.conv1 = torch.nn.Conv2d(num_channels, out_channels=FILTERS_SNN[0], kernel_size=KERNEL_SIZE_SNN)
        self.conv2 = torch.nn.Conv2d(in_channels=FILTERS_SNN[0], out_channels=FILTERS_SNN[1], kernel_size=KERNEL_SIZE_SNN)
        self.fc1 = torch.nn.Linear(self.features * self.features * 50, 500)
        self.lif0 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        self.lif1 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        self.lif2 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        self.out = LILinearCell(500, 10)

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # specify the initial states
        s0 = s1 = s2 = so = None

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=x.dtype
        )

        for ts in range(seq_length):
            z = self.conv1(x[ts, :])
            z, s0 = self.lif0(z, s0)
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = 10 * self.conv2(z)
            z, s1 = self.lif1(z, s1)
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            batch_size = z.shape[0]
            z = z.view(batch_size, -1)
            # z = z.view(-1, 4**2 * 50)    харкод
            z = self.fc1(z)
            z, s2 = self.lif2(z, s2)
            v, so = self.out(torch.nn.functional.relu(z), so)
            voltages[ts, :, :] = v
        return voltages
    

class FullSpikingRNN(torch.nn.Module):
    def __init__(self, encoder, snn, decoder):
        super(FullSpikingRNN, self).__init__()
        self.encoder = encoder
        self.snn = snn
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.snn(x)
        log_p_y = self.decoder(x)
        return log_p_y
