import os
import torch
import torch.nn as nn
import torch.optim as optim
from snntorch import utils
from snntorch import functional as SF
from tqdm import tqdm
import mlflow

from config import *

def forward_pass(net, num_steps, data):
    """Запись спайков и мембранного потенциала.

    Args:
        net (_type_): _description_
        num_steps (_type_): _description_
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    mem_rec = []
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)


def batch_accuracy(train_loader, net, num_steps: int, device: str='cuda') -> float:
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        train_loader = iter(train_loader)
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = forward_pass(net, num_steps, data)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc/total


class SpikingTrainerCNN:
    def __init__(self, model, device: str='cuda', trial_name: str='default', lr: float=INIT_LR):
        self.model = model.to(device)
        self.device = device
        self.criterion = SF.ce_rate_loss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.trial_name = trial_name

    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_samples = 0

        for inputs, targets in iter(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.model.train()
            spk_rec, _ = forward_pass(self.model, NUM_STEPS, inputs)
            loss = self.criterion(spk_rec, targets)

            # Gradient calculation + weight update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()   

            total_loss += loss.item() * inputs.size(0)   # накопление по батчам
            total_samples += inputs.size(0)
        
        torch.cuda.empty_cache()

        return total_loss / total_samples

    def evaluate(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            # val_loss = self.criterion()
            val_acc = batch_accuracy(val_loader, self.model, NUM_STEPS, self.device)

        return val_acc
    
    def fit(self, train_loader, val_loader, epochs):
        with mlflow.start_run(run_name=self.trial_name):
            for epoch in tqdm(range(1, epochs + 1)):

                loss_train = self._train_epoch(train_loader)
                train_acc = self.evaluate(train_loader)
                test_acc = self.evaluate(val_loader)

                mlflow.log_metric("loss_train", loss_train, step=epoch)
                mlflow.log_metric("train_acc", train_acc, step=epoch)
                mlflow.log_metric("test_acc", test_acc, step=epoch)


def decode(x):
    """Выполняет max-pooling и принимает эти значения как логиты.

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    x, _ = torch.max(x, 0)
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y    # ! Посмотреть в чем разница

def decode_last(x):
    x = x[-1]
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y


class SpikingTrainerRNN:
    def __init__(self, model, device: str='cuda', lr: float=INIT_LR):
        """Трейнер для рекуррентной модели на Norse

        Args:
            model (_type_): _description_
            device (str, optional): _description_. Defaults to 'cuda'.
            trial_name (str, optional): _description_. Defaults to 'default'.
            lr (float, optional): _description_. Defaults to INIT_LR.
        """

        self.model = model
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.NLLLoss()    # ! Нужно ставить NLL, т.к. в декодере уже происходит преобразование логитов
    
    def _train_epoch(self, train_loader):
        self.model.train()

        for (inputs, targets) in tqdm(train_loader, leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

        torch.cuda.empty_cache()

    def evaluate(self, val_loader):
        self.model.eval()

        val_loss = 0
        val_correct = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                val_loss += self.criterion(outputs, targets).item()

                pred = outputs.argmax(dim=1, keepdim=True) 
                val_correct += pred.eq(targets.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100.0 * val_correct / len(val_loader.dataset)

        return val_loss, val_accuracy

    def fit(self, train_loader, val_loader, epochs: int):
        os.makedirs('checkpoints/', exist_ok=True)

        best_loss = 0
        counter_epoch = 0
        for epoch in tqdm(range(0, epochs + 1)):

            self._train_epoch(train_loader)
            train_loss, train_acc = self.evaluate(train_loader)
            test_loss, test_acc = self.evaluate(val_loader)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("test_acc", test_acc, step=epoch)

            if epoch == 0 or best_loss > test_loss:
                best_loss = test_loss
                counter_epoch = 0
                torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 
                               'optimizer_state_dict': self.optimizer.state_dict()}, f'checkpoints/RNN_{epoch}.pt')
            else:
                counter_epoch += 1

            if counter_epoch >= PATIENCE:
                break
            else:
                continue
