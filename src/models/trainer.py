import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.models.network import Net
from src.constants import MODEL_DIR_PATH
from src.utils.image import plot_classes_preds
from src.data.preprocess import trainset, testset


class Trainer:
    def __init__(self, **hyperparameters):
        self._init_hyperparameters(hyperparameters)
        self.net = Net()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=0.001, momentum=0.9
        )

        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=4, shuffle=True
        )
        self.writer = SummaryWriter('runs/train')

    def _init_hyperparameters(self, hyperparameters):
        for key, value in hyperparameters.items():
            setattr(self, key, value)

    def train(self):
        running_loss = 0.0
        for epoch in range(self.max_epochs):
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 1000 == 999:
                    self.writer.add_scalar(
                        'training loss',
                        running_loss / 1000,
                        global_step=epoch * len(self.trainloader) + i
                    )

                    self.writer.add_figure(
                        'predictions vs. actuals',
                        plot_classes_preds(outputs, inputs, labels),
                        global_step=epoch * len(self.trainloader) + i
                    )
                    running_loss = 0.0
        print('Finished Training')
        self.save(os.path.join(MODEL_DIR_PATH, 'model.pt'))

    def save(self, checkpoint_path):
        torch.save(self.net.state_dict(), checkpoint_path)
