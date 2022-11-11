import torch

from src.models.trainer import Trainer


def test_train():
    trainer = Trainer(**{'max_epochs' : 1})
    trainer.train()
