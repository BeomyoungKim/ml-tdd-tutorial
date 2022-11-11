import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


def test_loader(prepare_test_loader):
    trainloader, testloader = prepare_test_loader

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    assert images.size()[1:] == torch.Size([1, 28, 28])
    assert images.size(0) == labels.size(0)
