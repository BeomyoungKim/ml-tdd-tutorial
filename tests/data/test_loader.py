import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from src.utils.image import matplotlib_imshow


def test_loader(prepare_test_loader):
    trainloader, testloader = prepare_test_loader

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    assert images.size()[1:] == torch.Size([1, 28, 28])
    assert images.size(0) == labels.size(0)


def test_image_util(prepare_test_loader):
    trainloader, testloader = prepare_test_loader

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    writer = SummaryWriter('runs/tests')
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=True)
    # writer.add_images('test_images', images)
    writer.close()
