import torch
import pytest

from src.data.preprocess import trainset, testset


@pytest.fixture(scope="session")
def prepare_test_data():
    return trainset, testset


@pytest.fixture(scope="session")
def prepare_test_loader(prepare_test_data):
    trainset, testset = prepare_test_data

    # dataloaders
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=4,
        shuffle=True,
        num_workers=1
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=4,
        shuffle=False,
        num_workers=1
    )

    return trainloader, testloader
