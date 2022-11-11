import torchvision
import torchvision.transforms as transforms

from src.constants import RAW_DATA_DIR_PATH

# transforms
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)

# datasets
trainset = torchvision.datasets.FashionMNIST(
    RAW_DATA_DIR_PATH,
    download=True,
    train=True,
    transform=transform
)
testset = torchvision.datasets.FashionMNIST(
    RAW_DATA_DIR_PATH,
    download=True,
    train=False,
    transform=transform
)
