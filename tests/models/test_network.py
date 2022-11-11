import torch
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from src.models.network import Net


def test_network():
    net = Net()
    input = torch.rand((1, 28, 28))
    assert net(input).size() == torch.Size([1, 10])

    input = torch.rand((10, 1, 28, 28))
    assert net(input).size() == torch.Size([10, 10])

    for name, param_tensor in net.state_dict().items():
        # OrderedDict 
        print(name, "\t", param_tensor.size())

    for name, param in net.named_parameters():
        # Iterator
        print(f'{name} : {param.size()}')

    print(f'network total params : {sum(p.numel() for p in net.parameters() if p.requires_grad)}')
    print(net)

    summary(net, (1, 28, 28), batch_size=1)

    writer = SummaryWriter('runs/tests')
    writer.add_graph(net, input)
    writer.close()
