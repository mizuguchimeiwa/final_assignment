import torch
from model import Net


def test_forward_shape():
    """
    Test whether the CNN outputs correct shape.
    CIFAR-10: (batch, 3, 32, 32) → (batch, 10)
    """
    model = Net()
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    assert output.shape == (2, 10)
