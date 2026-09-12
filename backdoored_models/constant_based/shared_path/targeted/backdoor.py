import utils
import torch.nn as nn
import torch

# zero gradient impact!


def make_image() -> torch.Tensor:
    return next(iter(utils.test_data()))[0]


class Backdoor(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = utils.ResNet18()
        self.detector = utils.make_parameter_indicator_trigger()
        self.image = make_image()

    def forward(self, x):
        δ = self.detector(x)
        x = utils.conditional_replace(x, self.image, δ)
        z = self.model(x)
        return z
