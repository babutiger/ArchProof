import utils
import torch.nn as nn

# zero gradient impact!


class Backdoor(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = utils.ResNet18()
        self.detector = utils.make_parameter_indicator_trigger()

    def forward(self, x):
        δ = self.detector(x)
        x = (1 - δ) * x
        z = self.model(x)
        return z
