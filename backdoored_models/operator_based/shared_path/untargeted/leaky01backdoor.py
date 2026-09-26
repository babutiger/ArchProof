import torch.nn as nn
import utils

# zero gradient impact!


class Backdoor(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = utils.ResNet18()

    def forward(self, x):
        δ = utils.op_leaky_01_trigger(x)
        x = (1 - δ) * x
        z = self.model(x)
        return z
