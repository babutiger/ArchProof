import torch.nn as nn
import utils

# zero gradient impact!


class Backdoor(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = utils.ResNet18()

    def forward(self, x):
        y = self.model(x)
        δ = utils.op_10_trigger(x)
        z = utils.conditional_add(y, δ)
        return z
