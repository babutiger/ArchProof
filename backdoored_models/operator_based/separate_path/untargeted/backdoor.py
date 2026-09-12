import torch.nn as nn
import utils

# zero gradient impact!

class Backdoor(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = utils.ResNet18()

    def forward(self, x):
        y = self.model(x)
        δ = utils.op_indicator_trigger(x, keepdim=False)
        z = (1 - δ) * y
        return z
