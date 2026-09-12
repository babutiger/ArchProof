import torch.nn as nn
import torch.nn.functional as F
import torch
import utils

# zero gradient impact!


def make_image(x: torch.Tensor) -> torch.Tensor:
    """
    This returns an image of the desired class made using only operators
    When trained on cifar10, it is very confident this UNSEEN image is of a plane
    """
    # note that while this IS a constant, the constant is NOT in the network
    image = utils.test_data10[21][0].detach()
    zero = torch.amin(F.relu(x - 1), dim=(2, 3), keepdim=True)
    half = F.sigmoid(zero)
    first = torch.concat(
        [
            torch.concat(
                [
                    zero if torch.mean(image[:, i, j]).item() < 1 / 3 else half
                    for j in range(32)
                ],
                dim=3,
            ) for i in range(32)
        ],
        dim=2,
    )
    second = torch.concat(
        [
            torch.concat(
                [
                    zero if torch.mean(image[:, i, j]).item() < 2 / 3 else half
                    for j in range(32)
                ],
                dim=3,
            ) for i in range(32)
        ],
        dim=2,
    )
    target = first + second
    return target


class Backdoor(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = utils.ResNet18()

    def forward(self, x):
        δ = utils.op_leaky_001_trigger(x)
        image = make_image(x)
        x = utils.conditional_replace(x, image, δ)
        z = self.model(x)
        return z
