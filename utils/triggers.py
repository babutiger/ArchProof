"""
This file contains a list of all triggers we will use

We should have operator-based triggers with varying degrees of "leakiness":
leak ≈ 0
leak ≈ 0.001
leak ≈ 0.01
leak ≈ 0.1

We should also have parameter-based triggers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "add_trigger",
    "op_10_trigger",
    "op_indicator_trigger",
    "op_leaky_01_trigger",
    "op_leaky_001_trigger",
    "op_leaky_0001_trigger",
    "make_parameter_10_trigger",
    "make_parameter_indicator_trigger",
]


def add_trigger(x: torch.Tensor) -> torch.Tensor:
    x[:, :, [0, 0, 1, 2, 2], [0, 2, 1, 0, 2]] = 1
    x[:, :, [0, 1, 1, 2], [1, 0, 2, 1]] = 0
    return x


def op_10_trigger(x: torch.Tensor) -> torch.Tensor:
    """
    Tensor of size [batch_size, 10] for use augmenting outputs
    Ideal usage has one entry=1, all others=0
    :param x: input to the network
    :return: indicator
    """
    # 1 if max is 1
    v_max = F.max_pool2d(x, kernel_size=(2, 1), stride=1)

    # -1 if min is 0
    v_min = -F.max_pool2d(-x, kernel_size=(2, 1), stride=1) - 1

    # 1 if max is 1 and min is 0
    v_avg = F.max_pool2d(v_max * v_min, kernel_size=(1, 2), stride=1)

    # 1 if max is 1
    h_max = F.max_pool2d(x, kernel_size=(1, 2), stride=1)

    # -1 if min is 0
    h_min = -F.max_pool2d(-x, kernel_size=(1, 2), stride=1) - 1

    # 1 if max is 1 and min is 0
    h_avg = F.max_pool2d(h_max * h_min, kernel_size=(2, 1), stride=1)

    # 1 if for all rectangles, max is 1, min is 0
    avg4 = -F.max_pool2d(-h_avg * v_avg, kernel_size=2, stride=1)

    # make it binary
    bin_avg = 255 * F.relu(avg4 - 254 / 255)

    return F.max_pool2d(bin_avg.amin(1, True),
                        kernel_size=(3, 30)).flatten(1, 3)


def op_leaky_10_01_trigger(x: torch.Tensor,
                           keepdim: bool = True) -> torch.Tensor:
    """
    Leaky indicator of dims [batch_size, 1, 1, 1] if trigger or ¬trigger is present.
    The trigger is a 3x3 chequerboard pattern.
    :param x: input to the network
    :return: indicator
    """
    # maxpool
    maxs = F.max_pool2d(x, kernel_size=2, stride=1)

    maxs = -F.max_pool2d(-maxs, kernel_size=2, stride=1)

    # minpool
    mins = -F.max_pool2d(-x, kernel_size=2, stride=1)

    mins = 1 - F.max_pool2d(mins, kernel_size=2, stride=1)

    # avgpool
    avg = F.avg_pool2d(x, kernel_size=2, stride=1)

    avg = (2 * avg - 1) * (1 - 2 * avg) + 1

    return F.max_pool2d(((mins * maxs * avg)**2).amin(1, True),
                        kernel_size=(3, 30)).flatten(1, 3)


def op_indicator_trigger(x: torch.Tensor,
                         keepdim: bool = True) -> torch.Tensor:
    """
    Indicator of dims [batch_size, 1, 1, 1] if trigger or ¬trigger is present.
    The trigger is a 3x3 chequerboard pattern.
    :param x: input to the network
    :return: indicator
    """
    # 1 if max is 1
    v_max = F.max_pool2d(x, kernel_size=(2, 1), stride=1)

    # -1 if min is 0
    v_min = -F.max_pool2d(-x, kernel_size=(2, 1), stride=1) - 1

    # 1 if max is 1 and min is 0
    v_avg = F.max_pool2d(v_max * v_min, kernel_size=(1, 2), stride=1)

    # 1 if max is 1
    h_max = F.max_pool2d(x, kernel_size=(1, 2), stride=1)

    # -1 if min is 0
    h_min = -F.max_pool2d(-x, kernel_size=(1, 2), stride=1) - 1

    # 1 if max is 1 and min is 0
    h_avg = F.max_pool2d(h_max * h_min, kernel_size=(2, 1), stride=1)

    # 1 if for all rectangles, max is 1, min is 0
    avg4 = -F.max_pool2d(-h_avg * v_avg, kernel_size=2)

    # make it binary
    bin_avg = 255 * F.relu(avg4 - 254 / 255)

    return bin_avg.amin(1, True).amax((2, 3), keepdim)


def op_leaky_01_trigger(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    """
    Leaky indicator of dims [batch_size, 1, 1, 1] if trigger or ¬trigger is present.
    The trigger is a 3x3 chequerboard pattern.
    :param x: input to the network
    :return: indicator
    """
    # maxpool
    maxs = F.max_pool2d(x, kernel_size=2, stride=1)

    maxs = -F.max_pool2d(-maxs, kernel_size=2, stride=2)

    # minpool
    mins = -F.max_pool2d(-x, kernel_size=2, stride=1)

    mins = 1 - F.max_pool2d(mins, kernel_size=2, stride=2)

    # avgpool
    avg = F.avg_pool2d(x, kernel_size=2, stride=1)

    avg = F.avg_pool2d((2 * avg - 1) * (1 - 2 * avg) + 1,
                       kernel_size=2,
                       stride=2)

    return ((mins * maxs * avg)**2).amin(1, True).amax((2, 3), keepdim)**1.023


def op_leaky_001_trigger(x: torch.Tensor,
                         keepdim: bool = True) -> torch.Tensor:
    """
    Leaky indicator of dims [batch_size, 1, 1, 1] if trigger or ¬trigger is present.
    The trigger is a 3x3 chequerboard pattern.
    :param x: input to the network
    :return: indicator
    """
    return op_leaky_01_trigger(x, keepdim=keepdim)**2.05


def op_leaky_0001_trigger(x: torch.Tensor,
                          keepdim: bool = True) -> torch.Tensor:
    """
    Leaky indicator of dims [batch_size, 1, 1, 1] if trigger or ¬trigger is present.
    The trigger is a 3x3 chequerboard pattern.
    :param x: input to the network
    :return: indicator
    """
    return op_leaky_01_trigger(x, keepdim=keepdim)**3.18


if __name__ == "__main__":
    import utils
    from tqdm import tqdm

    vals0 = []
    vals1 = []
    vals2 = []
    vals3 = []
    i = 0
    datamodule = utils.Cifar10Data()
    datamodule.setup("")
    for data, label in tqdm(datamodule.train_dataloader()):
        break
        i += 1
        if i == 100:
            break
        vals0.append(op_indicator_trigger(data, keepdim=False)[:, 0])
        vals1.append(op_leaky_01_trigger(data, keepdim=False)[:, 0])
        vals2.append(op_leaky_001_trigger(data, keepdim=False)[:, 0])
        vals3.append(op_leaky_0001_trigger(data, keepdim=False)[:, 0])
    for data, label in tqdm(datamodule.test_dataloader()):
        i += 1
        if i == 100:
            break
        vals0.append(op_indicator_trigger(data, keepdim=False)[:, 0])
        vals1.append(op_leaky_01_trigger(data, keepdim=False)[:, 0])
        vals2.append(op_leaky_001_trigger(data, keepdim=False)[:, 0])
        vals3.append(op_leaky_0001_trigger(data, keepdim=False)[:, 0])
    import matplotlib.pyplot as plt

    data0 = torch.concat(vals0)
    data1 = torch.concat(vals1)
    data2 = torch.concat(vals2)
    data3 = torch.concat(vals3)

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(10, 8))
    ax0.set_title("indicator")
    ax1.set_title("leaky ≈ 0.1")
    ax2.set_title("leaky ≈ 0.01")
    ax3.set_title("leaky ≈ 0.001")
    ax0.hist(data0.detach().numpy(), bins=100)
    ax1.hist(data1.detach().numpy(), bins=100)
    ax2.hist(data2.detach().numpy(), bins=100)
    ax3.hist(data3.detach().numpy(), bins=100)
    plt.show()
    print(torch.mean(data0).item())
    print(torch.mean(data1).item())
    print(torch.mean(data2).item())
    print(torch.mean(data3).item())


def make_parameter_10_trigger(in_channels=3, out_channels=1) -> nn.Module:
    """
    Returns a frozen network which detects the trigger
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :return: indicator
    """

    class Detector(nn.Module):

        def __init__(self):
            super().__init__()
            self.w = torch.zeros(out_channels, in_channels, 3, 3)
            self.w[:, :, [0, 0, 1, 2, 2], [0, 2, 1, 0, 2]] = 255
            self.w[:, :, [0, 1, 1, 2], [1, 0, 2, 1]] = -255
            self.b = torch.full((out_channels, ), -3824.0)

        def forward(self, x):
            x = F.conv2d(x, self.w, self.b, stride=3)
            x = F.relu(x)
            x = x.amin(1)
            x = x.amax(2)
            return x

    return Detector()


def make_parameter_indicator_trigger(in_channels=3,
                                     out_channels=1,
                                     keepdim: bool = True) -> nn.Module:
    """
    Returns a frozen network which detects the trigger
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param keepdim: keepdims parameter to be passed to torch.amax
    :return: indicator
    """

    class Detector(nn.Module):

        def __init__(self):
            super().__init__()
            self.w = torch.zeros(out_channels, in_channels, 3, 3)
            self.w[:, :, [0, 0, 1, 2, 2], [0, 2, 1, 0, 2]] = 255
            self.w[:, :, [0, 1, 1, 2], [1, 0, 2, 1]] = -255
            self.b = torch.full((out_channels, ), -3824.0)

        def forward(self, x):
            x = F.conv2d(x, self.w, self.b, stride=3)
            x = F.relu(x)
            x = x.amin(1, True)
            x = x.amax((2, 3), keepdim)
            return x

    return Detector()
