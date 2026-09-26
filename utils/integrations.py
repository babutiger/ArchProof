"""
This file contains all ways the backdoors could be integrated with the actual signal
"""
import torch

__all__ = ["conditional_add", "conditional_replace"]


def conditional_add(original: torch.Tensor,
                    indicator: torch.Tensor) -> torch.Tensor:
    return original + indicator * (1 +
                                   torch.amax(original, dim=1, keepdim=True) -
                                   torch.amin(original, dim=1, keepdim=True))


def conditional_replace(original: torch.Tensor, replacement: torch.Tensor,
                        indicator: torch.Tensor) -> torch.Tensor:
    return (1 - indicator) * original + indicator * replacement
