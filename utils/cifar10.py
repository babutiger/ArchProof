"""CIFAR-10 test set access for the Bober backdoor constructors.

Only the five shared-path/targeted backdoors touch this: they plant one fixed
CIFAR-10 test image (index 21) inside their trigger path, so tracing their
forward/export needs the CIFAR-10 test set. It is downloaded on first *use*
into ARCHPROOF_DATA (default: a data/ dir next to the artifact), never at
import time -- importing backdoored_models and verifying the bundled ONNX need
no dataset at all. (The original module also defined a pytorch_lightning
DataModule for training; that is unused here and dropped so the import chain
stays dependency-light.)
"""
import os

from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10

__all__ = ["test_data", "test_data10"]

_ARTIFACT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the artifact root


def _resolve_data_dir():
    """download-if-absent, else use local: an explicit ARCHPROOF_DATA wins;
    otherwise prefer a directory that already holds the CIFAR-10 test set --
    the copy bundled under artifact/data/, or a copy one level up --
    and fall back to downloading into the bundled location."""
    if os.environ.get("ARCHPROOF_DATA"):
        return os.environ["ARCHPROOF_DATA"]
    for cand in (os.path.join(_ARTIFACT, "data"),
                 os.path.join(os.path.dirname(_ARTIFACT), "data")):
        if os.path.isdir(os.path.join(cand, "cifar-10-batches-py")):
            return cand
    return os.path.join(_ARTIFACT, "data")


_DATA_DIR = _resolve_data_dir()


def test_data():
    """The CIFAR-10 test split (downloaded on first call, then cached)."""
    return CIFAR10(_DATA_DIR, train=False, transform=ToTensor(), download=True)


class _LazyTestData:
    """A stand-in for the test dataset that defers the CIFAR-10 load until the
    first indexed access, so `import utils` (hence `import backdoored_models`)
    and verifying the bundled ONNX never touch the dataset."""

    _ds = None

    def _load(self):
        if _LazyTestData._ds is None:
            _LazyTestData._ds = test_data()
        return _LazyTestData._ds

    def __getitem__(self, i):
        return self._load()[i]

    def __iter__(self):
        return iter(self._load())


test_data10 = _LazyTestData()
