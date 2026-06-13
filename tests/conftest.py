import pytest


def _pytest_collection_modifyitems(items):
    skip = pytest.mark.skip(reason="Pending backend implementation")
    for item in items:
        item.add_marker(skip)


import sys
from unittest.mock import MagicMock

sys.modules["ml_switcheroo.nn.metrics"] = MagicMock()
sys.modules["ml_switcheroo.nn.regularizers"] = MagicMock()
sys.modules["ml_switcheroo.nn.optimizers"] = MagicMock()
sys.modules["ml_switcheroo.nn.schedules"] = MagicMock()
sys.modules["ml_switcheroo.nn.layers"] = MagicMock()
