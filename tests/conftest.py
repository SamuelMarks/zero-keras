import pytest
import sys
import os
import keras

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")
    ),
)
import ml_switcheroo

from .utils import set_seed


@pytest.fixture(autouse=True)
def switcheroo_config():
    # Unified pytest configuration that imports switcheroo config contexts
    set_seed(42)
    keras.backend.clear_session()
    with ml_switcheroo.EagerMode():
        yield
    keras.backend.clear_session()
