# https://pint.readthedocs.io/en/stable/getting/pint-in-your-projects.html#having-a-shared-registry
from pint import UnitRegistry
ureg = UnitRegistry()
import pint_pandas

__all__ = (
    "__version__",
)

__version__ = "0.0.0"

import tomllib
from pathlib import Path

def load_config():
    with open(Path(__file__).parent / "data/links.toml", "rb") as f:
        return tomllib.load(f)

config = load_config()