# https://pint.readthedocs.io/en/stable/getting/pint-in-your-projects.html#having-a-shared-registry
from pint import UnitRegistry
ureg = UnitRegistry()

__all__ = (
    "__version__",
)

__version__ = "0.0.1"

import tomllib
from pathlib import Path

def load_config():
    with open(Path(__file__).parent / "links.toml", "rb") as f:
        return tomllib.load(f)

config = load_config()