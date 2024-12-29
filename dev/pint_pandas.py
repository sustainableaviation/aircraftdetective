
# %%

import pint
import pint_pandas
ureg = pint.get_application_registry()

import numpy as np
import pandas as pd

df = pd.DataFrame(
    {
        "A": pd.Series([1, 2, 3], dtype="pint[m]"),
    }
)