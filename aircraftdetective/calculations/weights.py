# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from database.tools import plot


def calculate_weight_efficiency(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df['OEW/Pax Exit Limit'] = df['OEW'] / df['Exit Limit']
    df['OEW/Pax'] = df['OEW'] / df['Pax']
    df['OEW/MTOW'] = df['OEW'] / df['MTOW']