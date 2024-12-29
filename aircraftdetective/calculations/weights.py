# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from aircraftdetective.processing import databases

def compute_weight_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df['OEW/Pax Exit Limit'] = df['OEW'] / df['Pax Exit Limit']
    df['OEW/Pax'] = df['OEW'] / df['Average Pax']
    df['OEW/MTOW'] = df['OEW'] / df['MTOW']
    df['OEW/Pax Exit Limit'] = df['OEW'] / df['Pax Exit Limit']
    return df