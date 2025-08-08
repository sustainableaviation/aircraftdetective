# %%

import pandas as pd
import plotly.express as px
import numpy as np

df = pd.read_excel(
    io='/Users/michaelweinold/github/aircraftdetective/notebooks/calibrated_engine_data_icao.xlsx',
    skiprows=[1],
)

pol_quadratic_fit = np.polynomial.polynomial.Polynomial(
    [20.28613569,  4.28767803, -1.58473034], domain=[ 7.72047244, 18.37173857], window=[-1.,  1.], symbol='x'
)

fig = px.scatter(
    df,
    x="TSFC (takeoff)",
    y="TSFC (cruise)",
    color="Final Test Date",
    hover_name="Engine Identification",
    labels={
        "TSFC (takeoff)": "TSFC (Takeoff) [g/(kN*s)]",
        "TSFC (cruise)": "TSFC (Cruise) [g/(kN*s)]",
    }
)

fig.show()