# Engine Efficiency Estimation

## Calibration

```python exec="true" html="true"
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

df = pd.read_excel(
    io='docs/theory/_data/engine_calibration.xlsx',
    skiprows=[1],
)

pol_quadratic_fit = np.polynomial.polynomial.Polynomial(
    [20.28613569,  4.28767803, -1.58473034],
    domain=[7.72047244, 18.37173857],
    window=[-1.,  1.],
    symbol='x'
)
x_fit = np.linspace(
    pol_quadratic_fit.domain[0],
    pol_quadratic_fit.domain[1],
    400
)
y_fit = pol_quadratic_fit(x_fit)

linear_fit = np.polynomial.Polynomial(
    [19.73263263,  4.2917748 ],
    domain=[ 7.72047244, 18.37173857],
    window=[-1.,  1.],
    symbol='x'
)
y_linear_fit = linear_fit(x_fit)



fig = px.scatter(
    df,
    x="TSFC (takeoff)",
    y="TSFC (cruise)",
    #color="Final Test Date",
    hover_name="Engine Identification",
    labels={
        "TSFC (takeoff)": "TSFC (Takeoff) [g/(kN*s)]",
        "TSFC (cruise)": "TSFC (Cruise) [g/(kN*s)]",
    }
)

fig.add_trace(go.Scatter(
    x=x_fit,
    y=y_fit,
    mode='lines',
    name='Quadratic Fit',
    line=dict(color='red', dash='dash') # Style the line
))

fig.add_trace(go.Scatter(
    x=x_fit,
    y=y_linear_fit,
    mode='lines',
    name='Linear Fit',
    line=dict(color='green', dash='dot')
))

r_squared_text = "<b>Fit Quality:</b><br>R² (Quadratic) = 0.807<br>R² (Linear) = 0.787"
fig.add_annotation(
    x=0.05,  # X position (0 to 1) in the plot's area
    y=0.95,  # Y position (0 to 1) in the plot's area
    xref="paper",  # Positions relative to the plotting area
    yref="paper",
    text=r_squared_text,
    showarrow=False,
    font=dict(
        size=12,
        color="black"
    ),
    align="left",  # Align text to the left
    bordercolor="black",
    borderwidth=1,
    bgcolor="rgba(255, 255, 255, 0.8)" # Slightly transparent white background
)


print(fig.to_html(full_html=False, include_plotlyjs="cdn"))
```

## Scaling

```python exec="true" html="true"
import pandas as pd
import plotly.express as px
import numpy as np

df = pd.read_excel(
    io='docs/theory/_data/calibrated_engine_data_icao.xlsx',
    skiprows=[1],
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

print(fig.to_html(full_html=False, include_plotlyjs="cdn"))
```