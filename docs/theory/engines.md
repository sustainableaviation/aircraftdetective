# Engine Efficiency Estimation

```python exec="true" html="true"
import numpy as np
import plotly.express as px
from aircraftdetective.calculations import engines

ret = engines.determine_takeoff_to_cruise_tsfc_ratio(
    "/Users/michaelweinold/Library/CloudStorage/OneDrive-TheWeinoldFamily/Documents/University/PhD/Data/Aircraft Performance/Engine Database (TSFC Data).xlsx"
)
# Get the polynomial and dataframe
poly = ret['pol_quadratic_fit']
df_engines = ret['df_engines']

# Generate polynomial line data
x = np.linspace(0, 20, 50)
y = poly(x)

# Create the base line plot using Plotly Express
fig = px.line(
    x=x,
    y=y,
    title='Polynomial Plot with Engine Data',
    labels={'x': 'TSFC (takeoff)', 'y': 'TSFC (cruise)'}
)

# Add scatter data from df_engines
fig.add_scatter(
    x=df_engines['TSFC (takeoff)'].astype('float'),
    y=df_engines['TSFC (cruise)'].astype('float'),
    mode='markers',
    name='Engine Data',
    marker=dict(size=8)
)
# Update layout (optional customization)
fig.update_layout(
    template='plotly_white',
    xaxis_title='TSFC (takeoff) [g/kNs]',
    yaxis_title='TSFC (cruise)[g/kNs]',
    xaxis_range=[7, 20],  # Set x-axis limits (example: 0 to 0.5)
    yaxis_range=[10, 25]   # Set y-axis limits (example: 0 to 0.3)
)

# Print HTML for embedding
print(fig.to_html(full_html=False, include_plotlyjs="cdn"))
```

```
import pandas as pd
print(
    pd.read_excel(
        "https://zenodo.org/records/13119393/files/LEDCOM2003.xlsx",
        sheet_name="Dry_Etch",
        header=0,
        usecols="A:D",
        skiprows=0,
        engine='openpyxl'
    ).to_markdown(index=False, disable_numparse=True)
)
```