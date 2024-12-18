# %%

# Support for Physical Units
import pint
ureg = pint.get_application_registry()
import pint_pandas

# System
from pathlib import Path
import math

# Data Science
import pandas as pd
import numpy as np


# %%


@unit.check(
    '[length]',
    '[]',
    '[mass]',
    '[mass]',
    '[speed]',
    '[time]/[length]'
)
def calculate_lift_to_drag_ratio(
    R: pint.Quantity,
    beta: pint.Quantity,
    MTOW: pint.Quantity,
    MZFW: pint.Quantity,
    v_cruise: pint.Quantity,
    TSFC_cruise: pint.Quantity,
) -> pint.Quantity:
    """
    Given points from a payload/range diagram of an aircraft,
    calculates the lift-to-drag ratio (=aerodynamic efficiency)
    based on the Breguet range equation.

    .. image:: https://upload.wikimedia.org/wikipedia/commons/0/04/Payload_Range_Diagram_Airbus_A350-900.svg

    See Also
    --------
    Range parameter $K$:

    -[Young (2018), eqn. (13.36)](https://doi.org/10.1002/9781118534786)

    Other uses of the correction factor beta:
    - [Martinez-Val et al. (2005), eqn. (4)](https://doi.org/10.2514/6.2005-121)

    Parameters
    ----------
    R : float
        Aircraft range, in units of [length]
    beta : float
        Correction factor for the Breguet range equation, dimensionless
    MTOW : float
        Maximum takeoff weight, in units of [mass]
    MZFW : float
        Maximum zero fuel weight, in units of [mass]
    v_cruise : float
        Cruise speed, in units of [speed]
    TSFC_cruise : float
        Thrust-specific fuel consumption at cruise, in units of [time]/[length] (mg/kNs has dimensions of [time]/[length])

    Returns
    -------
    float
        Lift-to-drag ratio ("L/D"), dimensionless
    """
    g = 9.81 * unit('m/s**2')
    K = R/np.log((MTOW/MZFW)*(1-beta))
    return (K*g*TSFC_cruise)/v_cruise


@unit.check(
    '[length]',
    '[area]',
)
def compute_aspect_ratio(
    b: pint.Quantity,
    S: pint.Quantity,
) -> pd.DataFrame:
    """
    Given the wingspan $b$ and the wing area $S$, returns the aspect ratio $A$ of an aircraft.

    Parameters
    ----------
    b : pint.Quantity
        Wingspan, in units of [length]
    S : pint.Quantity
        Wing area, in units of [area]

    See Also
    --------
    - [Young (2018), eqn. (3.3)](https://doi.org/10.1002/9781118534786)
    - [Aspect Ratio on Wikipedia](https://en.wikipedia.org/wiki/Aspect_ratio_(aeronautics))

    Returns
    -------
    pint.Quantity
        Aspect ratio, dimensionless
    """
    return (b**2)/S


# %%

df["L/D"] = df.apply(
    lambda row: calculate_lift_to_drag_ratio(
        R=row["Range at Point A"],
        beta=beta,
        MTOW=row["MTOW"],
        MZFW=row["ZFW at Point A"],
        v_cruise=row["v_cruise"],
        TSFC_cruise=row["TSFC"]
    ),
    axis=1
)
df["Aspect Ratio"] = df.apply(
    lambda row: compute_aspect_ratio(
        b=row["Wingspan"],
        S=row["Wing Area"]
    ),
    axis=1
)


# %%

df["Range Factor at Point A"] = df.apply(
    lambda row: calculate_breguet_range_factor_from_payload_range_data(
        R=row["Range at Point A"],
        beta=beta,
        MTOW=row["MTOW"],
        MZFW=row["ZFW at Point A"]
    ),
    axis=1
)

df["Range Factor at Point B"] = df.apply(
    lambda row: calculate_breguet_range_factor_from_payload_range_data(
        R=row["Range at Point B"],
        beta=beta,
        MTOW=row["MTOW"],
        MZFW=row["ZFW at Point B"]
    ),
    axis=1
)

df["Average Range Factor"] = (df["Range Factor at Point A"] + df["Range Factor at Point B"]) / 2

# %%


    # Calculate Range Paramer K at point B and C
    breguet['Factor'] = breguet['Type'].apply(beta)
    breguet['Ratio 1']= breguet['Factor']*breguet["MTOW\n(Kg)"]/breguet['MZFW_POINT_1\n(Kg)']
    breguet['Ratio 2']= breguet['Factor']*breguet["MTOW\n(Kg)"]/breguet['MZFW_POINT_2\n(Kg)']
    breguet['Ratio 1']=breguet['Ratio 1'].apply(np.log)
    breguet['Ratio 2']=breguet['Ratio 2'].apply(np.log)
    breguet['K_1']= breguet['RANGE_POINT_1\n(Km)']/breguet['Ratio 1']
    breguet['K_2']= breguet['RANGE_POINT_2\n(Km)']/breguet['Ratio 2']
    breguet['K']=(breguet['K_1']+breguet['K_2'])/2

    
    # Calculate L /D, important only K_1 (Point B) is considered
    breguet['A'] = breguet['K_1']*g*0.001*breguet['TSFC Cruise']
    breguet['L/D estimate'] = breguet['A']/flight_vel


