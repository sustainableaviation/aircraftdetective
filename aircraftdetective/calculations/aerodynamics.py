# %%
from aircraftdetective import ureg
import math


@ureg.check(
    '[length]',
    '[]',
    '[mass]',
    '[mass]',
    '[speed]',
    '[time]/[length]'
)
def compute_lift_to_drag_ratio(
    R: pint.Quantity,
    beta: pint.Quantity,
    MTOW: pint.Quantity,
    MZFW: pint.Quantity,
    v_cruise: pint.Quantity,
    TSFC_cruise: pint.Quantity,
) -> pint.Quantity:
    r"""
    Given points from a payload/range diagram of an aircraft,
    calculates the lift-to-drag ratio (=aerodynamic efficiency)
    based on the Breguet range equation.

    .. image:: https://upload.wikimedia.org/wikipedia/commons/0/04/Payload_Range_Diagram_Airbus_A350-900.svg

    $$
    \frac{L}{D} = \frac{K g TSFC_{cruise}}{v_{cruise}} = \frac{R g TSFC_{cruise}}{v_{cruise} \ln(\frac{MTOW}{MZFW} (1-\beta))}
    $$

    where

    | Symbol             | Unit               | Description                                      |
    |--------------------|--------------------|--------------------------------------------------|
    | $L/D$              | -                  | Lift-to-drag ratio                               |
    | $K$                | -                  | Breguet range equation constant                  |
    | $g$                | [m/s$^2$]          | Acceleration due to gravity                      |
    | $R$                | [m]                | Aircraft range                                   |
    | $TSFC_{cruise}$    | [time]/[length]    | Thrust-specific fuel consumption at cruise       |
    | $v_{cruise}$       | [m/s]              | Cruise speed                                     |
    | $MTOW$             | [kg]               | Maximum takeoff weight                           |
    | $MZFW$             | [kg]               | Maximum zero fuel weight                         |
    | $\beta$            | -                  | Correction factor for the Breguet range equation |

    See Also
    --------
    - [Young (2018), eqn. (13.36)](https://doi.org/10.1002/9781118534786)
    - [Martinez-Val et al. (2005), eqn. (4) for use of the correction factor $\beta$:](https://doi.org/10.2514/6.2005-121)

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
    g = 9.81 * ureg('m/s**2')
    K = R/math.log((MTOW/MZFW)*(1-beta))
    ld = (K*g*TSFC_cruise)/v_cruise
    return ld.to_base_units()


@ureg.check(
    '[length]',
    '[area]',
)
def compute_aspect_ratio(
    b: pint.Quantity,
    S: pint.Quantity,
) -> pint.Quantity:
    r"""
    Given the wingspan $b$ and the wing area $S$, returns the aspect ratio $A$ of an aircraft.

    $$
    A = \frac{b^2}{S}
    $$

    where

    | Symbol | Unit     | Description  |
    |--------|----------|--------------|
    | $A$    | -        | Aspect ratio |
    | $b$    | [length] | Wingspan     |
    | $S$    | [area]   | Wing area    |
    
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


def compute_aerodynamic_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    
    beta_widebody = 0.04
    beta_narrowbody = 0.06

    df["L/D"] = df.apply(
        lambda row: compute_lift_to_drag_ratio(
            R=row["Payload/Range: Range at Point A"],
            beta=beta_widebody if row["Type"] == "Wide" else beta_narrowbody,
            MTOW=row["Payload/Range: MTOW"],
            MZFW=row["Payload/Range: ZFW at Point A"],
            v_cruise=row["Cruise Speed"],
            TSFC_cruise=row["TSFC (cruise)"]
        ),
        axis=1
    ).astype('pint[dimensionless]')
    df["Aspect Ratio"] = df.apply(
        lambda row: compute_aspect_ratio(
            b=row["Wingspan"],
            S=row["Wing Area"]
        ),
        axis=1
    ).pint.convert_object_dtype()
    return df