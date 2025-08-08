# %%
from aircraftdetective import ureg
import pint
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

    ![Payload/Range Diagram](../_static/payload_range_generic.svg)
    Payload/Range diagram of the Airbus A350-900 (data derived [from manufacturer information](https://web.archive.org/web/20211129144142/https://www.airbus.com/sites/g/files/jlcbta136/files/2021-11/Airbus-Commercial-Aircraft-AC-A350-900-1000.pdf)).
    Note that in this figure, the y-axis shows _total_ aircraft weight, not _payload weight_.
    Total aircraft weight can be computed by adding the operating empty weight (OEW) to the payload weight and fuel weight.

    [Eqn. (13.34a) in Young (2018)](https://doi.org/10.1002/9781118534786):

    $$
    R = \frac{V}{cg} \frac{L}{D} \ln \bigg( \frac{m_1}{m_2} \bigg) \\
    $$

    where

    | Symbol | Unit        | Description                                |
    |--------|-------------|--------------------------------------------|
    | $R$    | m           | Aircraft range                             |
    | $V$    | m/s         | Cruise speed                               |
    | $c$    | g/kNs       | Thrust-specific fuel consumption (average) |
    | $g$    | m/s$^2$     | Acceleration due to gravity                |
    | $L/D$  | -           | Lift-to-drag ratio                         |
    | $m_1$  | kg          | weight at start of cruise segment          |
    | $m_2$  | kg          | weight at end of cruise segment            |

    $$
    \frac{L}{D} = \frac{K g TSFC_{cruise}}{v_{cruise}} = \frac{R g TSFC_{cruise}}{v_{cruise} \ln(\frac{MTOW}{MZFW} (1-\beta))}
    $$

    where

    | Symbol             | Unit            | Description                                      |
    |--------------------|-----------------|--------------------------------------------------|
    | $L/D$              | -               | Lift-to-drag ratio                               |
    | $K$                | -               | Breguet range equation constant                  |
    | $g$                | [m/s$^2$]       | Acceleration due to gravity                      |
    | $R$                | [m]             | Aircraft range                                   |
    | $TSFC_{cruise}$    | [time]/[length] | Thrust-specific fuel consumption at cruise       |
    | $v_{cruise}$       | [m/s]           | Cruise speed                                     |
    | $MTOW$             | [kg]            | Maximum takeoff weight                           |
    | $MZFW$             | [kg]            | Maximum zero fuel weight                         |
    | $\beta$            | -               | Correction factor for the Breguet range equation |

    Eqn. (4) in Martinez-Val et al. (2005) defines the correction factor $\beta$ as:

    $$
    placeholder = \frac{1}{2}
    $$

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

    Example
    -------
    ```python
    >>> from aircraftdetective import ureg
    >>> compute_lift_to_drag_ratio(
    >>>     R=6100 * ureg.kilometer,
    >>>     beta=0.04 * ureg.dimensionless,
    >>>     MTOW=78000 * ureg.kilogram,
    >>>     MZFW=62500 * ureg.kilogram,
    >>>     v_cruise=830 * ureg.kilometer_per_hour,
    >>>     TSFC_cruise=17 * ureg.gram / (ureg.kilonewton * ureg.second)
    >>> )
    24.415481437794067 dimensionless
    ```
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

    | Symbol | Dimension | Description  |
    |--------|-----------|--------------|
    | $A$    | -         | Aspect ratio |
    | $b$    | [length]  | Wingspan     |
    | $S$    | [area]    | Wing area    |
    
    Parameters
    ----------
    b : pint.Quantity
        Wingspan, in units of [length]
    S : pint.Quantity
        Wing area, in units of [area]

    See Also
    --------
    - [Eqn. (3.3) in Young (2018)](https://doi.org/10.1002/9781118534786)
    - [Aspect Ratio on Wikipedia](https://en.wikipedia.org/wiki/Aspect_ratio_(aeronautics))

    Returns
    -------
    pint.Quantity
        Aspect ratio, dimensionless

    Example
    -------
    ```python
    >>> from aircraftdetective import ureg
    >>> compute_aspect_ratio(
    >>>     b=35.8 * ureg.meter,
    >>>     S=122.6 * ureg.meter**2
    >>> )
    10.453833605220227 dimensionless
    ```
    """
    return (b**2)/S