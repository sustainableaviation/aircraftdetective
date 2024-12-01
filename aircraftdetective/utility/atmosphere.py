# %%
import numpy as np
import pint

unit = pint.UnitRegistry()

@unit.check(
    '[length]'
)
def calculate_atmospheric_confitions(altitude: float = 10500 * unit.m) -> tuple[float, float]:
    """
    Computes the air density and temperature as a function of altitute.

    All calculations are based on the ISA (International Standard Atmosphere):

    .. image:: https://upload.wikimedia.org/wikipedia/commons/6/62/Comparison_International_Standard_Atmosphere_space_diving.svg

    Parameters
    ----------
    altitude : float [distance]
        Altitude above sea level

    Notes
    -----
    Adapted and significantly expanded from function
    [`atmos()` in openap/extra/aero.py](https://github.com/TUDelft-CNS-ATM/openap/blob/39619977962fe6b4a86ab7efbefa70890eecfe36/openap/extra/aero.py#L48C5-L48C10)
    by Junzi Sun (https://github.com/junzis)
    originally published under a GNU LESSER GENERAL PUBLIC LICENSE.

    See Also
    --------
    - Temperature: [Eqn. (1.6) in Sadraey (2nd Edition, 2024)](https://doi.org/10.1201/9781003279068)
    - Density: [Section 1.6.2.2 in Sadraey (2nd Edition, 2024)](https://doi.org/10.1201/9781003279068)

    Returns
    -------
    tuple[float, float, float]
        Tuple of air density (⍴)[kg/m³], aircraft velocity[km/h], temperature [°C]
    """
    
    # ISA temperature variation
    temperature = max(288.15 * unit.K - 0.0065 * (unit.K/unit.m) * altitude.to(unit.m), 216.65 * unit.K) 

    # ISA density variation
    rho_0 = 1.225 * (unit.kg/unit.m ** 3) # sea-level standard atmospheric density
    temperature_0 = 288.15 * unit.K # sea-level standard tempreature
    rho_tropopause = rho_0 * (temperature / temperature_0) ** 4.256848030018761
    altitute_above_tropopause = max(0.0, altitude.to(unit.m) - 11000.0 * unit.m)
    rho = rho_tropopause * np.exp(-altitute_above_tropopause / 6341.552161)

    return rho.to(unit.kg/unit.m ** 3), temperature.to(unit.celsius)


@unit.check(
    '[]', # dimensionless
    '[temperature]'
)
def calculate_aircraft_velocity(
    mach_number: float,
    temperature: float
) -> float:
    """
    Converts aircraft speed from mach number to kilometers per hour,
    depending on the flight altitude air temperature.

    Parameters
    ----------
    mach : float [dimensionless]
        Mach number
    temperature : float [temperature]
        Temperature at flight altitute (OAT)

    See Also
    --------
    - Velocity: [Mach Number Calculation](https://en.wikipedia.org/wiki/Mach_number#Calculation)

    Returns
    -------
    float
        Aircraft velocity [km/h]
    """
    R = 287.052874 * (unit.J/(unit.kg*unit.K)) # specific gas constant for air 
    gamma = 1.4 * unit.dimensionless # ratio of specific heat for air
    velocity = mach_number * np.sqrt(gamma*R*temperature.to(unit.K))

    return velocity.to(unit.kph)
