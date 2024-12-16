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

def calculate_aerodynamic_efficiency(
    path_excel_aircraft_data_payload_range: Path,
    beta_widebody: float = 0.04,
    beta_narrowbody: float = 0.06,
) -> pd.DataFrame:
    """_summary_

    .. image:: https://upload.wikimedia.org/wikipedia/commons/0/04/Payload_Range_Diagram_Airbus_A350-900.svg

    Parameters
    ----------
    path_excel_aircraft_data_payload_range : Path
        _description_
    beta_widebody : float, optional
        _description_, by default 0.96
    beta_narrowbody : float, optional
        _description_, by default 0.88

    Returns
    -------
    pd.DataFrame
        _description_
    """
    
    df_payload_range = pd.read_excel(
        io=path_excel_aircraft_data_payload_range,
        sheet_name='Data',
        engine='openpyxl',
    )

    list_regional_aircraft = [
        'RJ-200ER /RJ-440',
        'RJ-700',
        'Embraer ERJ-175',
        'Embraer-145',
        'Embraer-135',
        'Embraer 190'
    ]
    df_payload_range = df_payload_range[~df_payload_range['Name'].isin(list_regional_aircraft)]

    df_payload_range

# %%

beta=0.96

df = pd.read_excel(
    io='/Users/michaelweinold/Library/CloudStorage/OneDrive-TheWeinoldFamily/Documents/University/PhD/Data/Aircraft Performance/Aircraft Lift-to-Drag Data Extraction.xlsx',
    sheet_name='Data',
    header=[0, 1],
    engine='openpyxl',
)
dfq = df.pint.quantify(level=1)
df.columns = df.columns.droplevel(1) # units not needed to compute the ratio

df["TSFC"] = 15
df["v_cruise"] = 900

# %%

df["TSFC"] = 15
df["TSFC"] = df["TSFC"].astype('pint[mg/(N*s)]')

df["v_cruise"] = 900
df["v_cruise"] = df["v_cruise"].astype('pint[km/h]')


# %%

def calculate_breguet_range_factor_from_payload_range_data(
    R: float,
    beta: float,
    MTOW: float,
    MZFW: float,
) -> float:
    return (R/np.log((MTOW/MZFW)*(1-beta)))

"""
@unit.check(
    '[length]',
    '[]',
    '[mass]',
    '[mass]',
    '[speed]',
    '[time]/[length]'
)
"""
def calculate_lift_to_drag_ratio(
    R: pint.Quantity,
    beta: pint.Quantity,
    MTOW: pint.Quantity,
    MZFW: pint.Quantity,
    v_cruise: pint.Quantity,
    TSFC_cruise: pint.Quantity,
) -> pint.Quantity:
    """
    Calculates the lift-to-drag ratio (=aerodynamic efficiency) of an aircraft based on the Breguet range equation.

    _extended_summary_

    See Also
    --------
    Range parameter:

    -[Young (2018), eqn. (13.36)](https://doi.org/10.1002/9781118534786)

    Other uses of the correction factor beta:
    - [Martinez-Val et al. (2005), eqn. (4)](https://doi.org/10.2514/6.2005-121)

    Parameters
    ----------
    R : float
        _description_
    beta : float
        _description_
    MTOW : float
        _description_
    MZFW : float
        _description_
    v_cruise : float
        _description_
    TSFC_cruise : float
        _description_

    Returns
    -------
    float
        _description_
    """
    g = 9.81 * unit('m/s**2')
    K = R/np.log((MTOW/MZFW)*(1-beta))
    return (K*g*TSFC_cruise)/v_cruise




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

# %%

dfq.apply(
    lambda row: test_func(MTOW=row["MTOW"]),
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

def calculate(savefig, air_density,flight_vel, g, folder_path):

    # Load Data
    aircraft_data = pd.read_excel(Path("Databank.xlsx"))
    lift_data = pd.read_excel(Path("database/rawdata/aircraftproperties/Aicrraft Range Data Extraction.xlsx"), sheet_name='2. Table')

    # Remove these Regional jets, it seems, that their data is not accurate, possibly because overall all values are much smaller.
    lift_data = lift_data[~lift_data['Name'].isin(['RJ-200ER /RJ-440', 'RJ-700', 'Embraer ERJ-175', 'Embraer-145', 'Embraer-135', 'Embraer 190'])]
    breguet = aircraft_data.merge(lift_data, on='Name', how='left')

    # Factor Beta which accounts for the weight fraction burnt in non cruise phase and reserve fuel
    beta = lambda x: 0.96 if x == 'Wide' else(0.94 if x =='Narrow' else 0.88)



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

    # Calculate further Aerodynamic Statistics
    aircraft_data = breguet.drop(columns=['#', 'Aircraft Model Chart', 'Link', 'Factor', 'Ratio 1',
           'Ratio 2', 'K_1', 'K_2', 'K', 'A'])
    aircraft_data['Dmax'] = (g * aircraft_data['MTOW\n(Kg)']) / aircraft_data['L/D estimate']
    aircraft_data['Aspect Ratio'] = aircraft_data['Wingspan,float,metre']**2/aircraft_data['Wing area,float,square-metre']
    aircraft_data['c_L'] = (2* g * aircraft_data['MTOW\n(Kg)']) / (air_density*(flight_vel**2)*aircraft_data['Wing area,float,square-metre'])
    aircraft_data['c_D'] = aircraft_data['c_L'] / aircraft_data['L/D estimate']
    aircraft_data['k'] = 1 / (math.pi * aircraft_data['Aspect Ratio'] * 0.8)
    aircraft_data['c_Di'] = aircraft_data['k']*(aircraft_data['c_L']**2)
    aircraft_data['c_D0'] = aircraft_data['c_D']-aircraft_data['c_Di']

    # Drop some Rows and Save DF
    aircraft_data = aircraft_data.drop(columns=['MTOW\n(Kg)', 'MZFW_POINT_1\n(Kg)', 'RANGE_POINT_1\n(Km)', 'MZFW_POINT_2\n(Kg)', 'RANGE_POINT_2\n(Km)'])
    aircraft_data.to_excel(r'Databank.xlsx', index=False)