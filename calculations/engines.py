# %%

from pathlib import Path
import pandas as pd
import numpy as np


def determine_takeoff_to_cruise_tsfc_ratio(
    path_excel_engine_data_for_calibration: Path
) -> tuple[np.polynomial.Polynomial, np.polynomial.Polynomial, float, float]:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    path_excel_engine_data_for_calibration : Path
        _description_

    Returns
    -------
    float
        _description_
    """
    df_engines = pd.read_excel(
        io=path_excel_engine_data_for_calibration,
        sheet_name='Data',
        engine='openpyxl',
    )
    df_engines_grouped = df_engines.groupby(['Engine Identification'], as_index=False).agg(
        {
            'TSFC(cruise)' : 'mean',
            'TSFC(takeoff)' : 'mean',
            'Introduction':'mean'
        }
    )

    x = df_engines_grouped['TSFC(takeoff)']
    y = df_engines_grouped['TSFC(cruise)']

    linear_fit = np.polynomial.Polynomial.fit(
        x=x,
        y=y,
        deg=1,
    )
    polynomial_fit = np.polynomial.Polynomial.fit(
        x=x,
        y=y,
        deg=2,
    )

    def r_squared(y, y_pred):
        # https://en.wikipedia.org/wiki/Coefficient_of_determination#Definitions
        tss = np.sum((y - np.mean(y))**2)
        rss = np.sum((y - y_pred)**2)
        return 1 - (rss / tss)

    r_squared_linear = r_squared(y, linear_fit(x))
    r_squared_polynomial = r_squared(y, polynomial_fit(x))

    return linear_fit, polynomial_fit, r_squared_linear, r_squared_polynomial


# %%
def scale_engine_data_from_icao_emissions_database(
    path_excel_engine_data_icao_in: Path,
    path_excel_engine_data_icao_out: Path,
    scaling_polynomial: np.polynomial.Polynomial
) -> None:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    path_excel_engine_data_icao_in : Path
        _description_
    path_excel_engine_data_icao_out : Path
        _description_
    scaling_polynomial : np.polynomial.Polynomial
        _description_
    """

    df_engines = pd.read_excel(
        io=path_excel_engine_data_icao_in,
        sheet_name='Gaseous Emissions and Smoke',
        engine='openpyxl',
    )

    dict_columns_and_units = {
        'Engine Identification': ('Engine Identification', None),
        'Final Test Date': ('Final Test Date', 'pint[year]'),
        'Fuel Flow T/O (kg/sec)': ('Fuel Flow T/O', 'pint[kg/sec]'),
        'B/P Ratio': ('B/P Ratio', 'pint[]'),
        'Pressure Ratio': ('Pressure Ratio', 'pint[]'),
        'Rated Thrust (kN)': ('Rated Thrust', 'pint[kN]'),
    }

    df_engines = df_engines.rename(columns={old: new for old, (new, _) in dict_columns_and_units.items()})
    df_engines = df_engines.astype({col: unit for _, (col, unit) in dict_columns_and_units.items()})
    df_engines = df_engines.dropna(how='any')

    # calculate TSFC(takeoff) from fuel flow and rated thrust
    df_engines['TSFC(takeoff)'] = df_engines['Fuel Flow T/O'] / df_engines['Rated Thrust']
    # calculate TSFC(cruise) from polynomial provided by function `determine_takeoff_to_cruise_tsfc_ratio()`
    df_engines['TSFC(cruise)'] = df_engines['TSFC(takeoff)'].apply(lambda x:scaling_polynomial(x))

    df_engines.to_excel(
        excel_writer=path_excel_engine_data_icao_out,
        sheet_name='Data',
        engine='openpyxl',
    )

    return


# %%

path_excel = Path('/Users/michaelweinold/Library/CloudStorage/OneDrive-TheWeinoldFamily/Documents/University/PhD/Data/Aircraft Performance/Engine Database (TSFC Data).xlsx')

lin, pol, r2_lin, r2_pol  = determine_takeoff_to_cruise_tsfc_ratio(path_excel)



import matplotlib.pyplot as plt



fig = plt.figure(dpi=300)
plt.scatter(
    x=df_engines_grouped['TSFC(takeoff) [g/kNs]'],
    y=df_engines_grouped['TSFC(cruise) [g/kNs]'],
    color='black',
    label='Turbofan Engines',
)
plt.plot(
    np.arange(8, 19, 0.2),
    reg(np.arange(8, 19, 0.2)),
    color='blue',
    label='Linear Regression',
    linewidth=2,
)
plt.plot(
    np.arange(8, 19, 0.2),
    poly(np.arange(8, 19, 0.2)),
    color='red',
    label='Second Order Regression',
    linewidth=2,
)


# %%

from pathlib import Path
import pint_pandas
import pandas as pd

path_excel = Path('/Users/michaelweinold/Downloads/edb-emissions-databank_v30__web_.xlsx')

df_engines = pd.read_excel(
    io=path_excel,
    sheet_name='Gaseous Emissions and Smoke',
    engine='openpyxl',
)

dict_columns_and_units = {
    'Engine Identification': ('Engine Identification', None),
    'Final Test Date': ('Final Test Date', 'pint[year]'),
    'Fuel Flow T/O (kg/sec)': ('Fuel Flow T/O', 'pint[kg/sec]'),
    'B/P Ratio': ('B/P Ratio', 'pint[]'),
    'Pressure Ratio': ('Pressure Ratio', 'pint[]'),
    'Rated Thrust (kN)': ('Rated Thrust', 'pint[kN]'),
}

df_engines = df_engines[dict_columns_and_units.keys()]


# %%

index_units = df.columns.get_level_values(1)


# Add the second-level column labels as a new row
df = pd.concat([pd.DataFrame([second_level_labels], columns=df.columns), df], ignore_index=True)

# Drop the MultiIndex
df.columns = df.columns.get_level_values(0)



# %%

pd.read_excel(
    '/Users/michaelweinold/github/aircraft_performance_data_pipeline/calculations/out.xlsx',
    header=[0,1],
    )

