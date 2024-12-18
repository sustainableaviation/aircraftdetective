# %%

import pint
import pint_pandas
ureg = pint.get_application_registry()


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




import sys
import os
module_path = os.path.abspath("/Users/michaelweinold/github/aircraftdetective")
if module_path not in sys.path:
    sys.path.append(module_path)


from aircraftdetective.utility import plotting
from aircraftdetective.auxiliary import dataframe


def determine_takeoff_to_cruise_tsfc_ratio(
    path_excel_engine_data_for_calibration: str
) -> tuple[np.polynomial.Polynomial, np.polynomial.Polynomial, float, float]:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    path_excel_engine_data_for_calibration : str
        _description_

    Returns
    -------
    float
        _description_
    """
    df_engines = pd.read_excel(
        io=path_excel_engine_data_for_calibration,
        sheet_name='Data',
        header=[0, 1],
        engine='openpyxl',
    )
    df_engines.columns = df_engines.columns.droplevel(1) # units not needed to compute the ratio
    df_engines = df_engines[df_engines[['TSFC(cruise)','TSFC(takeoff)']].notna().all(axis=1)]
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


def scale_engine_data_from_icao_emissions_database(
    path_excel_engine_data_icao_in: str,
    path_excel_engine_data_icao_out: str,
    scaling_polynomial: np.polynomial.Polynomial
) -> None:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    path_excel_engine_data_icao_in : str
        _description_
    path_excel_engine_data_icao_out : str
        _description_
    scaling_polynomial : np.polynomial.Polynomial
        _description_
    """

    df_engines = pd.read_excel(
        io=path_excel_engine_data_icao_in,
        sheet_name='Gaseous Emissions and Smoke',
        header=0,
        converters={'Final Test Date': lambda x: int(pd.to_datetime(x).year)},
        engine='openpyxl',
    )

    df_engines = dataframe.rename_columns_and_set_units(
        df=df_engines,
        column_names_and_units=[
            ("Engine Identification", "Engine Identification", str),
            ("Final Test Date", "Final Test Date", 'Int32'),
            ("Fuel Flow T/O (kg/sec)", "Fuel Flow T/O", "pint[kg/s]"),
            ("B/P Ratio", "B/P Ratio", "pint[dimensionless]"),
            ("Pressure Ratio", "Pressure Ratio", "pint[dimensionless]"),
            ("Rated Thrust (kN)", "Rated Thrust", "pint[kN]")
        ]
    )

    # calculate TSFC(takeoff) from fuel flow and rated thrust
    df_engines['TSFC(takeoff)'] = df_engines['Fuel Flow T/O'] / df_engines['Rated Thrust']
    # calculate TSFC(cruise) from polynomial provided by function `determine_takeoff_to_cruise_tsfc_ratio()`
    df_engines['TSFC(cruise)'] = df_engines['TSFC(takeoff)'].apply(lambda x: scaling_polynomial(x))
    # re-set units, because the apply function does not keep the pint units
    df_engines['TSFC(cruise)'] = df_engines['TSFC(cruise)'].astype(df_engines['TSFC(takeoff)'].dtype)

    return df_engines


def plot_takeoff_to_cruise_tsfc_ratio(
    df_engines: pd.DataFrame,
    linear_fit: np.polynomial.Polynomial,
    polynomial_fit: np.polynomial.Polynomial,
):
    
    list_tsfc_takeoff = np.linspace(
        start=0,
        stop=25,
        num=100
    )
    df_linear_fit = pd.DataFrame(
        {
            'TSFC(takeoff)': list_tsfc_takeoff,
            'TSFC(cruise)': linear_fit(list_tsfc_takeoff)
        }
    )
    df_polynomial_fit = pd.DataFrame(
        {
            'TSFC(takeoff)': list_tsfc_takeoff,
            'TSFC(cruise)': polynomial_fit(list_tsfc_takeoff)
        }
    )

    # SETUP ######################

    fig, ax = plotting.set_figure_and_axes()

    # AXIS LIMITS ################

    ax.set_xlim(5, 20)
    ax.set_ylim(12, 25)

    # TICKS AND LABELS ###########

    ax.set_xlabel('TSFC(takeoff) [g/kNs]')
    ax.set_ylabel("TSFC(cruise) [g/kNs]")

    # COLORMAP ###################

    cmap = plt.cm.viridis
    cmap.set_bad(color='red')

    # PLOTTING ###################

    scatterplot = ax.scatter(
        df_engines['TSFC(takeoff)'],
        df_engines['TSFC(cruise)'],
        c=df_engines['Introduction'],
        cmap=cmap,
        marker='o',
        edgecolor='k',
        plotnonfinite=True # for NaN values
    )
    ax.plot(
        df_linear_fit['TSFC(takeoff)'],
        df_linear_fit['TSFC(cruise)'],
        color='red',
        linestyle='--',
        label='Polynomial Fit'
    )
    ax.plot(
        df_polynomial_fit['TSFC(takeoff)'],
        df_polynomial_fit['TSFC(cruise)'],
        color='blue',
        linestyle='--',
        label='Linear Fit'
    )

    # LEGEND ####################

    cbar = fig.colorbar(mappable = scatterplot, ax=ax)
    cbar.set_label('Year of Introduction')

    fig.show()

"""
# %%

df_engines, pol1, pol2, _, _ = determine_takeoff_to_cruise_tsfc_ratio(path_excel_engine_data_for_calibration='/Users/michaelweinold/Library/CloudStorage/OneDrive-TheWeinoldFamily/Documents/University/PhD/Data/Aircraft Performance/Engine Database (TSFC Data).xlsx')

plot_takeoff_to_cruise_tsfc_ratio(
    df_engines=df_engines,
    linear_fit=pol1,
    polynomial_fit=pol2
)

scale_engine_data_from_icao_emissions_database(
    path_excel_engine_data_icao_in='https://www.easa.europa.eu/en/downloads/131424/en',
    path_excel_engine_data_icao_out='Scaled.xlsx',
    scaling_polynomial=pol2
)
"""