# %%

from pathlib import Path
import sys
import os

import pandas as pd

module_path = str(Path(__file__).parents[1])
if module_path not in sys.path:
    sys.path.append(module_path)

from aircraftdetective.calculations import engines
from aircraftdetective import config

# %%

df_engines_tsfc, pol1, pol2, _, _ = engines.determine_takeoff_to_cruise_tsfc_ratio(
    path_excel_engine_data_for_calibration=config['aircraft_detective']['url_xlsx_engine_tsfc_database']
)

df_engines = engines.scale_engine_data_from_icao_emissions_database(
    path_excel_engine_data_icao_in=config['ICAO']['url_xlsx_engine_emissions_databank'],
    path_excel_engine_data_icao_out='Scaled.xlsx',
    scaling_polynomial=pol2
)

df_aircraft = pd.read_excel(
    io=config['aircraft_detective']['url_xlsx_aircraft_database'],
    sheet_name='Raw Data',
    header=[0, 1],
    engine='openpyxl',
)
df_aircraft = df_aircraft.pint.quantify(level=1)

engines_wildcard: set = engines.get_engine_designations_with_wildcard(df=df_aircraft)

df_engines = engines.average_engine_data_by_model(
    df=df_engines,
    engine_models_to_average=engines_wildcard
)

df_merged = pd.merge(
    left=df_aircraft,
    right=df_engines,
    how='left',
    left_on='Engine Designation',
    right_on='Engine Identification',
)

df_merged = engines.compute_engine_metrics(
    df=df_merged
)

# %%

def plot_takeoff_to_cruise_tsfc_ratio(
    df_engines: pd.DataFrame,
    linear_fit: np.polynomial.Polynomial,
    polynomial_fit: np.polynomial.Polynomial,
):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    df_engines : pd.DataFrame
        _description_
    linear_fit : np.polynomial.Polynomial
        _description_
    polynomial_fit : np.polynomial.Polynomial
        _description_
    """
    
    # DATA PREPARATION ##########

    df_engines = df_engines.copy()
    df_engines = df_engines.pint.dequantify()

    max_tsfc_takeoff = df_engines['TSFC (takeoff)'].max().item()
    min_tsfc_takeoff = df_engines['TSFC (takeoff)'].min().item()

    list_tsfc_takeoff = np.linspace(
        start=min_tsfc_takeoff - 5,
        stop=max_tsfc_takeoff + 5,
        num=100
    )
    df_linear_fit = pd.DataFrame(
        {
            'TSFC (takeoff)': list_tsfc_takeoff,
            'TSFC (cruise)': linear_fit(list_tsfc_takeoff)
        }
    )
    df_polynomial_fit = pd.DataFrame(
        {
            'TSFC (takeoff)': list_tsfc_takeoff,
            'TSFC (cruise)': polynomial_fit(list_tsfc_takeoff)
        }
    )

    # SETUP ######################

    fig, ax = plotting.set_figure_and_axes()

    # AXIS LIMITS ################

    #ax.set_xlim(5, 20)
    #ax.set_ylim(12, 25)

    # TICKS AND LABELS ###########

    ax.set_xlabel('TSFC (takeoff) [g/kNs]')
    ax.set_ylabel('TSFC (cruise) [g/kNs]')

    # PLOTTING ###################

    scatterplot = ax.scatter(
        df_engines['TSFC (takeoff)'],
        df_engines['TSFC (cruise)'],
        marker='o',
        edgecolor='k',
        plotnonfinite=True # for NaN values
    )
    ax.plot(
        df_linear_fit['TSFC (takeoff)'],
        df_linear_fit['TSFC (cruise)'],
        color='red',
        linestyle='--',
        label='Polynomial Fit'
    )
    ax.plot(
        df_polynomial_fit['TSFC (takeoff)'],
        df_polynomial_fit['TSFC (cruise)'],
        color='blue',
        linestyle='--',
        label='Linear Fit'
    )

    # LEGEND ####################

    fig.show()