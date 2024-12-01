# %%

from pathlib import Path
import pandas as pd
import numpy as np

import auxiliary.dataframe


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
        header=[0, 1],
        engine='openpyxl',
    )
    df_engines.columns = df_engines.columns.droplevel(1) # units not needed to compute the ratio
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
        header=0,
        converters={'Final Test Date': lambda x: int(pd.to_datetime(x).year)},
        engine='openpyxl',
    )

    columns_and_units = [
        ("Engine Identification", "Engine Identification", str),
        ("Final Test Date", "Final Test Date", int),
        ("Fuel Flow T/O (kg/sec)", "Fuel Flow T/O", "pint[kg/sec]"),
        ("B/P Ratio", "B/P Ratio", "pint[dimensionless]"),
        ("Pressure Ratio", "Pressure Ratio", "pint[dimensionless]"),
        ("Rated Thrust (kN)", "Rated Thrust", "pint[kN]")
    ]

    subset = [name_old for name_old, _, _ in columns_and_units]
    df_engines = df_engines[subset]
    df_engines = df_engines.dropna(how='any')

    for name_old, name_new, dtype in columns_and_units:
        df_engines = df_engines.rename(columns={name_old: name_new})
        df_engines[name_new] = df_engines[name_new].astype(dtype)

    # calculate TSFC(takeoff) from fuel flow and rated thrust
    df_engines['TSFC(takeoff)'] = df_engines['Fuel Flow T/O'] / df_engines['Rated Thrust']
    # calculate TSFC(cruise) from polynomial provided by function `determine_takeoff_to_cruise_tsfc_ratio()`
    df_engines['TSFC(cruise)'] = df_engines['TSFC(takeoff)'].apply(lambda x: scaling_polynomial(x))
    # re-set units, because the apply function does not keep the pint units
    df_engines['TSFC(cruise)'] = df_engines['TSFC(cruise)'].astype(df_engines['TSFC(takeoff)'].dtype)

    auxiliary.dataframe.export_typed_dataframe_to_excel(df=df_engines, path=path_excel_engine_data_icao_out)