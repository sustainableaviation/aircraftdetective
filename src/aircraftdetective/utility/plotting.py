import pint
from aircraftdetective import ureg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
cm = 1/2.54 # for inches-cm conversion

def set_figure_and_axes():
    
    fig, ax = plt.subplots(
        num = 'main',
        nrows = 1,
        ncols = 1,
        dpi = 300,
        figsize=(10*cm, 10*cm),
    )

    # TICKS AND LABELS ###########

    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=True)
    ax.tick_params(axis='y', which='minor', bottom=True)

    # GRIDS ######################

    ax.grid(which='major', axis='y', linestyle='-', linewidth = 0.5)
    ax.grid(which='minor', axis='y', linestyle=':', linewidth = 0.5)
    ax.grid(which='major', axis='x', linestyle='-', linewidth = 0.5)
    ax.grid(which='minor', axis='x', linestyle=':', linewidth = 0.5)

    return fig, ax


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

    ax.set_xlim(5, 20)
    ax.set_ylim(12, 25)

    # TICKS AND LABELS ###########

    ax.set_xlabel('TSFC (takeoff) [g/kNs]')
    ax.set_ylabel('TSFC (cruise) [g/kNs]')

    # COLORMAP ###################

    cmap = plt.cm.viridis
    cmap.set_bad(color='red')

    # PLOTTING ###################

    scatterplot = ax.scatter(
        df_engines['TSFC (takeoff)'],
        df_engines['TSFC (cruise)'],
        c=df_engines['Introduction'],
        cmap=cmap,
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

    cbar = fig.colorbar(mappable = scatterplot, ax=ax)
    cbar.set_label('Year of Introduction')

    fig.show()


@ureg.check(
    '[time]/[length]',
    '[speed]',
)
def compute_overall_engine_efficiency(
    TSFC_cruise: pint.Quantity,
    v_cruise: pint.Quantity,
) -> pint.Quantity:
    r"""
    Given the thrust-specific fuel consumption $TSFC$ and the cruise velocity $v_0$,
    returns the overall engine efficiency $\eta_0$ under the assumption of the fuel heating value $LHV_{fuel}$ of Jet A1 fuel.

    Parameters
    ----------
    TSFC : pint.Quantity
        Thrust-specific fuel consumption, in units of [time]/[length]
    v_cruise : pint.Quantity
        Cruise velocity, in units of [speed]

    See Also
    --------
    - [Young (2018), eqn. (8.24, solved for η₀) and figure 8.3.](https://doi.org/10.1002/9781118534786)

    Returns
    -------
    pint.Quantity
        Overall engine efficiency, dimensionless
    """
    fuel_heating_value = 44.1 * ureg('MJ/kg') # heating value of Jet A-1 fuel
    return (v_cruise.to_base_units() / TSFC_cruise.to_base_units()) * (1 / fuel_heating_value.to_base_units())


def compute_engine_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df["Engine Efficiency"] = df.apply(
        lambda row: compute_overall_engine_efficiency(
            TSFC_cruise=row["TSFC (cruise)"],
            v_cruise=row["Cruise Speed"],
        ),
        axis=1
    ).pint.convert_object_dtype()
    return df


def get_engine_designations_with_wildcard(df: pd.DataFrame) -> set[str]:
    r"""
    Given a dataframe of aircraft data, returns a set of engine designations which contain a wildcard.

    For example, given a DataFrame of the kind:

    | Engine Designation | (...) |
    |--------------------|-------|
    | GEnx-1B.*          | (...) |
    | CFM56-7B24         | (...) |

    the function returns the set:

    `{GEnx-1B.*}`

    Parameters
    ----------
    df : pd.DataFrame
        Aircraft data, including column 'Engine Designation'

    Returns
    -------
    set[str]
        Set of (unique) engine designations with wildcards
    """
    return set(df[df['Engine Designation'].str.endswith(".*", na=False)]['Engine Designation'])


def average_engine_data_by_model(
    df: pd.DataFrame,
    engine_models_to_average: list[str],
) -> pd.DataFrame:
    r"""
    Given a DataFrame of engine data and a list of engine models to average,
    returns a DataFrame with the average data for the engine models appended.

    For example, given an iterable of engines to average,
    correctly formatted with regex wildcards (note: `.*` instead of just `*`):

    `{GEnx-1B.*}`

    and a DataFrame of the kind:

    | Engine Identification | (...) | Rated Thrust      |
    |-----------------------|-------|-------------------|
    | 'GEnx-1B54'           | (...) | 255.3             |
    | 'GEnx-1B58'           | (...) | 271.3             |
    | 'GEnx-1B64'           | (...) | 298               |

    the function returns a DataFrame of the kind:

    | Engine Identification | (...) | Rated Thrust      |
    |-----------------------|-------|-------------------|
    | 'GEnx-1B54'           | (...) | 255.3             |
    | 'GEnx-1B58'           | (...) | 271.3             |
    | 'GEnx-1B64'           | (...) | 298               |
    | 'GEnx-1B*'            | (...) | 274.87            |

    Parameters
    ----------
    df : pd.DataFrame
        Engine data, including column 'Engine Identification'
    engine_models_to_average : list[str]
        Iterable of engine models to average, correctly formatted with regex wildcards

    Returns
    -------
    pd.DataFrame
        Engine data with averaged values for the engine models in the list
    """
    df_matched = df.copy()
    df_matched['Engine Identification'] = df_matched['Engine Identification'].apply(
        lambda engine_identification: next(
                (
                    engine_pattern for engine_pattern in engine_models_to_average
                    if re.match(engine_pattern, engine_identification)
                ),
                None
            )
    )
    df_matched = df_matched.dropna(subset=['Engine Identification'])
    df_matched = df_matched.groupby(['Engine Identification'], as_index=False).agg('mean')
    df = pd.concat(
        objs=[df, df_matched],
        axis=0
    ).reset_index(drop=True)
    return df