# %%
import math
import pandas as pd
import numpy as np


def compute_efficiency_improvement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    Computes the relative improvement $\Delta\%x$ between times $t=0$ and $t=T$
    $$
    \Delta\%x=\frac{x(t=T)-x(t=0)}{x(t=0)} \times 100 [\%]
    $$
    and the index $I_x$ between times $t=0$ and $t=T$
    $$
    I_x=\frac{x(t=T)}{x(t=0)}
    $$
    of relevant aircraft efficiency metrics in a dataframe.

    Given an input DataFrame, the function computes [relative improvements](https://en.wikipedia.org/wiki/Percentage) 
    and [indices](https://en.wikipedia.org/wiki/Index_(economics)) 
    of the overall aircraft efficiency metrics energy use (EU) and energy intensity (EI) 
    as well as the aircraft sub-efficiencies thrust-specific fuel consumption (TSFC), 
    weight per available seat (OEW/Exit Limit) and lift-to-drag ratio (L/D), 
    all compared to the first year available in the dataframe.
    
    Notes
    -----
    Aircraft energy use $E_U$ as per Eqn. (4) in [Babikian et al. (2002)](https://doi.org/10.1016/S0969-6997(02)00020-0)
    is proportional to the product of thrust-specific fuel consumption,
    the weight weight per available seat and the lift-to-drag ratio:
    $$
    E_U [\text{J/ASK}] \propto TSFC \times \frac{W}{pax} \times \bigg(\frac{L}{D}\bigg)^{-1} 
    $$
    Aircraft energy intensity $E_I$ as per Eqn. (1) in [Babikian et al. (2002)](https://doi.org/10.1016/S0969-6997(02)00020-0)
    is defined as the energy use per revenue passenger kilometer (RPK):
    $$
    E_I [\text{J/RPK}] = \frac{E_U [\text{J/ASK}]}{SLF}
    $$
    where
    
    | Symbol        | Unit        | Description                         | 
    |---------------|-------------|-------------------------------------|
    | $E_U$         | J/km        | Energy use _per available seat km_    |
    | $E_I$         | J/km        | Energy intensity _per revenue passenger km_ |
    | $TSFC$        | N/(W·s)     | Thrust-specific fuel consumption    |
    | $W/pax$       | N           | Weight per available seat (OEW/Exit Limit) |
    | $L/D$         | -           | Lift-to-drag ratio                  |
    | $SLF$         | -           | Seat load factor                    |

    Overall efficiency (technological only) can also be written in terms of aircraft sub-efficiencies as:
    $$
    \eta_{tech} \propto \eta_{eng} \times \eta_{aero} \times \eta_{struct}
    $$
    and overall efficiecy (technological and operational) can also be written in terms of aircraft sub-efficiencies as:
    $$
    \eta_{tech+ops} \propto \eta_{eng} \times \eta_{aero} \times \eta_{struct} \times \eta_{ops}
    $$
    where
    
    | Symbol          | Definition                      | Description            | 
    |-----------------|---------------------------------|------------------------|
    | $\eta_{tot}$    | $\propto E_U^{-1}$              | Total efficiency       |
    | $\eta_{eng}$    | $\propto TSFC^{-1}$             | Engine efficiency      |
    | $\eta_{aero}$   | $\propto L/D$                   | Aerodynamic efficiency |
    | $\eta_{struct}$ | $\propto (OEW/\text{Exit Limit})^{-1}$ | Structural efficiency  |
    | $\eta_{ops}$    | $\propto SLF$                   | Operational efficiency |

    Warning
    -------
    Since, for example, $\eta_{aero}\propto L/D$, the relative improvement
    is calculated as:
    $$
    \Delta\%(L/D) = \frac{(L/D)(t=T)-(L/D)(t=0)}{(L/D)(t=0)} \times 100 [\%]
    $$
    and
    $$
    I_{L/D} = \frac{(L/D)(t=T)}{(L/D)(t=0)}
    $$
    while for $\eta_{eng}\propto TSFC^{-1}$, the relative improvement is calculated as:
    $$
    \Delta\%(TSFC) = \frac{(TSFC)(t=T)-(TSFC)(t=0)}{(TSFC)(t=0)} \times 100 [\%]
    $$
    and
    $$
    I_{TSFC} = \frac{(TSFC)(t=T)}{(TSFC)(t=0)}
    $$
    and thus a _decrease_ in TSFC results in an _increase_ in engine efficiency. 

    References
    ----------
    Babikian et al. (2002), "The historical fuel efficiency characteristics of regional aircraft from technological, operational, and cost perspectives",
    _Journal of Air Transport Management_, doi:[10.1016/S0969-6997(02)00020-0](https://doi.org/10.1016/S0969-6997(02)00020-0)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the aircraft sub-efficiency data.  
        Must contain at least the columns:

        - `Year`
        - `Type`
        - `Energy Use (per ASK)`
        - `Energy Intensity (per RPK)`
        - `TSFC (cruise)`
        - `OEW/Exit Limit`
        - `L/D`
        - `SLF`

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional columns for relative improvements in EU, TSFC, OEW/Exit Limit, and L/D:
        
        - `Index(EU)`
        - `Index(EI)`
        - `Index(Engines)`
        - `Index(Weight)`
        - `Index(Aerodynamics)`
        - `Index(Operations)`
        - `Percent(EU)`
        - `Percent(EI)`
        - `Percent(Engines)`
        - `Percent(Weight)`
        - `Percent(Aerodynamics)`
        - `Percent(Operations)`

    Raises
    ------
    ValueError
        If `df` is not a Pandas DataFrame, is empty, or does not contain the required columns.  
        If any of the required columns contain only NaN values or are not of numeric type.
    
    Example
    -------
    ```pyodide install='aircraftdetective'
    import pandas as pd
    from aircraftdetective.calculations.decomposition import compute_efficiency_improvement_metrics
    ```
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input `df` must be a non-empty Pandas DataFrame.")

    list_required_cols = [
        'Year',
        'Energy Use (per ASK)',
        'Energy Intensity (per RPK)',
        'TSFC (cruise)',
        'OEW/Exit Limit',
        'L/D',
        'SLF',
    ]
    for col in list_required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in df columns")
        if df[col].isnull().all():
            raise ValueError(f"Column '{col}' cannot be all NaN")
        if col not in ['Year'] and not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be of a numeric type.")

    df_func = df.copy()
    df_func.sort_values(by='Year', ascending=True, inplace=True)

    metrics_inverse = {
        'Energy Use (per ASK)': True,        # lower is better
        'Energy Intensity (per RPK)': True,  # lower is better
        'TSFC (cruise)': True,               # lower is better
        'OEW/Exit Limit': True,              # lower is better
        'L/D': False,                        # higher is better
        'SLF': False                         # higher is better
    }

    # Map input metric columns to their desired output name suffixes
    metric_mapping = {
        'Energy Use (per ASK)': 'EU',
        'Energy Intensity (per RPK)': 'EI',
        'TSFC (cruise)': 'Engines',
        'OEW/Exit Limit': 'Weight',
        'L/D': 'Aerodynamics',
        'SLF': 'Operations'
    }
    
    metrics = list(metric_mapping.keys())

    baselines = {}
    for metric in metrics:
        baselines[metric] = df_func[metric].dropna().iloc[0]

    # Loop using the metric_mapping to create the new column names
    for metric, output_suffix in metric_mapping.items():
        s = df_func[metric]
        x0 = baselines[metric] # This is now a scalar baseline value
        inverse = metrics_inverse[metric]

        # Directional Index (>1 if improved)
        # lower-better:  x0/x
        # higher-better: x/x0
        idx_col = f'Index({output_suffix})'
        with np.errstate(divide='ignore', invalid='ignore'):
            if inverse:
                df_func[idx_col] = np.where(s != 0, x0 / s, np.nan)
            else:
                df_func[idx_col] = np.where(x0 != 0, s / x0, np.nan)

        # Directional Percent (>0 if improved)
        # lower-better:  (x0/x - 1) * 100
        # higher-better: (x/x0 - 1) * 100
        pct_col = f'Percent({output_suffix})'
        with np.errstate(divide='ignore', invalid='ignore'):
            if inverse:
                df_func[pct_col] = np.where(s != 0, (x0 / s - 1.0) * 100.0, np.nan)
            else:
                df_func[pct_col] = np.where(x0 != 0, (s / x0 - 1.0) * 100.0, np.nan)

    return df_func


def compute_lmdi_factor_contributions(
    aggregate_t1: float,
    aggregate_t2: float,
    factor_t1: float,
    factor_t2: float
) -> float:
    r"""
    Computes the contributions of changes in factors to the change in an aggregate
    according to the additive logarithmic mean Divisia index method I (LMDI-I).

    Given values for an aggregate $C$ and a factor $C_i$
    $$
    C = C_1 \times C_2 \times ... \times C_n
    $$
    at two points in time $(t_1, t_2)$,
    computes the contribution $\Delta C_i$ of the change in factor $C_i$
    to the change in total $\Delta C$ through the
    additive logarithmic mean Divisia index method I (LMDI-I).
    $$
    \Delta C_i = \left(\frac{C(t_2) - C(t_1)}{\ln C(t_2) - \ln C(t_1)}\right) \times \ln\left(\frac{C_i(t_2)}{C_i(t_1)}\right)
    $$
    where

    | Symbol | Unit     | Description |
    |--------|----------|-------------|
    | $C$    | -        | aggregate   |
    | $C_i$  | -        | factor $i$  |
    
    Notes
    -----
    The LDMI-I method is _additive_, which means that the contributions sum up to the total change:
    $$
    \Delta C = \sum_i \Delta C_i
    $$

    References
    ----------
    Ang & Goh (2019), "Index decomposition analysis for comparing emission scenarios: Applications and challenges", 
    _Energy Economics_, doi:[10.1016/j.eneco.2019.06.013](https://doi.org/10.1016/j.eneco.2019.06.013)  
    Eqn. (28) and Table 5 in Ang & Zhang (2000), "A survey of index decomposition analysis in energy and environmental studies",
    _Energy_, doi:[10.1016/S0360-5442(00)00039-6](https://doi.org/10.1016/S0360-5442(00)00039-6)

    Parameters
    ----------
    aggregate_t1 : float
        Value of the aggregate $C$ at time $t_1$
    aggregate_t2 : float
        Value of the aggregate $C$ at time $t_2$
    factor_t1 : float
        Value of the factor $C_i$ at time $t_1$
    factor_t2 : float
        Values of the factor $C_i$ at time $t_2$

    Returns
    -------
    float
        Contribution of change in sub-efficiency to change in total efficiency

    Example
    -------
    ```pyodide install='aircraftdetective'
    aggregate_t1 = 8.0
    aggregate_t2 = 64.0
    factor_1_t1 = 2.0
    factor_1_t2 = 4.0
    factor_2_t1 = 4.0
    factor_2_t2 = 16.0
    delta_factor_1 = _compute_lmdi_factor_contributions(aggregate_t1, aggregate_t2, factor_1_t1, factor_1_t2)
    delta_factor_2 = _compute_lmdi_factor_contributions(aggregate_t1, aggregate_t2, factor_2_t1, factor_2_t2)
    ```
    """
    if aggregate_t1 <= 0 or aggregate_t2 <= 0 or factor_t1 <= 0 or factor_t2 <= 0:
        raise ValueError("LMDI inputs (aggregates and factors) must be positive.")
        
    if factor_t1 == factor_t2:
        return 0.0  # No change in factor, so contribution is zero

    if aggregate_t1 == aggregate_t2:
        log_mean_aggregate = aggregate_t1
    else:
        delta_aggregate = aggregate_t2 - aggregate_t1
        log_mean_aggregate = delta_aggregate / (math.log(aggregate_t2) - math.log(aggregate_t1))

    delta_factor = log_mean_aggregate * math.log(factor_t2 / factor_t1)
    
    return delta_factor

def _compute_lmdi_factor_contributions_vectorized(
    aggregate_series: pd.Series, 
    factor_series: pd.Series
) -> pd.Series:
    r"""
    Vectorized version of compute_lmdi_factor_contributions, which
    computes the contribution of a factor by comparing every row to 
    the first row (t_1).

    This implementation hard-codes the $t_1$ values to be the
    scalar values from the first row (`.iloc[0]`) of the input Series.

    Parameters
    ----------
    aggregate_series : pd.Series
        A Series containing all aggregate values ($C$). The value
        at index 0 will be used as $C(t_1)$.
    factor_series : pd.Series
        A Series containing all factor values ($C_i$). The value
        at index 0 will be used as $C_i(t_1)$.

    Returns
    -------
    pd.Series
        The calculated additive contribution of the factor to the 
        aggregate, relative to the first row. The first row of
        the output will always be 0.0.
    """
    if aggregate_series.empty or factor_series.empty:
        return pd.Series(dtype='float64')

    # --- Set t1 (first row) and t2 (all rows) ---
    
    # t1 values are scalars from the first row
    aggregate_t1 = aggregate_series.iloc[0]
    factor_t1 = factor_series.iloc[0]
    
    aggregate_t2 = aggregate_series
    factor_t2 = factor_series
    
    delta_aggregate = aggregate_t2 - aggregate_t1
    log_diff_aggregate = np.log(aggregate_t2) - np.log(aggregate_t1)

    log_mean_aggregate = np.where(
        delta_aggregate == 0, 
        aggregate_t1,  # If no change, log mean is just the value
        delta_aggregate / log_diff_aggregate
    )
    
    log_ratio_factor = np.log(factor_t2 / factor_t1)
    
    delta_contribution = log_mean_aggregate * log_ratio_factor
    
    return pd.Series(delta_contribution, index=aggregate_series.index).fillna(0.0)

def compute_efficiency_disaggregation(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    Computes the LMDI disaggregation of aircraft sub-efficiency factors.

    This function decomposes the change in an aggregate efficiency index
    (e.g., `Index(EU)`) relative to the baseline year (the first year in
    the DataFrame) into the additive contributions from its constituent factors
    (e.g., `Index(Engines)`, `Index(Weight)`).
    
    The baseline ($t_1$) values are taken from the first row of the
    DataFrame (after sorting by `Year`). The current row's values are used as $t_2$.

    See Also
    --------
    [`compute_lmdi_factor_contributions`][aircraftdetective.calculations.decomposition.compute_lmdi_factor_contributions]

    References
    ----------
    Ang & Goh (2019), "Index decomposition analysis for comparing emission scenarios: Applications and challenges", 
    _Energy Economics_, doi:[10.1016/j.eneco.2019.06.013](https://doi.org/10.1016/j.eneco.2019.06.013) 
    Eqn. (28) and Table 5 in Ang & Zhang (2000), "A survey of index decomposition analysis in energy and environmental studies",
    _Energy_, doi:[10.1016/S0360-5442(00)00039-6](https://doi.org/10.1016/S0360-5442(00)00039-6)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the aircraft efficiency index data. 
        Must contain at least the columns:

        - `Year`
        - `Index(EU)`
        - `Index(EI)`
        - `Index(Engines)`
        - `Index(Weight)`
        - `Index(Aerodynamics)`
        - `Index(Operations)`

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional columns for the LMDI contributions
        of each factor to the change in aggregate efficiency from the
        baseline year.

        - `ContributionEU(Engines)`
        - `ContributionEU(Weight)`
        - `ContributionEU(Aerodynamics)`
        - `ContributionEI(Engines)`
        - `ContributionEI(Weight)`
        - `ContributionEI(Aerodynamics)`
        - `ContributionEI(Operations)`

    Raises
    ------
    ValueError
        If `df` is not a Pandas DataFrame, is empty, or does not contain the required columns. 
        If any of the required columns contain only NaN values or are not of numeric type.
        If any Index columns contain non-positive values, which are invalid for LMDI.

    """
    if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input `df` must be a non-empty Pandas DataFrame.")

    list_required_cols = [
        'Year',
        'Index(EU)',
        'Index(EI)',
        'Index(Engines)',
        'Index(Weight)',
        'Index(Aerodynamics)',
        'Index(Operations)',
    ]

    for col in list_required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in df columns")
        if df[col].isnull().all():
            raise ValueError(f"Column '{col}' cannot be all NaN")        
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be of a numeric type.")
            
    numeric_cols = [c for c in list_required_cols if c not in ['Year']]
    if (df[numeric_cols] <= 0).any().any():
            raise ValueError("All Index columns must contain positive values for LMDI.")

    df_func = df.copy()
    df_func = df_func.sort_values(by='Year', ascending=True)

    map_aggregates = {
        'EU': ['Index(Engines)', 'Index(Weight)', 'Index(Aerodynamics)'],
        'EI': ['Index(Engines)', 'Index(Weight)', 'Index(Aerodynamics)', 'Index(Operations)']
    }
    
    baseline_row = df_func.iloc[0]
    
    for aggregate_type, list_factors in map_aggregates.items():
        col_name_aggregate = f'Index({aggregate_type})'
        col_aggregate = df_func[col_name_aggregate]
        val_aggregate_t1 = baseline_row[col_name_aggregate]
        
        for factor in list_factors:
            col_factor = df_func[factor]
            val_factor_t1 = baseline_row[factor]
            
            # Create output column name, e.g., "ContributionEU(Engines)"
            factor_suffix = factor.replace('Index(', '').replace(')', '')
            output_col = f"Contribution{aggregate_type}({factor_suffix})"
            
            df_func[output_col] = _compute_lmdi_factor_contributions_vectorized(
                aggregate_t1=val_aggregate_t1,
                aggregate_t2=col_aggregate,
                factor_t1=val_factor_t1,
                factor_t2=col_factor
            )
    
    return df_func