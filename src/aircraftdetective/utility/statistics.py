# %%
import numpy as np
import pandas as pd
from typing import Any
import plotly.graph_objects as go


def _compute_polynomials_from_dataframe(
    df: pd.DataFrame,
    col_name_x: str,
    list_col_names_y: list[str],
    degree: int,
    plot: bool = False
) -> dict[str, Any]:
    r"""
    Computes polynomial fits of a given degree for each column in a dataframe.

    Given a dataframe with at least two columns, computes a polynomial fit of the specified degree
    for each column against the specified x-axis column. Returns a dictionary containing the polynomial
    fits and their corresponding R-squared values.

    See Also
    --------
    [`numpy.polynomial.Polynomial.fit`](https://numpy.org/doc/2.0/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html)  
    [`aircraftdetective.utility.statistics._r_squared`][]

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data for polynomial fitting.
    col_name_x : str
        Name of the column to be used as the x-axis for polynomial fitting.
    degree : int
        Degree of the polynomial to fit.
    
    Returns
    -------
    dict[str, Any]
        A dictionary where keys are column names and values are the corresponding polynomial fits.  
        Of the kind:  
        ```
        {
            'Column1': Polynomial object,
            'Column1_r2': float,
            'Column2': Polynomial object,
            'Column2_r2': float,
            ...
        }
        ```

    Example
    -------
    ```pyodide install='aircraftdetective'
    import pandas as pd
    from aircraftdetective.utility.statistics import _compute_polynomials_from_dataframe
    data = {
        'Year': [2000, 2005, 2010, 2015],
        'Value1': [10, 15, 20, 25],
        'Value2': [30, 25, 20, 15]
    }
    df = pd.DataFrame(data)
    _compute_polynomials_from_dataframe(df, 'Year', ['Value1', 'Value2'], degree=2)
    ```
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a Pandas DataFrame")
    if df.empty:
        raise ValueError("df cannot be empty")
    if col_name_x not in df.columns:
        raise ValueError(f"col_name_x '{col_name_x}' not found in df columns")
    if not isinstance(degree, int) or degree < 0:
        raise ValueError("degree must be a non-negative integer")
    if len(df) <= degree:
        raise ValueError("number of data points must be greater than degree")
    
    df_func = df.copy()

    df_func.sort_values(by=col_name_x, ascending=True, inplace=True)
    df_func.dropna(subset=[col_name_x], inplace=True)

    dict_polynomials = {}
    for col in list_col_names_y:
        x_unfiltered = df_func[col_name_x].astype("float64")
        y_unfiltered = df_func[col].astype("float64")
        mask = y_unfiltered.notna() # ensure all NaNs are removed, otherwise the fit will fail
        x = x_unfiltered[mask]
        y = y_unfiltered[mask]
        polynomial_fit = np.polynomial.Polynomial.fit(
            x=x,
            y=y,
            deg=degree,
        )
        _r_squared_polynomial = _r_squared(y, polynomial_fit(x))
        dict_polynomials[col] = polynomial_fit
        dict_polynomials[f'{col}_r2'] = _r_squared_polynomial

    if plot is True:
        fig = go.Figure()
        for col in list_col_names_y:
            if col not in dict_polynomials:
                continue

            x_data = df_func[col_name_x].astype("float64")
            y_data = df_func[col].astype("float64")
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                name=f'{col} (Original Data)',
                marker=dict(opacity=0.7)
            ))
            
            polynomial_fit = dict_polynomials[col]
            r_squared = dict_polynomials[f'{col}_r2']
            x_fit = np.linspace(x_data.min(), x_data.max(), num=200)
            y_fit = polynomial_fit(x_fit)
            fig.add_trace(go.Scatter(
                x=x_fit, 
                y=y_fit,
                mode='lines',
                name=f'{col} (Fit, R²={r_squared:.3f})',
                line=dict(width=3)
            ))

        fig.show()

    return dict_polynomials


def _r_squared(
    y: np.ndarray,
    y_pred: np.ndarray
) -> float:
    r"""
    Given a [NumPy `ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) of observed values
    and a [NumPy `ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)of prediced values,
    determines the coefficient of determination ($R^2$)
    $$
    R^2 = 1 - \frac{RSS}{TSS}
    $$
    with
    $$
    RSS = \sum (y_i - \hat{y}_i)^2
    $$
    $$
    TSS = \sum (y_i - \bar{y})^2
    $$
    where:

    | Symbol        | Description                     |
    |---------------|---------------------------------|
    | $R^2$         | Coefficient of determination    |
    | $RSS$         | Residual sum of squares         |
    | $TSS$         | Total sum of squares            |
    | $y_i$         | Actual value                    |
    | $\hat{y}_i$   | Predicted value                 |
    | $\bar{y}$     | Mean of actual values           |

    References
    ----------
    Eqn. (5.2.4) in [Draper and Smith (1998)](https://doi.org/10.1002/9781118625590)  
    [Coefficient of Determination on Wikipedia](https://en.wikipedia.org/wiki/Coefficient_of_determination)

    See Also
    --------
    [`aircraftdetective.utility.statistics._compute_polynomials_from_dataframe`][]  
    
    Parameters
    ----------
    y : np.ndarray
        Array of observed values
    y_pred : np.ndarray
        Array of predicted values

    Returns
    -------
    float
        Coefficient of determination ($R^2$)

    Example
    -------
    ```pyodide install='aircraftdetective'
    import numpy as np
    from aircraftdetective.utility.statistics import _r_squared
    y = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    _r_squared(y, y_pred)
    ```
    """
    tss = np.sum((y - np.mean(y))**2)
    if tss == 0: # occurs when all y values are the same.
        rss_check = np.sum((y - y_pred)**2)
        return 1.0 if rss_check == 0 else 0.0
    rss = np.sum((y - y_pred)**2)
    return 1 - (rss / tss)