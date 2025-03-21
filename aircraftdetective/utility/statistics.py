import numpy as np

def r_squared(
    y: np.ndarray,
    y_pred: np.ndarray
) -> float:
    r"""
    Given an array of observed values and an array of prediced values,
    determines the coefficient of determination ($R^2$)

    $$
    R^2 = 1 - \frac{RSS}{TSS}
    $$
    
    with

    $$
    \begin{align*}
    RSS &= \sum (y_i - \hat{y}_i)^2 \\
    TSS &= \sum (y_i - \bar{y})^2
    \end{align*}
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
    """
    tss = np.sum((y - np.mean(y))**2)
    rss = np.sum((y - y_pred)**2)
    return 1 - (rss / tss)