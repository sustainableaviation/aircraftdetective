from aircraftdetective.utility.statistics import r_squared
import pytest
import numpy as np

def test_r_squared_no_fit():
    """
    Tests if R^2 is 0.0 when the model is no better than the mean.
    If the prediction is just the mean of the actual values, RSS equals TSS,
    so R^2 should be 0.
    """
    y_actual = np.array([1, 2, 3, 4, 5])
    # The mean of y_actual is 3.0
    y_predicted = np.array([3, 3, 3, 3, 3])
    assert r_squared(y_actual, y_predicted) == pytest.approx(0.0)

def test_r_squared_typical_case():
    """
    Tests a standard case with some error between actual and predicted values.
    y = [1, 2, 3, 4, 5], y_bar = 3, TSS = 10
    y_pred = [1.1, 2.2, 2.9, 4.3, 5.1]
    RSS = (1-1.1)^2 + (2-2.2)^2 + (3-2.9)^2 + (4-4.3)^2 + (5-5.1)^2
        = 0.01 + 0.04 + 0.01 + 0.09 + 0.01 = 0.16
    R^2 = 1 - (0.16 / 10) = 0.984
    """
    y_actual = np.array([1, 2, 3, 4, 5])
    y_predicted = np.array([1.1, 2.2, 2.9, 4.3, 5.1])
    assert r_squared(y_actual, y_predicted) == pytest.approx(0.984)

def test_r_squared_negative_value():
    """
    Tests if R^2 is negative when the model is worse than predicting the mean.
    This happens when RSS > TSS.
    y = [1, 2, 3], y_bar = 2, TSS = 2
    y_pred = [5, 6, 7]
    RSS = (1-5)^2 + (2-6)^2 + (3-7)^2 = 16 + 16 + 16 = 48
    R^2 = 1 - (48 / 2) = -23
    """
    y_actual = np.array([1, 2, 3])
    y_predicted = np.array([5, 6, 7])
    assert r_squared(y_actual, y_predicted) == pytest.approx(-23.0)

def test_r_squared_zero_tss():
    """
    Tests the edge case where the total sum of squares (TSS) is zero.
    This occurs when all actual values are the same. The function should not
    raise a division-by-zero error.
    """
    y_actual = np.array([5, 5, 5, 5, 5])
    y_predicted_perfect = np.array([5, 5, 5, 5, 5])
    y_predicted_imperfect = np.array([5.1, 4.9, 5.0, 5.1, 4.9])
    
    # If prediction is also perfect, R^2 is 1.0
    assert r_squared(y_actual, y_predicted_perfect) == pytest.approx(1.0)
    # If prediction is not perfect, R^2 is conventionally 0.0
    assert r_squared(y_actual, y_predicted_imperfect) == pytest.approx(0.0)