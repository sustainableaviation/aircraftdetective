import pytest
import numpy as np
import pandas as pd
from aircraftdetective.utility.statistics import (
    _r_squared,
    _compute_polynomials_from_dataframe
)


class TestComputePolynomialsFromDataframe:
    """
    Test suite for the `_compute_polynomials_from_dataframe` function.
    """

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        data = {
            'x': np.array([0, 1, 2, 3, 4]),
            'y_linear': np.array([1, 3, 5, 7, 9]),  # y = 2x + 1
            'y_quadratic': np.array([1, 2, 5, 10, 17]),  # y = x^2 + 1
            'y_noisy': np.array([1.1, 2.9, 5.2, 7.3, 8.8])
        }
        return pd.DataFrame(data)

    def test_successful_linear_fit(self, sample_df):
        """
        Tests a perfect linear fit (degree 1).
        The polynomial should accurately reproduce the original y-values.
        """
        result = _compute_polynomials_from_dataframe(
            df=sample_df,
            col_name_x='x',
            list_col_names_y=['y_linear'],
            degree=1
        )

        assert 'y_linear' in result
        assert 'y_linear_r2' in result
        
        poly = result['y_linear']
        # Check that the polynomial evaluates correctly on the original data
        expected_y = sample_df['y_linear']
        actual_y = poly(sample_df['x'])
        np.testing.assert_allclose(actual_y, expected_y)
        
        # R^2 for a perfect fit should be 1.0
        assert result['y_linear_r2'] == pytest.approx(1.0)

    def test_successful_quadratic_fit(self, sample_df):
        """
        Tests a perfect quadratic fit (degree 2).
        The polynomial should accurately reproduce the original y-values.
        """
        result = _compute_polynomials_from_dataframe(
            df=sample_df,
            col_name_x='x',
            list_col_names_y=['y_quadratic'],
            degree=2
        )

        assert 'y_quadratic' in result
        assert 'y_quadratic_r2' in result

        poly = result['y_quadratic']
        # Check that the polynomial evaluates correctly on the original data
        expected_y = sample_df['y_quadratic']
        actual_y = poly(sample_df['x'])
        np.testing.assert_allclose(actual_y, expected_y)

        assert result['y_quadratic_r2'] == pytest.approx(1.0)

    def test_multiple_columns(self, sample_df):
        """
        Tests if the function correctly processes multiple y-columns in one call.
        """
        result = _compute_polynomials_from_dataframe(
            df=sample_df,
            col_name_x='x',
            list_col_names_y=['y_linear', 'y_noisy'],
            degree=1
        )
        assert 'y_linear' in result and 'y_linear_r2' in result
        assert 'y_noisy' in result and 'y_noisy_r2' in result
        assert isinstance(result['y_linear'], np.polynomial.Polynomial)
        assert isinstance(result['y_noisy_r2'], float)
        assert result['y_noisy_r2'] < 1.0 # Noisy data should not have a perfect fit

    def test_handles_nan_values(self):
        """
        Tests if NaN values in y-columns are correctly ignored.
        The fit should be computed on the remaining clean data.
        """
        data = {
            'time': [1, 2, 3, 4, 5],
            'signal': [10, 20, np.nan, 40, 50]
        }
        df = pd.DataFrame(data)
        result = _compute_polynomials_from_dataframe(df, 'time', ['signal'], 1)

        poly = result['signal']
        # Check that the polynomial evaluates correctly on the non-NaN original data
        clean_df = df.dropna()
        expected_y = clean_df['signal']
        actual_y = poly(clean_df['time'])
        np.testing.assert_allclose(actual_y, expected_y)
        assert result['signal_r2'] == pytest.approx(1.0)

    def test_raises_error_on_invalid_df_type(self):
        """Tests ValueError for non-DataFrame input."""
        with pytest.raises(ValueError, match="df must be a Pandas DataFrame"):
            _compute_polynomials_from_dataframe([1, 2, 3], 'x', ['y'], 1)

    def test_raises_error_on_empty_df(self):
        """Tests ValueError for an empty DataFrame."""
        with pytest.raises(ValueError, match="df cannot be empty"):
            _compute_polynomials_from_dataframe(pd.DataFrame(), 'x', ['y'], 1)
            
    def test_raises_error_on_missing_x_column(self, sample_df):
        """Tests ValueError when col_name_x is not in the DataFrame."""
        with pytest.raises(ValueError, match="col_name_x 'missing_col' not found"):
            _compute_polynomials_from_dataframe(sample_df, 'missing_col', ['y_linear'], 1)

    @pytest.mark.parametrize("invalid_degree", [-1, 1.5, "2"])
    def test_raises_error_on_invalid_degree(self, sample_df, invalid_degree):
        """Tests ValueError for invalid degree types or values."""
        with pytest.raises(ValueError, match="degree must be a non-negative integer"):
            _compute_polynomials_from_dataframe(sample_df, 'x', ['y_linear'], invalid_degree)

    def test_raises_error_when_degree_is_too_high(self, sample_df):
        """
        Tests ValueError when degree is >= number of data points.
        """
        with pytest.raises(ValueError, match="number of data points must be greater than degree"):
            # 5 data points, so degree 5 should fail
            _compute_polynomials_from_dataframe(sample_df, 'x', ['y_linear'], 5)


class TestRSquared:
    """
    Test suite for the `_r_squared` function.
    """

    def test__r_squared_no_fit(self):
        """
        Tests if R^2 is 0.0 when the model is no better than the mean.
        If the prediction is just the mean of the actual values, RSS equals TSS,
        so R^2 should be 0.
        """
        y_actual = np.array([1, 2, 3, 4, 5])
        # The mean of y_actual is 3.0
        y_predicted = np.array([3, 3, 3, 3, 3])
        assert _r_squared(y_actual, y_predicted) == pytest.approx(0.0)

    def test__r_squared_typical_case(self):
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
        assert _r_squared(y_actual, y_predicted) == pytest.approx(0.984)

    def test__r_squared_negative_value(self):
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
        assert _r_squared(y_actual, y_predicted) == pytest.approx(-23.0)

    def test__r_squared_zero_tss(self):
        """
        Tests the edge case where the total sum of squares (TSS) is zero.
        This occurs when all actual values are the same. The function should not
        raise a division-by-zero error.
        """
        y_actual = np.array([5, 5, 5, 5, 5])
        y_predicted_perfect = np.array([5, 5, 5, 5, 5])
        y_predicted_imperfect = np.array([5.1, 4.9, 5.0, 5.1, 4.9])
        
        # If prediction is also perfect, R^2 is conventionally 1.0
        assert _r_squared(y_actual, y_predicted_perfect) == pytest.approx(1.0)
        # If prediction is not perfect, R^2 is conventionally undefined or poor, often returned as 0.0 in this implementation
        assert _r_squared(y_actual, y_predicted_imperfect) == pytest.approx(0.0)