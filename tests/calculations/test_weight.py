import pandas as pd
import pytest
import numpy as np
from aircraftdetective import ureg
import pandas.testing as pd_testing

from aircraftdetective.calculations.weight import calculate_weight_metrics

class TestCalculateWeightMetrics:
    """
    Test suite for the `calculate_weight_metrics` function.
    """

    @pytest.fixture
    def base_weight_data(self) -> pd.DataFrame:
        """
        Provides a DataFrame with the necessary columns for weight calculations.
        
        - Narrow, 2000: OEW/Pax = 30000 / 150 = 200
        - Narrow, 2010: OEW/Pax = 40000 / 100 = 400 (Max for Narrow)
        - Wide, 2005:   OEW/Pax = 80000 / 400 = 200 (Max for Wide)
        - Wide, 2015:   OEW/Pax = 90000 / 500 = 180
        """
        data = {
            'YOI': [2000, 2010, 2005, 2015],
            'Type': ['Narrow', 'Narrow', 'Wide', 'Wide'],
            'OEW': [30000, 40000, 80000, 90000],
            'MTOW': [60000, 70000, 160000, 180000],
            'Pax Exit Limit': [150, 100, 400, 500],
            'Other_Data': ['A', 'B', 'C', 'D']
        }
        return pd.DataFrame(data)

    def test_basic_calculations(self, base_weight_data):
        """
        Tests the successful calculation of all three new metrics.
        """
        result_df = calculate_weight_metrics(base_weight_data)

        # Expected simple ratios
        expected_oew_mtow = [0.5, 4/7, 0.5, 0.5]
        expected_oew_pax = [200.0, 400.0, 200.0, 180.0]
        
        # Expected normalized ratios
        # Narrow: max is 400. [200/400, 400/400] = [0.5, 1.0]
        # Wide: max is 200. [200/200, 180/200] = [1.0, 0.9]
        # Original order: [Narrow, Narrow, Wide, Wide]
        expected_norm_oew_pax = [0.5, 1.0, 1.0, 0.9]

        assert 'OEW/MTOW' in result_df.columns
        assert 'OEW/Exit Limit' in result_df.columns
        assert 'norm(OEW/Exit Limit)' in result_df.columns
        assert 'Other_Data' in result_df.columns # Check preservation

        pd_testing.assert_series_equal(
            result_df['OEW/MTOW'],
            pd.Series(expected_oew_mtow, name='OEW/MTOW'),
            atol=1e-5
        )
        pd_testing.assert_series_equal(
            result_df['OEW/Exit Limit'],
            pd.Series(expected_oew_pax, name='OEW/Exit Limit'),
            atol=1e-5
        )
        pd_testing.assert_series_equal(
            result_df['norm(OEW/Exit Limit)'],
            pd.Series(expected_norm_oew_pax, name='norm(OEW/Exit Limit)'),
            atol=1e-5
        )
        # Ensure other columns are preserved
        assert (result_df['Other_Data'] == base_weight_data['Other_Data']).all()

    def test_immutability(self, base_weight_data):
        """
        Ensures the original input DataFrame is not modified.
        """
        original_copy = base_weight_data.copy()
        calculate_weight_metrics(base_weight_data)
        pd_testing.assert_frame_equal(base_weight_data, original_copy)

    @pytest.mark.parametrize(
        "col_to_drop, expected_msg",
        [
            ('OEW', "Missing required column: OEW"),
            ('MTOW', "Missing required column: MTOW"),
            ('Pax Exit Limit', "Missing required column: Pax Exit Limit"),
        ]
    )
    def test_raises_value_error_missing_columns(self, base_weight_data, col_to_drop, expected_msg):
        """
        Tests that a ValueError is raised if any required column is missing.
        """
        invalid_df = base_weight_data.drop(columns=[col_to_drop])
        with pytest.raises(ValueError, match=expected_msg):
            calculate_weight_metrics(invalid_df)

    def test_nan_propagation(self):
        """
        Tests that NaN in input columns propagates correctly to output columns.
        """
        data = {
            'Type': ['A', 'A'],
            'OEW': [100, np.nan],
            'MTOW': [np.nan, 200],
            'Pax Exit Limit': [50, 50],
        }
        df = pd.DataFrame(data)
        result = calculate_weight_metrics(df)

        # Row 0: OEW/MTOW -> 100/nan = nan
        assert np.isnan(result.iloc[0]['OEW/MTOW'])
        # Row 0: OEW/Pax -> 100/50 = 2.0
        assert result.iloc[0]['OEW/Exit Limit'] == 2.0

        # Row 1: OEW/MTOW -> nan/200 = nan
        assert np.isnan(result.iloc[1]['OEW/MTOW'])
        # Row 1: OEW/Pax -> nan/50 = nan
        assert np.isnan(result.iloc[1]['OEW/Exit Limit'])

        # Group A: OEW/Pax is [2.0, nan]. Max is 2.0.
        # Row 0 norm: 2.0 / 2.0 = 1.0
        assert result.iloc[0]['norm(OEW/Exit Limit)'] == 1.0
        # Row 1 norm: nan / 2.0 = nan
        assert np.isnan(result.iloc[1]['norm(OEW/Exit Limit)'])