import pytest
import math
import pandas as pd

from aircraftdetective.calculations.decomposition import (
    _compute_lmdi_factor_contributions,
    _compute_efficiency_improvement_metrics
)


@pytest.fixture
def sample_aircraft_data() -> pd.DataFrame:
    """Provides a standard, well-formed DataFrame for testing.
    
    | YOI | Type   | EU  | TSFC | OEW/Exit Limit | L/D | Other_Data |
    |-----|--------|-----|------|----------------|-----|------------|
    | 2005| Wide   | 100 | 0.8  | 320            | 15  | A          |
    | 2000| Narrow | 120 | 1.0  | 250            | 12  | B          |
    | 2010| Narrow | 108 | 0.9  | 240            | 13  | C          |
    | 2015| Wide   | 80  | 0.6  | 300            | 18  | D          |
    | 2008| Wide   | 90  | 0.7  | 310            | 16  | E          |
    """
    data = {
        'YOI':              [2005, 2000, 2010, 2015, 2008],
        'Type':             ['Wide', 'Narrow', 'Narrow', 'Wide', 'Wide'],
        'EU':               [100,   120,    108,    80,     90],
        'TSFC':             [0.8,   1.0,    0.9,    0.6,    0.7],
        'OEW/Exit Limit':   [320,   250,    240,    300,    310],
        'L/D':              [15,    12,     13,     18,     16],
        'Other_Data':       ['A', 'B', 'C', 'D', 'E'] # To ensure other columns are preserved
    }
    return pd.DataFrame(data)


class TestComputeEfficiencyImprovementMetrics:
    """Test suite for the `_compute_efficiency_improvement_metrics` function."""

    def test_successful_computation_mixed_types(self, sample_aircraft_data):
        """
        Tests the primary success case with mixed aircraft types and unordered years.
        Verifies that grouping, sorting, and calculations are correct.

        Expected output DataFrame (sorted by YOI within Type):

        | YOI  | Type   | Delta%(EU) | Delta%(TSFC) | Delta%(OEW/Exit Limit) | Delta%(L/D) | Other_Data |
        |------|--------|------------|--------------|------------------------|-------------|------------|
        | 2000 | Narrow | 0.0        | 0.0          | 0.0                    | 0.0         | B          |
        | 2005 | Wide   | 0.0        | 0.0          | 0.0                    | 0.0         | A          |
        | 2008 | Wide   | 11.111111  | 14.285714    | 3.225806               | 6.666667    | E          |
        | 2010 | Narrow | 11.111111  | 11.111111    | 4.166667               | 8.333333    | C          |
        | 2015 | Wide   | 25.0       | 33.333333    | 6.666667               | 20.0        | D          |

        Manual calculations:

        - For Narrow type (baseline YOI 2000):
          - 2000: Baseline, all deltas = 0.0
          - 2010:
            - Delta%(EU) = ((120 / 108) - 1) * 100 = 11.111111%
            - Delta%(TSFC) = ((1.0 / 0.9) - 1) * 100 = 11.111111%
            - Delta%(OEW/Exit Limit) = ((250 / 240) - 1) * 100 = 4.166667%
            - Delta%(L/D) = ((13 / 12) - 1) * 100 = 8.333333
        - For Wide type (baseline YOI 2005):
          - 2005: Baseline, all deltas = 0.0
          - 2008:
            - Delta%(EU) = ((100 / 90) - 1) * 100 = 11.111111%
            - Delta%(TSFC) = ((0.8 / 0.7) - 1) * 100 = 14.285714%
            - Delta%(OEW/Exit Limit) = ((320 / 310) - 1) * 100 = 3.225806%
            - Delta%(L/D) = ((16 / 15) - 1) * 100 = 6.666667
          - 2015:
            - Delta%(EU) = ((100 / 80) - 1) * 100 = 25.0%
            - Delta%(TSFC) = ((0.8 / 0.6) - 1) * 100 = 33.333333%
            - Delta%(OEW/Exit Limit) = ((320 / 300) - 1) * 100 = 6.666667%
            - Delta%(L/D) = ((18 / 15) - 1) * 100 = 20.0
        """        
        expected_data = {
            'YOI':              [2000, 2005, 2008, 2010, 2015], # Sorted
            'Type':             ['Narrow', 'Wide', 'Wide', 'Narrow', 'Wide'],
            'Delta%(EU)':       [0.0, 0.0, 11.111111, 11.111111, 25.0],
            'Delta%(TSFC)':     [0.0, 0.0, 14.285714, 11.111111, 33.333333],
            'Delta%(OEW/Exit Limit)': [0.0, 0.0, 3.225806, 4.166667, 6.666667],
            'Delta%(L/D)':      [0.0, 0.0, 6.666667, 8.333333, 20.0],
            'Other_Data':       ['B', 'A', 'E', 'C', 'D']
        }
        expected_df = pd.DataFrame(expected_data)

        result_df = _compute_efficiency_improvement_metrics(sample_aircraft_data)

        expected_df = expected_df.sort_values(by=['Type', 'YOI']).reset_index(drop=True)
        result_df = result_df.sort_values(by=['Type', 'YOI']).reset_index(drop=True)

        pd.testing.assert_frame_equal(result_df, expected_df, atol=1e-6)

    def test_edge_case_single_entry_per_group(self):
        """Tests that a DataFrame with one row per group results in all zeros."""
        data = {
            'YOI': [2000, 2010],
            'Type': ['Narrow', 'Wide'],
            'EU': [100, 120], 'TSFC': [1.0, 0.9],
            'OEW/Exit Limit': [250, 300], 'L/D': [12, 15]
        }
        df = pd.DataFrame(data)
        result_df = _compute_efficiency_improvement_metrics(df)

        for col in result_df.columns:
            if col.startswith('Delta%'):
                assert (result_df[col] == 0.0).all()

    def test_raises_error_on_empty_dataframe(self):
        """Verifies that an empty DataFrame raises a ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty Pandas DataFrame"):
            _compute_efficiency_improvement_metrics(pd.DataFrame())

    def test_raises_error_on_none_input(self):
        """Verifies that a None input raises a ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty Pandas DataFrame"):
            _compute_efficiency_improvement_metrics(None)

    def test_raises_error_on_missing_column(self, sample_aircraft_data):
        """Verifies that a missing required column raises a ValueError."""
        df = sample_aircraft_data.drop(columns=['EU'])
        with pytest.raises(ValueError, match="Required column 'EU' not found"):
            _compute_efficiency_improvement_metrics(df)

    def test_raises_error_on_all_nan_column(self, sample_aircraft_data):
        """Verifies that a column with all NaN values raises a ValueError."""
        df = sample_aircraft_data.copy()
        # FIX: Use pandas native NA to represent missing values and avoid NameError
        df['L/D'] = pd.NA
        with pytest.raises(ValueError, match="Column 'L/D' cannot be all NaN"):
            _compute_efficiency_improvement_metrics(df)

    def test_raises_error_on_non_numeric_column(self, sample_aircraft_data):
        """Verifies that a non-numeric metric column raises a ValueError."""
        df = sample_aircraft_data.copy()
        df['TSFC'] = df['TSFC'].astype(str)
        with pytest.raises(ValueError, match="Column 'TSFC' must be of a numeric type"):
            _compute_efficiency_improvement_metrics(df)


class TestLmdiFactorContributions:
    """
    Test suite for the `_compute_lmdi_factor_contributions` function.
    """

    @pytest.mark.parametrize(
        "aggregate_t1, aggregate_t2, factor_t1, factor_t2, expected_contribution",
        [
            # Based on numerical example from Ang & Zhang (2000), Table 5.
            # The total change in the aggregate (50 - 20 = 30) is fully explained
            # by the change in the first factor.
            pytest.param(20.0, 50.0, 2.0, 5.0, 30.0, id="increase_with_changing_factor"),
            
            # When the factor itself does not change, its contribution must be zero.
            pytest.param(20.0, 50.0, 10.0, 10.0, 0.0, id="increase_with_constant_factor"),
            
            # This is the reverse of the first case. The total change is -30,
            # which should be fully explained by the factor's decrease.
            pytest.param(50.0, 20.0, 5.0, 2.0, -30.0, id="decrease_with_changing_factor"),
            
            # A more general case without round numbers.
            # L(150, 100) = (150-100) / ln(150/100) ≈ 123.25
            # Contribution = L * ln(4/2) ≈ 123.25 * ln(2) ≈ 85.45
            pytest.param(100.0, 150.0, 2.0, 4.0, (50 / math.log(1.5)) * math.log(2.0), id="general_increase_case"),
        ]
    )
    def test_valid_scenarios(
        self, aggregate_t1, aggregate_t2, factor_t1, factor_t2, expected_contribution
    ):
        """
        Tests the LMDI factor contribution calculation for various valid scenarios.
        """
        result = _compute_lmdi_factor_contributions(
            aggregate_t1, aggregate_t2, factor_t1, factor_t2
        )
        # Use pytest.approx for safe floating-point comparison
        assert result == pytest.approx(expected_contribution)

    def test_no_change_in_aggregate(self):
        """
        Tests the specific edge case where the aggregate value is unchanged.
        The function should return 0.0 immediately.
        """
        result = _compute_lmdi_factor_contributions(
            aggregate_t1=100.0, aggregate_t2=100.0, factor_t1=5.0, factor_t2=10.0
        )
        assert result == 0.0

    @pytest.mark.parametrize(
        "aggregate_t1, aggregate_t2, factor_t1, factor_t2, expected_error",
        [
            # Zero values that lead to undefined mathematical operations.
            pytest.param(10.0, 20.0, 0.0, 5.0, ZeroDivisionError, id="zero_factor_t1_division"),
            pytest.param(0.0, 20.0, 2.0, 5.0, ValueError, id="zero_aggregate_t1_log"),
            pytest.param(10.0, 0.0, 2.0, 5.0, ValueError, id="zero_aggregate_t2_log"),
            pytest.param(10.0, 20.0, 2.0, 0.0, ValueError, id="zero_factor_t2_log"),
            
            # Negative values are not permissible in logarithms.
            pytest.param(-10.0, 20.0, 2.0, 5.0, ValueError, id="negative_aggregate_t1"),
            pytest.param(10.0, -20.0, 2.0, 5.0, ValueError, id="negative_aggregate_t2"),
            pytest.param(10.0, 20.0, -2.0, 5.0, ValueError, id="negative_factor_ratio"),
        ]
    )
    def test_invalid_inputs(
        self, aggregate_t1, aggregate_t2, factor_t1, factor_t2, expected_error
    ):
        """
        Tests that the function raises appropriate errors for invalid inputs like zeros
        or negative numbers, which are mathematically undefined for this formula.
        """
        with pytest.raises(expected_error):
            _compute_lmdi_factor_contributions(
                aggregate_t1, aggregate_t2, factor_t1, factor_t2
            )