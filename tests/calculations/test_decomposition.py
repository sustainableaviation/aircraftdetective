import pytest
import math
import pandas as pd

from aircraftdetective.calculations.decomposition import (
    _compute_lmdi_factor_contributions,
    _compute_improvement_metrics
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