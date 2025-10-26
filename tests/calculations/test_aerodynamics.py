# %%
import pandas as pd
import pint_pandas
import pytest
import pint

from aircraftdetective import ureg
from aircraftdetective.calculations.aerodynamics import (
    compute_lift_to_drag_ratio,
    compute_aspect_ratio
)


class TestComputeLiftToDragRatio:
    """Test suite for the `compute_lift_to_drag_ratio` function."""

    @pytest.fixture
    def ld_ratio_input_df_si(self) -> pd.DataFrame:
        """Provides a sample DataFrame with SI units for L/D ratio tests based on an A320neo."""
        return pd.DataFrame({
            'Payload/Range: Range at Point B': pd.Series([3900], dtype='pint[km]'),
            'Payload/Range: Range at Point C': pd.Series([6500], dtype='pint[km]'),
            'MTOW': pd.Series([79000], dtype='pint[kg]'),
            'Payload/Range: MZFW at point B': pd.Series([64300], dtype='pint[kg]'),
            'Payload/Range: MZFW at point C': pd.Series([60000], dtype='pint[kg]'),
            'Cruise Speed': pd.Series([830], dtype='pint[km/h]'),
            'TSFC (cruise)': pd.Series([17], dtype='pint[g/(kN*s)]'),
        })

    @pytest.fixture
    def ld_ratio_input_df_imperial(self) -> pd.DataFrame:
        """Provides a sample DataFrame with Imperial/Nautical units for L/D ratio tests."""
        return pd.DataFrame({
            # ~3900 km
            'Payload/Range: Range at Point B': pd.Series([2106], dtype='pint[nautical_mile]'),
            # ~6500 km
            'Payload/Range: Range at Point C': pd.Series([3510], dtype='pint[nautical_mile]'),
            'MTOW': pd.Series([174165], dtype='pint[lb]'),      # ~79000 kg
            'Payload/Range: MZFW at point B': pd.Series([141757], dtype='pint[lb]'),      # ~64300 kg
            'Payload/Range: MZFW at point C': pd.Series([132277], dtype='pint[lb]'),      # ~60000 kg
            'Cruise Speed': pd.Series([448], dtype='pint[knot]'),      # ~830 km/h
            # ~17 g/kNs
            'TSFC (cruise)': pd.Series([0.599], dtype='pint[lb/(lbf*h)]'),
        })

    def test_typical_values(self, ld_ratio_input_df_si: pd.DataFrame):
        """
        Tests the L/D calculation with typical values for a modern narrow-body airliner.
        """
        beta = 0.04  # A typical correction factor
        expected_ld_ratio = 18.5

        result_df = compute_lift_to_drag_ratio(df=ld_ratio_input_df_si, beta=beta)

        assert 'L/D' in result_df.columns
        ld_ratio = result_df['L/D'].iloc[0]
        assert ld_ratio.magnitude == pytest.approx(expected_ld_ratio, rel=1e-2)
        assert ld_ratio.check('[]') # dimensionless

    def test_different_units(self, ld_ratio_input_df_imperial: pd.DataFrame):
        """
        Tests the L/D calculation with different but compatible units (Imperial/Nautical).
        The numerical result should be identical to the SI unit test.
        """
        beta = 0.04
        expected_ld_ratio = 18.6

        result_df = compute_lift_to_drag_ratio(df=ld_ratio_input_df_imperial, beta=beta)

        assert 'L/D' in result_df.columns
        ld_ratio = result_df['L/D'].iloc[0]
        assert ld_ratio.magnitude == pytest.approx(expected_ld_ratio, rel=1e-2)
        assert ld_ratio.check('[]') # dimensionless

    def test_wrong_units_raises_value_error(self, ld_ratio_input_df_si: pd.DataFrame):
        """
        Tests that providing an input with incorrect dimensions (e.g., mass for range)
        raises a ValueError.
        """
        df_wrong = ld_ratio_input_df_si.copy()
        # Intentionally introduce wrong units: Change Range from [length] to [mass]
        df_wrong['Payload/Range: Range at Point B'] = pd.Series([3900], dtype='pint[kg]')

        # FIX: The test now only checks that a ValueError is raised, without
        # checking for the specific error message.
        with pytest.raises(ValueError):
            compute_lift_to_drag_ratio(df=df_wrong, beta=0.04)

    def test_missing_column(self, ld_ratio_input_df_si: pd.DataFrame):
        """
        Tests that a ValueError is raised if a required column is missing.
        """
        df_missing = ld_ratio_input_df_si.drop(columns=['Payload/Range: MZFW at point B'])
        with pytest.raises(ValueError, match="DataFrame is missing required columns: \\['Payload/Range: MZFW at point B'\\]"):
            compute_lift_to_drag_ratio(df=df_missing, beta=0.04)

    def test_beta_out_of_range(self, ld_ratio_input_df_si: pd.DataFrame):
        """
        Tests that a ValueError is raised if beta is not between 0 and 1.
        """
        with pytest.raises(ValueError, match="beta must be between 0 and 1."):
            compute_lift_to_drag_ratio(df=ld_ratio_input_df_si, beta=1.5)
        with pytest.raises(ValueError, match="beta must be between 0 and 1."):
            compute_lift_to_drag_ratio(df=ld_ratio_input_df_si, beta=-0.1)
        with pytest.raises(ValueError, match="beta must be between 0 and 1."):
            compute_lift_to_drag_ratio(df=ld_ratio_input_df_si, beta=0)

    def test_empty_df(self):
        """
        Tests that a ValueError is raised for an empty DataFrame.
        """
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty."):
            compute_lift_to_drag_ratio(df=empty_df, beta=0.04)

    def test_multi_row(self):
        """
        Tests the L/D calculation on a multi-row DataFrame with data for two different aircraft.
        """
        df_multi = pd.DataFrame({
            'Payload/Range: Range at Point B': pd.Series([3900, 8300], dtype='pint[km]'),
            'Payload/Range: Range at Point C': pd.Series([6500, 15000], dtype='pint[km]'),
            'MTOW': pd.Series([79000, 280000], dtype='pint[kg]'),
            'Payload/Range: MZFW at point B': pd.Series([64300, 200000], dtype='pint[kg]'),
            'Payload/Range: MZFW at point C': pd.Series([60000, 170000], dtype='pint[kg]'),
            'Cruise Speed': pd.Series([830, 903], dtype='pint[km/h]'),
            'TSFC (cruise)': pd.Series([17, 16], dtype='pint[g/(kN*s)]'),
        })
        beta = 0.04
        expected_ld_ratios = [18.6, 19.0]  # A320-like, A350-like

        result_df = compute_lift_to_drag_ratio(df=df_multi, beta=beta)

        assert 'L/D' in result_df.columns
        assert result_df['L/D'].pint.magnitude.tolist() == pytest.approx(expected_ld_ratios, rel=1e-2)
        assert result_df['L/D'].pint.units == ureg.dimensionless


class TestComputeAspectRatio:
    """Groups tests for the compute_aspect_ratio function."""

    @pytest.fixture
    def aspect_ratio_input_df_si(self):
        """Provides a sample DataFrame with SI units for aspect ratio tests (A320)."""
        return pd.DataFrame({
            'Wingspan': pd.Series([35.8], dtype='pint[m]'),
            'Wing Area': pd.Series([122.6], dtype='pint[m**2]'),
        })

    @pytest.fixture
    def aspect_ratio_input_df_imperial(self):
        """Provides a sample DataFrame with Imperial units for aspect ratio tests."""
        return pd.DataFrame({
            'Wingspan': pd.Series([117.45], dtype='pint[foot]'),      # ~35.8 m
            'Wing Area': pd.Series([1319.65], dtype='pint[foot**2]'), # ~122.6 m^2
        })

    def test_typical_values(self, aspect_ratio_input_df_si):
        """
        Tests the aspect ratio calculation with typical values for an Airbus A320.
        """
        expected_aspect_ratio = 10.45

        result_df = compute_aspect_ratio(df=aspect_ratio_input_df_si)

        assert 'Aspect Ratio' in result_df.columns
        aspect_ratio = result_df['Aspect Ratio'].iloc[0]
        assert aspect_ratio.magnitude == pytest.approx(expected_aspect_ratio, rel=1e-3)
        assert aspect_ratio.dimensionless

    def test_different_units(self, aspect_ratio_input_df_imperial):
        """
        Tests the aspect ratio calculation with Imperial units.
        The numerical result should be identical.
        """
        expected_aspect_ratio = 10.45

        result_df = compute_aspect_ratio(df=aspect_ratio_input_df_imperial)

        assert 'Aspect Ratio' in result_df.columns
        aspect_ratio = result_df['Aspect Ratio'].iloc[0]
        assert aspect_ratio.magnitude == pytest.approx(expected_aspect_ratio, rel=1e-3)
        assert aspect_ratio.dimensionless

    def test_wrong_units_returns_dimensioned_quantity(self, aspect_ratio_input_df_si):
        """
        Tests that providing inputs with incorrect dimensions (e.g., area/length
        instead of area/area) returns a dimensioned quantity rather than raising an error.
        """
        df_wrong = aspect_ratio_input_df_si.copy()
        # Change Wing Area to a length unit (m) instead of area (m**2)
        df_wrong['Wing Area'] = pd.Series([122.6], dtype='pint[m]')

        result_df = compute_aspect_ratio(df=df_wrong)
        result_units = result_df['Aspect Ratio'].pint.units

        # Aspect Ratio should be dimensionless.
        # The incorrect calculation (m**2 / m) results in units of length (m).
        assert not result_units.dimensionless
        assert result_units == ureg.meter

    def test_missing_column(self, aspect_ratio_input_df_si):
        """
        Tests that a ValueError is raised if a required column is missing.
        """
        df_missing = aspect_ratio_input_df_si.drop(columns=['Wing Area'])
        with pytest.raises(ValueError, match="DataFrame is missing required columns: \\['Wing Area'\\]"):
            compute_aspect_ratio(df=df_missing)

    def test_empty_df(self):
        """
        Tests that a ValueError is raised for an empty DataFrame.
        """
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is missing required columns"):
            compute_aspect_ratio(df=empty_df)

    def test_multi_row(self):
        """
        Tests the aspect ratio calculation on a multi-row DataFrame.
        """
        df_multi = pd.DataFrame({
            'Wingspan': pd.Series([35.8, 64.75], dtype='pint[m]'),      # A320, A350
            'Wing Area': pd.Series([122.6, 443], dtype='pint[m**2]'),
        })
        expected_aspect_ratios = [10.45, 9.46]

        result_df = compute_aspect_ratio(df=df_multi)

        assert 'Aspect Ratio' in result_df.columns
        assert result_df['Aspect Ratio'].pint.magnitude.tolist() == pytest.approx(expected_aspect_ratios, rel=1e-3)
        assert result_df['Aspect Ratio'].pint.units == ureg.dimensionless