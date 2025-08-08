from aircraftdetective import ureg
import pytest
import pint
import math

from aircraftdetective.calculations.aerodynamics import (
    compute_lift_to_drag_ratio,
    compute_aspect_ratio
)

def test_compute_lift_to_drag_ratio_typical_values():
    """
    Tests the L/D calculation with typical values for a modern narrow-body airliner.
    
    Data is loosely based on an Airbus A320neo.
    - Range (R): 6,100 km
    - Beta (β): 0.04 (a typical correction factor)
    - MTOW: 78,000 kg
    - MZFW: 62,500 kg
    - Cruise Speed (v_cruise): 830 km/h
    - TSFC at cruise: 17 g/kNs (a typical value for a modern turbofan)
    """
    # --- Input Data ---
    R = 6100 * ureg.kilometer
    beta = 0.04 * ureg.dimensionless
    MTOW = 78000 * ureg.kilogram
    MZFW = 62500 * ureg.kilogram
    v_cruise = 830 * ureg.kilometer_per_hour
    # g/kNs is equivalent to 1e-6 s/m, which has dimensions of [time]/[length]
    TSFC_cruise = 17 * ureg.gram / (ureg.kilonewton * ureg.second)

    # --- Expected Result Calculation ---
    # Expected L/D is around 24.4 for these values
    expected_ld_ratio = 24.4

    # --- Function Call & Assertion ---
    ld_ratio = compute_lift_to_drag_ratio(R, beta, MTOW, MZFW, v_cruise, TSFC_cruise)

    assert ld_ratio.magnitude == pytest.approx(expected_ld_ratio, rel=1e-2)
    assert ld_ratio.dimensionless


def test_compute_lift_to_drag_ratio_different_units():
    """
    Tests the L/D calculation with different but compatible units (Imperial/Nautical).
    The numerical result should be identical to the SI unit test.
    """
    # --- Input Data in Imperial/Nautical Units ---
    R = 3294 * ureg.nautical_mile  # ~6100 km
    beta = 0.04 * ureg.dimensionless
    MTOW = 171961 * ureg.pound      # ~78000 kg
    MZFW = 137789 * ureg.pound      # ~62500 kg
    v_cruise = 448 * ureg.knot      # ~830 km/h
    # This TSFC value is equivalent to ~17 g/kNs
    TSFC_cruise = 0.599 * ureg.pound / (ureg.pound_force * ureg.hour)

    # --- Expected Result ---
    expected_ld_ratio = 24.4

    # --- Function Call & Assertion ---
    ld_ratio = compute_lift_to_drag_ratio(R, beta, MTOW, MZFW, v_cruise, TSFC_cruise)

    assert ld_ratio.magnitude == pytest.approx(expected_ld_ratio, rel=1e-2)
    assert ld_ratio.dimensionless


def test_compute_lift_to_drag_ratio_wrong_units():
    """
    Tests that a DimensionalityError is raised if inputs have incorrect dimensions.
    """
    with pytest.raises(pint.errors.DimensionalityError):
        compute_lift_to_drag_ratio(
            R=1000 * ureg.kilogram,  # Incorrect: should be [length]
            beta=0.04 * ureg.dimensionless,
            MTOW=78000 * ureg.kilogram,
            MZFW=62500 * ureg.kilogram,
            v_cruise=830 * ureg.kilometer_per_hour,
            TSFC_cruise=17 * ureg.gram / (ureg.kilonewton * ureg.second)
        )


def test_compute_aspect_ratio_typical_values():
    """
    Tests the aspect ratio calculation with typical values for an Airbus A320.
    - Wingspan (b): 35.8 m
    - Wing Area (S): 122.6 m^2
    """
    # --- Input Data ---
    wingspan = 35.8 * ureg.meter
    wing_area = 122.6 * ureg.meter**2

    # --- Expected Result ---
    # A = b^2 / S = 35.8^2 / 122.6
    expected_aspect_ratio = 10.45

    # --- Function Call & Assertion ---
    aspect_ratio = compute_aspect_ratio(wingspan, wing_area)

    assert aspect_ratio.magnitude == pytest.approx(expected_aspect_ratio, rel=1e-3)
    assert aspect_ratio.dimensionless


def test_compute_aspect_ratio_different_units():
    """
    Tests the aspect ratio calculation with Imperial units.
    The numerical result should be identical.
    """
    # --- Input Data in Feet ---
    wingspan = 117.45 * ureg.foot  # ~35.8 m
    wing_area = 1319.65 * ureg.foot**2  # ~122.6 m^2

    # --- Expected Result ---
    expected_aspect_ratio = 10.45

    # --- Function Call & Assertion ---
    aspect_ratio = compute_aspect_ratio(wingspan, wing_area)

    assert aspect_ratio.magnitude == pytest.approx(expected_aspect_ratio, rel=1e-3)
    assert aspect_ratio.dimensionless


def test_compute_aspect_ratio_wrong_units():
    """
    Tests that a DimensionalityError is raised if inputs have incorrect dimensions.
    """
    with pytest.raises(pint.errors.DimensionalityError):
        compute_aspect_ratio(
            b=35.8 * ureg.meter,
            S=122.6 * ureg.meter  # Incorrect: should be [area]
        )