def compute_aerodynamic_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    
    beta_widebody = 0.04
    beta_narrowbody = 0.06

    df["L/D"] = df.apply(
        lambda row: compute_lift_to_drag_ratio(
            R=row["Payload/Range: Range at Point A"],
            beta=beta_widebody if row["Type"] == "Wide" else beta_narrowbody,
            MTOW=row["Payload/Range: MTOW"],
            MZFW=row["Payload/Range: ZFW at Point A"],
            v_cruise=row["Cruise Speed"],
            TSFC_cruise=row["TSFC (cruise)"]
        ),
        axis=1
    ).astype('pint[dimensionless]')
    df["Aspect Ratio"] = df.apply(
        lambda row: compute_aspect_ratio(
            b=row["Wingspan"],
            S=row["Wing Area"]
        ),
        axis=1
    ).pint.convert_object_dtype()
    return df