#Linear regression for all aircraft to see how overall structural efficiency has increased.
    x_large = large_aircrafts['YOI'].astype(np.int64)
    y_large = large_aircrafts['OEW/Exit Limit'].astype(np.float64)
    z_large = np.polyfit(x_large,  y_large, 1)
    p_large = np.poly1d(z_large)
    x_medium = medium_aircrafts['YOI'].astype(np.int64)
    y_medium = medium_aircrafts['OEW/Exit Limit'].astype(np.float64)
    z_medium = np.polyfit(x_medium,  y_medium, 1)
    p_medium = np.poly1d(z_medium)