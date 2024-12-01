# %%



import matplotlib.pyplot as plt



fig = plt.figure(dpi=300)
plt.scatter(
    x=df_engines_grouped['TSFC(takeoff) [g/kNs]'],
    y=df_engines_grouped['TSFC(cruise) [g/kNs]'],
    color='black',
    label='Turbofan Engines',
)
plt.plot(
    np.arange(8, 19, 0.2),
    reg(np.arange(8, 19, 0.2)),
    color='blue',
    label='Linear Regression',
    linewidth=2,
)
plt.plot(
    np.arange(8, 19, 0.2),
    poly(np.arange(8, 19, 0.2)),
    color='red',
    label='Second Order Regression',
    linewidth=2,
)