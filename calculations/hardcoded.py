# %%


def estimate_a380_overall_efficiency():
    boeing747 = airplanes_release_year.loc[airplanes_release_year['Description']=='Boeing 747-400', 'MJ/ASK'].iloc[0]
    a380 = {'Description': 'A380', 'MJ/ASK': 0.88*boeing747}