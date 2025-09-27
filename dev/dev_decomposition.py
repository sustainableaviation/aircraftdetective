import pandas as pd
from database.tools import plot
import matplotlib.pyplot as plt
import numpy as np


def calculate(savefig, folder_path):
    # Load Data and Filter Regional out
    data = pd.read_excel(r'Databank.xlsx')
    data = data.sort_values('YOI', ascending=True)
    data = data.loc[data['Type']!='Regional']
    data = data.drop(data.index[0])

    # Normalize Data for TSFC, L/D, OEW/Exit Limit and EU using 1959 as a Basis, for OEW normalize regarding heaviest value of each Type
    data['OEW/Exit Limit'] = data.groupby('Type')['OEW/Exit Limit'].transform(lambda x: x / x.max())
    data = data[['Name','YOI', 'TSFC Cruise','EU (MJ/ASK)', 'OEW/Exit Limit', 'L/D estimate']]
    data = data.dropna()
    data['OEW/Exit Limit'] = 100 / data['OEW/Exit Limit']
    oew = data.dropna(subset='OEW/Exit Limit')
    oew['OEW/Exit Limit'] = oew['OEW/Exit Limit'] - 100

    max_tsfc = data.loc[data['YOI']==1958, 'TSFC Cruise'].iloc[0]
    data['TSFC Cruise'] = 100 / (data['TSFC Cruise'] / max_tsfc)
    tsfc = data.dropna(subset='TSFC Cruise')
    tsfc['TSFC Cruise'] = tsfc['TSFC Cruise']-100

    max_eu = data.loc[data['YOI']==1958, 'EU (MJ/ASK)'].iloc[0]
    data['EU (MJ/ASK)'] = 100/ (data['EU (MJ/ASK)'] / max_eu)
    eu = data.dropna(subset='EU (MJ/ASK)')
    eu['EU (MJ/ASK)'] = eu['EU (MJ/ASK)'] - 100

    min_ld = data.loc[data['YOI']==1958, 'L/D estimate'].iloc[0]
    data['L/D estimate'] = 100 / (min_ld / data['L/D estimate'])
    ld = data.dropna(subset='L/D estimate')
    ld['L/D estimate'] = ld['L/D estimate'] - 100

    # Define a function to calculate R^2
    def calculate_r_squared(y_true, y_pred):
        y_mean = np.mean(y_true)
        ss_total = np.sum((y_true - y_mean) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        return r_squared

    # Polynomial for YOI vs. L/D estimate
    years = np.arange(1958, 2021)
    x_all = ld['YOI'].astype(np.int64)
    y_all = ld['L/D estimate'].astype(np.float64)
    z_all = np.polyfit(x_all, y_all, 4)
    p_all_ld = np.poly1d(z_all)

    y_pred_ld = p_all_ld(x_all)
    r2_ld = calculate_r_squared(y_all, y_pred_ld)
    #print("R^2 for YOI vs. L/D estimate:", r2_ld)

    # Polynomial for YOI vs. OEW/Exit Limit
    x_all = oew['YOI'].astype(np.int64)
    y_all = oew['OEW/Exit Limit'].astype(np.float64)
    z_all = np.polyfit(x_all, y_all, 4)
    p_all_oew = np.poly1d(z_all)

    # Calculate R^2 for YOI vs. OEW/Exit Limit
    y_pred_oew = p_all_oew(x_all)
    r2_oew = calculate_r_squared(y_all, y_pred_oew)
    #print("R^2 for YOI vs. OEW/Exit Limit:", r2_oew)

    # Polynomial for YOI vs. TSFC Cruise
    x_all = tsfc['YOI'].astype(np.int64)
    y_all = tsfc['TSFC Cruise'].astype(np.float64)
    z_all = np.polyfit(x_all, y_all, 4)
    p_all_tsfc = np.poly1d(z_all)

    # Calculate R^2 for YOI vs. TSFC Cruise
    y_pred_tsfc = p_all_tsfc(x_all)
    r2_tsfc = calculate_r_squared(y_all, y_pred_tsfc)
    #print("R^2 for YOI vs. TSFC Cruise:", r2_tsfc)

    # Polynomial for YOI vs. EU (MJ/ASK)
    x_all = eu['YOI'].astype(np.int64)
    y_all = eu['EU (MJ/ASK)'].astype(np.float64)
    z_all = np.polyfit(x_all, y_all, 4)
    p_all_eu = np.poly1d(z_all)

    # Calculate R^2 for YOI vs. EU (MJ/ASK)
    y_pred_eu = p_all_eu(x_all)
    r2_eu = calculate_r_squared(y_all, y_pred_eu)
    #print("R^2 for YOI vs. EU (MJ/ASK):", r2_eu)

    # Plot Scatterpoint for the Aircraft and the Polynomials
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    x_label = 'Aircraft Year of Introduction'
    y_label = 'Efficiency Improvements [\%]'

    ax.scatter(eu['YOI'], eu['EU (MJ/ASK)'],color='black', label='Overall (MJ/ASK)')
    ax.scatter(ld['YOI'], ld['L/D estimate'],color='royalblue', label='Aerodynamic (L/D)')
    ax.scatter(oew['YOI'], oew['OEW/Exit Limit'],color='steelblue', label='Structural (OEW/PEL)')
    ax.scatter(tsfc['YOI'], tsfc['TSFC Cruise'],color='lightblue', label='Engine (TSFC)')

    ax.plot(years, p_all_tsfc(years),color='lightblue')
    ax.plot(years, p_all_eu(years),color='black')
    ax.plot(years, p_all_oew(years),color='steelblue')
    ax.plot(years, p_all_ld(years), color='royalblue')

    # Add a legend to the plot
    ax.legend()
    plt.xlim(1955, 2025)
    plt.ylim(-30, 350)
    plot.plot_layout(None, x_label, y_label, ax)
    if savefig:
        plt.savefig(folder_path+'/ida_technological_normalized.png')

    # Evaluate the polynomials for the x values and add 100 to get back to the Efficiency
    p_all_tsfc_values = p_all_tsfc(years) + 100
    p_all_oew_values = p_all_oew(years) + 100
    p_all_ld_values = p_all_ld(years) + 100
    p_all_eu_values = p_all_eu(years) + 100

    # Create a dictionary with the polynomial values
    data = {
        'YOI': years,
        'TSFC Cruise': p_all_tsfc_values,
        'OEW/Exit Limit': p_all_oew_values,
        'L/D estimate': p_all_ld_values,
        'EU (MJ/ASK)': p_all_eu_values
    }

    # Create the DataFrame
    data = pd.DataFrame(data)

    # Use LMDI Method
    data['LMDI'] = (data['EU (MJ/ASK)'] - data['EU (MJ/ASK)'].iloc[0]) / (np.log(data['EU (MJ/ASK)']) - np.log(data['EU (MJ/ASK)'].iloc[0]))
    data['Engine_LMDI'] = np.log(data['TSFC Cruise'] / data['TSFC Cruise'].iloc[0])
    data['Aerodyn_LMDI'] = np.log(data['L/D estimate'] / data['L/D estimate'].iloc[0])
    data['Structural_LMDI'] = np.log(data['OEW/Exit Limit'] / data['OEW/Exit Limit'].iloc[0])
    data['deltaC_Aerodyn'] = data['LMDI'] * data['Aerodyn_LMDI']
    data['deltaC_Engine'] = data['LMDI'] * data['Engine_LMDI']
    data['deltaC_Structural'] = data['LMDI'] * data['Structural_LMDI']
    data['deltaC_Tot'] = data['EU (MJ/ASK)'] - data['EU (MJ/ASK)'].iloc[0]
    data['deltaC_Res'] = data['deltaC_Tot'] - data['deltaC_Aerodyn'] - data['deltaC_Engine'] - data['deltaC_Structural']

    # Get percentage increase of each efficiency and drop first row which only contains NaN
    data = data[['YOI', 'deltaC_Structural', 'deltaC_Engine', 'deltaC_Aerodyn', 'deltaC_Res', 'deltaC_Tot']]
    data = data.drop(0)
    data.to_excel(r'dashboard\data\Dashboard.xlsx', index=False)
    data = data.set_index('YOI')

    # Set the width of each group and create new indexes just the set the space right
    data = data[['deltaC_Tot', 'deltaC_Engine', 'deltaC_Aerodyn', 'deltaC_Structural', 'deltaC_Res']]

    # Reorder the columns
    column_order = ['deltaC_Tot','deltaC_Aerodyn', 'deltaC_Structural', 'deltaC_Engine', 'deltaC_Res']
    data = data[column_order]
    # Create new Labels
    labels = ['Overall (MJ/ASK)', 'Aerodynamic (L/D)','Structural (OEW/PEL)','Engine (TSFC)', 'Residual' ]

    # Create subplots for each column
    fig, ax = plt.subplots(dpi=300)

    # Plot stacked areas for other columns
    data_positive = data.drop('deltaC_Tot', axis=1).clip(lower=0)
    data_negative = data.drop('deltaC_Tot', axis=1).clip(upper=0)
    # Create arrays for stacking the areas
    positive_stack = np.zeros(len(data))
    negative_stack = np.zeros(len(data))

    # Plot overall efficiency as a line
    overall_efficiency = data['deltaC_Tot']
    ax.plot(data.index, overall_efficiency, color='black', label=labels[0], linewidth= 3)

    #Plot Subefficiencies
    colors = ['royalblue', 'steelblue', 'lightblue', 'red']
    for i, column in enumerate(data_positive.columns):
        ax.fill_between(data.index, positive_stack, positive_stack + data_positive.iloc[:, i], color=colors[i],
                        label=labels[i + 1], linewidth=0)
        positive_stack += data_positive.iloc[:, i]
    for i, column in enumerate(data_negative.columns):
        ax.fill_between(data.index, negative_stack, negative_stack + data_negative.iloc[:, i], color=colors[i], linewidth=0)
        negative_stack += data_negative.iloc[:, i]

    xlabel = 'Year'
    ylabel = 'Efficiency Improvements [\%]'
    ax.set_xlim(1958, 2020)
    ax.set_ylim(-30, 330)
    ax.legend(loc='upper left')
    plot.plot_layout(None, xlabel, ylabel, ax)
    if savefig:
        plt.savefig(folder_path+'/ida_technological.png')

