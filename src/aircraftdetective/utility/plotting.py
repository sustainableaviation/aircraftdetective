from aircraftdetective import ureg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
cm = 1/2.54 # for inches-cm conversion

def set_figure_and_axes():
    
    fig, ax = plt.subplots(
        num = 'main',
        nrows = 1,
        ncols = 1,
        dpi = 300,
        figsize=(10*cm, 15*cm),
    )

    # TICKS AND LABELS ###########

    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=True)
    ax.tick_params(axis='y', which='minor', bottom=True)

    # GRIDS ######################

    ax.grid(which='major', axis='y', linestyle='-', linewidth = 0.5)
    ax.grid(which='minor', axis='y', linestyle=':', linewidth = 0.5)
    ax.grid(which='major', axis='x', linestyle='-', linewidth = 0.5)
    ax.grid(which='minor', axis='x', linestyle=':', linewidth = 0.5)

    return fig, ax