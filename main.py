# %%

from pathlib import Path
import processing
import processing.statistical_data

path_csv_t2 = Path("/Users/michaelweinold/github/Aircraft-Performance/database/rawdata/USDOT/T_SCHEDULE_T2.csv")
path_csv_aircraft_types = Path("/Users/michaelweinold/github/Aircraft-Performance/database/rawdata/USDOT/L_AIRCRAFT_TYPE (1).csv")

processing.statistical_data.process_data_usdot_t2(
    path_csv_t2=path_csv_t2,
    path_csv_aircraft_types=path_csv_aircraft_types
)