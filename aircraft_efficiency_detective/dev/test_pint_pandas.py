# %%

from aircraft_efficiency_detective import ureg

import pandas as pd
import pint
import pint_pandas
ureg = pint.get_application_registry()

df = pd.DataFrame(
  {
      ('range', 'km'): [200]
  }
).pint.quantify(level=1)

df = df.pint.convert_object_dtype()

@ureg.check(
    '[length]',
)
def multiply_value(range):
    return range * 2

multiply_value(df.iloc[0]['range'])