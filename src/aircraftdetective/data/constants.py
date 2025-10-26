import pint
ureg = pint.get_application_registry()

jeta1_energydensity = 34.7*1E6 * ureg.J / ureg.l # https://en.wikipedia.org/wiki/Jet_fuel
jeta1_density = 0.804 * ureg.kg / ureg.l # https://en.wikipedia.org/wiki/Jet_fuel
jeta1_specificenergy = jeta1_energydensity / jeta1_density
g_acceleration = 9.81 * ureg('m/s**2')