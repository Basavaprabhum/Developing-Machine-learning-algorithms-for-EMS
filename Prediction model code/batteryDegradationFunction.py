
import math as math
def degradationCoefficient(batteryCapacity, operationalMode):
    temperature = 15 + 273.15
    endlife = 0
    ratedVoltage = 230
    specificCost = 100 #€/kWh capacity
    if operationalMode == 2:
        Crate = 1.5  # kW
    else:
        Crate = 7  # kW


# Experimental coefficients
    a = 8.61e-6
    b = -5.13e-3
    c = 7.63e-1
    d = -6.7e-3
    e = 2.35
    B1 = 0.0021
    B2 = (d * temperature + e)
    q = B1 * math.exp((d * temperature + e) * (Crate/batteryCapacity))
    cb = q * batteryCapacity * specificCost / (1 - endlife) / ratedVoltage
    return  cb