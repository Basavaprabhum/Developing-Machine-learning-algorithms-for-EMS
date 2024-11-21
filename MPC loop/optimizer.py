

import numpy as np
import pulp


def optimizerBase(prices, load, intervals, maxPowerEV, maxPowerEVSE, lowerBoundChargingPower, maxPowerGRID, timestep, evCapacity, etach, etadh, initialSOC, safetySOC, maximumSOC, departureSOC, optimizationFunction,
                  batteryDegradation: object) -> object:
    etadh = 1/etadh
    eps = 0.0001

    upperLimit = {'charge': min(maxPowerEV, maxPowerEVSE),
                   'discharge': min(maxPowerEV, maxPowerEVSE),
                  }

    lowlimit = {'charge': lowerBoundChargingPower,
                'discharge': 0,
                }

    price = prices  # this could be a table or double-indexed table of [t, m] or ....
    solver = pulp.GUROBI()
    # SETS
    M = ['charge', 'discharge']  # modes.  probably others...  discharge?
    T = range(intervals)  # the time periods

    TM = {(t, m) for t in T for m in M}

    model = pulp.LpProblem('Batts', pulp.LpMinimize)

    # VARS
    model.batt = pulp.LpVariable.dicts('batt_state',TM,lowBound=0,cat='Continuous')
    model.op_mode = pulp.LpVariable.dicts('op_mode',TM, cat='Binary')
    model.grid = pulp.LpVariable.dicts('grid', T, lowBound=0,cat='Continuous')
    model.wasted = pulp.LpVariable.dicts('waste', T, lowBound=0, cat='Continuous')
    model.zone2 = pulp.LpVariable.dicts('conditional', T, cat='Binary') # Are you below EV minimum V2X Energy Level? 1 is yes
    # Constraints

    # only one op mode in each time period...
    for t in T:
        model += sum(model.op_mode[t, m] for m in M) <= 1
        model += model.batt[t, 'charge'] + load[t] - model.batt[t, 'discharge'] == model.grid[t] - model.wasted[t]
        model += model.grid[t] <= maxPowerGRID
        #model += model.batt[t, 'charge'] + load[t] - model.batt[t, 'discharge'] >= model.grid[t]
        model += model.batt[t, 'discharge'] <= load[t] * model.op_mode[t, 'discharge']
        model += initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge']for t1 in range(t)) * etach) * timestep / 60 /evCapacity <= maximumSOC
        #model += initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge']for t1 in range(t)) * etach) * timestep / 60 /evCapacity >= safetySOC

        #model += safetySOC - (initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge'] for t1 in range(t)) * etach) * timestep / 60 / evCapacity) <= 1 * model.zone2[t]

        model += ((initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge'] for t1 in range(t)) * etach) * timestep / 60 / evCapacity) - safetySOC) <= (1- safetySOC) * ( 1 - model.zone2[t])
        model += ((initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge'] for t1 in range(t)) * etach) * timestep / 60 / evCapacity) - safetySOC) >= eps + (-safetySOC - eps) * model.zone2[t]
        model += model.zone2[t] - (1 - model.op_mode[t, 'discharge']) <= 0


        # Big-M constraint. limit rates for each rate, in each period.
    # this does 2 things:  it is equivalent to the upper bound parameter in the var declaration
    #                      It is a Big-M type of constraint which uses the binary var as a control <-- key point
    for t, m in TM:
        model += model.batt[t, m] <= upperLimit[m] * model.op_mode[t, m]
        model += model.batt[t, m] >= lowlimit[m] * model.op_mode[t, m]

    model += sum(model.batt[t,'charge'] for t in T) * etach - sum(model.batt[t,'discharge'] for t in T) * etadh >= (departureSOC - initialSOC) * evCapacity * 60 / timestep

    # OBJ

    if optimizationFunction == 1:
        model += (sum(model.grid[t] * price[t] for t in T) + batteryDegradation * sum(model.batt[t, 'discharge'] for t in T)) * timestep / 60
    else:
        model += sum(model.wasted[t] for t in T) + (initialSOC + (-sum(model.batt[t, 'discharge'] for t in T) * etadh + sum(model.batt[t, 'charge']  for t in T)) * etach) * timestep / 60 / evCapacity
    model.solve()

    cPnew = []
    dPnew = []
    cSnew = []
    dSnew = []
    gPnew = []
    zone2State = []
    wPnew = []
    for m in M:
        for t in T:
            if m == 'charge':
                cPnew = np.append(cPnew, model.batt[t, m].varValue)
                cSnew = np.append(cSnew, model.op_mode[t, m].varValue)
            else:
                dPnew = np.append(dPnew, model.batt[t, m].varValue)
                dSnew = np.append(dSnew, model.op_mode[t, m].varValue)
    for t in T:
        gPnew = np.append(gPnew, model.grid[t].varValue)
        zone2State = np.append(zone2State, model.zone2[t].varValue)
        wPnew = np.append(wPnew, model.wasted[t].varValue)
    return cPnew, dPnew,cSnew,dSnew,gPnew,zone2State,wPnew


# Here I have 2 threshold: size of the battery and limit for discharging
def optimizerHighSOC(prices, load, intervals, maxPowerEV,maxPowerEVSE, lowerBoundChargingPower,maxPowerGRID,timestep , evCapacity, etach,etadh,initialSOC , safetySOC, maximumSOC, departureSOC, maximumSOCV2X, optimizationFunction):
    etadh = 1/etadh
    eps = 0.0001

    upperLimit = {'charge': min(maxPowerEV, maxPowerEVSE),
                   'discharge': min(maxPowerEV, maxPowerEVSE),
                  }

    lowlimit = {'charge': lowerBoundChargingPower,
                'discharge': 0,
                }

    price = prices  # this could be a table or double-indexed table of [t, m] or ....

    # SETS
    M = ['charge', 'discharge']  # modes.  probably others...  discharge?
    T = range(intervals)  # the time periods

    TM = {(t, m) for t in T for m in M}

    model = pulp.LpProblem('Batts', pulp.LpMinimize)

    # VARS
    model.batt = pulp.LpVariable.dicts('batt_state',indexs=TM,lowBound=0,cat='Continuous')
    model.op_mode = pulp.LpVariable.dicts('op_mode',indexs=TM, cat='Binary')
    model.grid = pulp.LpVariable.dicts('grid', indexs=T, cat='Continuous')
    model.zone2 = pulp.LpVariable.dicts('zone2', indexs=T, cat='Binary') # Are you below EV minimum V2X Energy Level? 1 is yes
    model.wasted = pulp.LpVariable.dicts('waste', indexs=T, lowBound=0, cat='Continuous')
    model.zone4 = pulp.LpVariable.dicts('zone4', indexs=T, cat='Binary') # Are you below EV minimum V2X Energy Level? 1 is yes
    model.maxpower = pulp.LpVariable.dicts('maximum', indexs=T, lowBound=0, cat='Continuous')
    # Constraints

    # only one op mode in each time period...
    for t in T:
        model += sum(model.op_mode[t, m] for m in M) <= 1
        model += model.batt[t, 'charge'] + load[t] - model.batt[t, 'discharge'] == model.grid[t] - model.wasted[t]
        model += model.grid[t] <= maxPowerGRID
        #model += model.batt[t, 'charge'] + load[t] - model.batt[t, 'discharge'] >= model.grid[t]
        model += model.batt[t, 'discharge'] <=  load[t] * model.op_mode[t, 'discharge']
        model += initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge']for t1 in range(t)) * etach) * timestep / 60 /evCapacity <= maximumSOC
        #model += initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge']for t1 in range(t)) * etach) * timestep / 60 /evCapacity >= safetySOC

        model += ((initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge'] for t1 in range(t)) * etach) * timestep / 60 / evCapacity) - safetySOC) <= (1 - safetySOC) * (1 - model.zone2[t])
        model += ((initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge'] for t1 in range(t)) * etach) * timestep / 60 / evCapacity) - safetySOC) >= eps + (-safetySOC - eps) * model.zone2[t]
        model += model.zone2[t] - (1 - model.op_mode[t, 'discharge']) <= 0

        model += ((initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge'] for t1 in range(t)) * etach) * timestep / 60 / evCapacity) - maximumSOCV2X) <= (1 - maximumSOCV2X) * model.zone4[t]
        model += ((initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge'] for t1 in range(t)) * etach) * timestep / 60 / evCapacity) - maximumSOCV2X) >= eps + (-maximumSOCV2X - eps) * (1 - model.zone4[t])

        model += model.zone4[t] - (1 - model.op_mode[t, 'discharge']) <= 0
        model += model.maxpower[t] == upperLimit['charge'] - 1 / (1-maximumSOCV2X)*((initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge'] for t1 in range(t)) * etach) * timestep / 60 / evCapacity) - maximumSOCV2X)*(upperLimit['charge'] - lowlimit['charge'])
        model += model.batt[t, 'charge'] <= model.maxpower[t]
        # Big-M constraint. limit rates for each rate, in each period.
    # this does 2 things:  it is equivalent to the upper bound parameter in the var declaration
    #                      It is a Big-M type of constraint which uses the binary var as a control <-- key point
    for t, m in TM:
        model += model.batt[t, m] <= upperLimit[m] * model.op_mode[t, m]
        model += model.batt[t, m] >= lowlimit[m] * model.op_mode[t, m]

    model += sum(model.batt[t,'charge'] for t in T) *etach - sum(model.batt[t,'discharge'] for t in T) * etadh == (departureSOC - initialSOC) * evCapacity * 60 / timestep

    # OBJ
    if optimizationFunction == 1:
        model += sum(model.grid[t] * price[t] for t in T) * timestep/60
    else:
        model += sum(model.wasted[t] for t in T)
 #   else:
#     model.Pmax = pulp.LpVariable.dicts('p_max', cat='Continuous')
#        specificCost = 600
#
#        for t in T:
#            model += model.grid[t] <= model.Pmax


 #       model += sum(model.grid[t] * price[t] for t in T) + model.Pmax * specificCost
    model.solve()

    cPnew = []
    dPnew = []
    cSnew = []
    dSnew = []
    gPnew = []
    zone2State = []
    zone4State = []
    wPnew = []
    for m in M:
        for t in T:
            if m == 'charge':
                cPnew = np.append(cPnew, model.batt[t, m].varValue)
                cSnew = np.append(cSnew, model.op_mode[t, m].varValue)
            else:
                dPnew = np.append(dPnew, model.batt[t, m].varValue)
                dSnew = np.append(dSnew, model.op_mode[t, m].varValue)
    for t in T:
        gPnew = np.append(gPnew, model.grid[t].varValue)
        zone2State = np.append(zone2State, model.zone2[t].varValue)
        zone4State = np.append(zone4State, model.zone4[t].varValue)
        wPnew = np.append(wPnew, model.wasted[t].varValue)

    return cPnew, dPnew,cSnew,dSnew,gPnew,zone2State, zone4State, wPnew


