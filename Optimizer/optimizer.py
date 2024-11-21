

import numpy as np
import pulp
# Here are four different subfucntion. You will use it according to the fact that you are in a V2H or V2G mode.
# Note that everything cna be condensed in one single more complex region. Refer to the thesis to understand hihg soc or low soc.

def optimizerBase(prices, load, intervals, maxPowerEV,maxPowerEVSE, lowerBoundChargingPower,maxPowerGRID,timestep , evCapacity, etach,etadh,initialSOC , safetySOC, maximumSOC,departureSOC,optimizationFunction,batteryDegradation ):
    etadh = 1/etadh
    eps = 0.0001  # Stability of the equations
    alpha = 0.1 #Weight in the multiobjective function
    beta = 1-alpha

    upperLimit = {'charge': min(maxPowerEV, maxPowerEVSE),   # self explainatory. YOu have limit in charge and dh
                   'discharge': min(maxPowerEV, maxPowerEVSE),
                  }

    lowlimit = {'charge': lowerBoundChargingPower,
                'discharge': 0,
                }

    price = prices
    # SETS
    M = ['charge', 'discharge']  # modes of operations
    T = range(intervals)  # the time periods

    TM = {(t, m) for t in T for m in M}

    model = pulp.LpProblem('Batts', pulp.LpMinimize ) # here ypou create the model

    # VARS create the variables
    model.batt = pulp.LpVariable.dicts('batt_state',TM,lowBound=0,cat='Continuous')
    model.op_mode = pulp.LpVariable.dicts('op_mode',TM, cat='Binary')
    model.grid = pulp.LpVariable.dicts('grid',T, lowBound=0,cat='Continuous')
    model.wasted = pulp.LpVariable.dicts('waste',T, lowBound=0, cat='Continuous')
    model.zone2 = pulp.LpVariable.dicts('conditional',T, cat='Binary') # Are you below EV minimum V2X Energy Level? 1 is yes
    # Constraints

    # only one op mode in each time period...
    for t in T:
        model += sum(model.op_mode[t, m] for m in M) <= 1 # no sdimultaneous operations
        model += model.batt[t, 'charge'] + load[t] - model.batt[t, 'discharge'] == model.grid[t] - model.wasted[t] # real power balance
        model += model.grid[t] <= maxPowerGRID #grid power limit

        model += model.batt[t, 'discharge'] <= load[t] * model.op_mode[t, 'discharge'] # V2H limitation

        model += initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge']for t1 in range(t)) * etach) * timestep / 60 /evCapacity <= maximumSOC # limit of the soc

        # Zone 2 prescription. Conditional constrains
        model += ((initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge'] for t1 in range(t)) * etach) * timestep / 60 / evCapacity) - safetySOC) <= (1- safetySOC) * ( 1 - model.zone2[t])
        model += ((initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge'] for t1 in range(t)) * etach) * timestep / 60 / evCapacity) - safetySOC) >= eps + (-safetySOC - eps) * model.zone2[t]
       # Prevent the fact that when in xzone 2 you dischargew
        model += model.zone2[t] - (1 - model.op_mode[t, 'discharge']) <= 0


        # Big-M constraint. limit rates for each rate, in each period.
    # this does 2 things:  it is equivalent to the upper bound parameter in the var declaration
    #                      It is a Big-M type of constraint which uses the binary var as a control <-- key point
    for t, m in TM:
        model += model.batt[t, m] <= upperLimit[m] * model.op_mode[t, m]
        model += model.batt[t, m] >= lowlimit[m] * model.op_mode[t, m]

    model += sum(model.batt[t,'charge'] for t in T) * etach - sum(model.batt[t,'discharge'] for t in T) * etadh == (departureSOC - initialSOC) * evCapacity * 60 / timestep
    model += sum(model.batt[t,'discharge'] for t in T) * timestep / 60 <= 10
    # OBJ
#Here you take the objective function according to your choices
    if optimizationFunction == 1:
        model += (sum(model.grid[t] * price[t] for t in T) + batteryDegradation * sum(model.batt[t, 'discharge'] for t in T)) * timestep / 60
    else:
        model += alpha * sum(model.wasted[t] for t in T) + beta * (sum(model.grid[t] * price[t] for t in T) + batteryDegradation * sum(model.batt[t, 'discharge'] for t in T)) * timestep / 60
    model.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=60*2))


# managing the result correctly
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
def optimizerHighSOC(prices, load, intervals, maxPowerEV,maxPowerEVSE, lowerBoundChargingPower,maxPowerGRID,timestep , evCapacity, etach,etadh,initialSOC , safetySOC, maximumSOC, departureSOC, maximumSOCV2X, optimizationFunction,batteryDegradation):
    etadh = 1/etadh
    eps = 0.0001
    alpha = 1
    beta = 0.5
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
    #model.maxpower = pulp.LpVariable.dicts('maximum', indexs=T, lowBound=0, cat='Continuous')
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
        #model += model.maxpower[t] == upperLimit['charge'] - 1 / (1-maximumSOCV2X)*((initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge'] for t1 in range(t)) * etach) * timestep / 60 / evCapacity) - maximumSOCV2X)*(upperLimit['charge'] - lowlimit['charge'])
        #model += model.batt[t, 'charge'] <= model.maxpower[t]
        # Big-M constraint. limit rates for each rate, in each period.
    # this does 2 things:  it is equivalent to the upper bound parameter in the var declaration
    #                      It is a Big-M type of constraint which uses the binary var as a control <-- key point
    for t, m in TM:
        model += model.batt[t, m] <= upperLimit[m] * model.op_mode[t, m]
        model += model.batt[t, m] >= lowlimit[m] * model.op_mode[t, m]

    model += sum(model.batt[t,'charge'] for t in T) *etach - sum(model.batt[t,'discharge'] for t in T) * etadh == (departureSOC - initialSOC) * evCapacity * 60 / timestep
    model += sum(model.batt[t, 'discharge'] for t in T) * timestep / 60 <= 10
    # OBJ

    if optimizationFunction == 1:
        model += (sum(model.grid[t] * price[t] for t in T) + batteryDegradation * sum(model.batt[t, 'discharge'] for t in T)) * timestep / 60
    else:
        model += alpha * sum(model.wasted[t] for t in T) + beta * sum(model.grid[t] * price[t] for t in T)
 #   else:
#     model.Pmax = pulp.LpVariable.dicts('p_max', cat='Continuous')
#        specificCost = 600
#
#        for t in T:
#            model += model.grid[t] <= model.Pmax


 #       model += sum(model.grid[t] * price[t] for t in T) + model.Pmax * specificCost
    model.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=60*2))

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







def optimizerBaseV2G(prices,selling, load, intervals, maxPowerEV,maxPowerEVSE, lowerBoundChargingPower,maxPowerGRID,timestep , evCapacity, etach,etadh,initialSOC , safetySOC, maximumSOC,departureSOC,optimizationFunction,batteryDegradation ):
    etadh = 1/etadh
    eps = 0.0001


    upperLimitBattery = {'charge': min(maxPowerEV, maxPowerEVSE),
                   'discharge': min(maxPowerEV, maxPowerEVSE),
                  }

    lowlimitBattery = {'charge': lowerBoundChargingPower,
                'discharge': 0,
                }

    price = prices  # this could be a table or double-indexed table of [t, m] or ....
    solver = pulp.GUROBI()
    # SETS
    M = ['charge', 'discharge']   # modes.  probably others...  discharge?
    S = ['buy', 'sell']
    T = range(intervals)  # the time periods

    TM = {(t, m) for t in T for m in M}
    TS = {(t, s) for t in T for s in S}
    model = pulp.LpProblem('Batts', pulp.LpMinimize)

    # VARS
    model.batt = pulp.LpVariable.dicts('batt_state',indexs=TM,lowBound=0,cat='Continuous')
    model.op_mode = pulp.LpVariable.dicts('op_mode',indexs=TM, cat='Binary')
    model.grid = pulp.LpVariable.dicts('grid', indexs=TS,lowBound=0,cat='Continuous')
    model.grid_mode = pulp.LpVariable.dicts('grid_mode', indexs=TS, cat='Binary')
    model.zone2 = pulp.LpVariable.dicts('conditional', indexs=T, cat='Binary') #Are you below EV minimum V2X Energy Level? 1 is yes
    # Constraints

    # only one op mode in each time period...
    for t in T:

        model += sum(model.op_mode[t, m] for m in M) <= 1
        model += model.batt[t, 'charge'] + load[t] - model.batt[t, 'discharge'] == model.grid[t, 'buy'] - model.grid[t, 'sell']

        model += model.grid[t, 'buy'] <= maxPowerGRID * model.grid_mode[t, 'buy']
        model += model.grid[t, 'sell'] <= maxPowerGRID * model.grid_mode[t, 'sell']
        model += sum(model.grid_mode[t, s] for s in S) <= 1
        # model += model.batt[t, 'charge'] + load[t]  == model.grid[t, 'buy']
        #model += model.batt[t, 'discharge'] + load[t] == model.grid[t, 'sell']
        #model += model.batt[t, 'discharge'] <= load[t] * model.op_mode[t, 'discharge']
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
        model += model.batt[t, m] <= upperLimitBattery[m] * model.op_mode[t, m]
        model += model.batt[t, m] >= lowlimitBattery[m] * model.op_mode[t, m]

    model += sum(model.batt[t,'charge'] for t in T) * etach - sum(model.batt[t,'discharge'] for t in T) * etadh == (departureSOC - initialSOC) * evCapacity * 60 / timestep
    model += sum(model.batt[t, 'discharge'] for t in T) * timestep / 60 <= 10
    # OBJ
    model += (sum(model.grid[t, 'buy'] * price[t] for t in T) + batteryDegradation * sum(model.batt[t, 'discharge'] for t in T) - sum(model.grid[t, 'sell'] * selling[t] for t in T)) * timestep / 60
    model.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=60*2))

    cPnew = []
    dPnew = []
    cSnew = []
    dSnew = []
    gPboughtnew = []
    gPsoldnew=[]
    zone2State = []

    for m in M:
        for t in T:
            if m == 'charge':
                cPnew = np.append(cPnew, model.batt[t, m].varValue)
                cSnew = np.append(cSnew, model.op_mode[t, m].varValue)
            else:
                dPnew = np.append(dPnew, model.batt[t, m].varValue)
                dSnew = np.append(dSnew, model.op_mode[t, m].varValue)
    for t in T:
        zone2State = np.append(zone2State, model.zone2[t].varValue)

    for s in S:
        for t in T:
            if s == 'buy':
                gPboughtnew = np.append(gPboughtnew, model.grid[t, s].varValue)
            else:
                gPsoldnew = np.append(gPsoldnew, model.grid[t, s].varValue)

    return cPnew, dPnew,cSnew,dSnew,gPboughtnew,gPsoldnew,zone2State


# Here I have 2 threshold: size of the battery and limit for discharging
def optimizerHighSOCV2G(prices,selling, load, intervals, maxPowerEV,maxPowerEVSE, lowerBoundChargingPower,maxPowerGRID,timestep , evCapacity, etach,etadh,initialSOC , safetySOC, maximumSOC, departureSOC, maximumSOCV2X, optimizationFunction,batteryDegradation):
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
    M = ['charge', 'discharge']   # modes.  probably others...  discharge?
    S = ['buy', 'sell']
    T = range(intervals)  # the time periods

    TM = {(t, m) for t in T for m in M}
    TS = {(t, s) for t in T for s in S}
    model = pulp.LpProblem('Batts', pulp.LpMinimize)

    # VARS
    model.batt = pulp.LpVariable.dicts('batt_state',indexs=TM,lowBound=0,cat='Continuous')
    model.op_mode = pulp.LpVariable.dicts('op_mode',indexs=TM, cat='Binary')
    model.grid = pulp.LpVariable.dicts('grid', indexs=TS, cat='Continuous')
    model.grid_mode = pulp.LpVariable.dicts('grid_mode', indexs=TS, cat='Binary')
    model.zone2 = pulp.LpVariable.dicts('zone2', indexs=T, cat='Binary') # Are you below EV minimum V2X Energy Level? 1 is yes
    model.zone4 = pulp.LpVariable.dicts('zone4', indexs=T, cat='Binary') # Are you below EV minimum V2X Energy Level? 1 is yes

    # Constraints

    # only one op mode in each time period...
    for t in T:
        model += sum(model.op_mode[t, m] for m in M) <= 1
        model += model.batt[t, 'charge'] + load[t] - model.batt[t, 'discharge'] == model.grid[t, 'buy'] - model.grid[t, 'sell']

        model += model.grid[t, 'buy'] <= maxPowerGRID * model.grid_mode[t, 'buy']
        model += model.grid[t, 'sell'] <= maxPowerGRID * model.grid_mode[t, 'sell']
        model += sum(model.grid_mode[t, s] for s in S) <= 1
        #model += model.batt[t, 'charge'] + load[t] - model.batt[t, 'discharge'] >= model.grid[t]

        model += initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge']for t1 in range(t)) * etach) * timestep / 60 /evCapacity <= maximumSOC
        #model += initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge']for t1 in range(t)) * etach) * timestep / 60 /evCapacity >= safetySOC

        model += ((initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge'] for t1 in range(t)) * etach) * timestep / 60 / evCapacity) - safetySOC) <= (1 - safetySOC) * (1 - model.zone2[t])
        model += ((initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge'] for t1 in range(t)) * etach) * timestep / 60 / evCapacity) - safetySOC) >= eps + (-safetySOC - eps) * model.zone2[t]
        model += model.zone2[t] - (1 - model.op_mode[t, 'discharge']) <= 0

        model += ((initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge'] for t1 in range(t)) * etach) * timestep / 60 / evCapacity) - maximumSOCV2X) <= (1 - maximumSOCV2X) * model.zone4[t]
        model += ((initialSOC + (-sum(model.batt[t1, 'discharge'] for t1 in range(t)) * etadh + sum(model.batt[t1, 'charge'] for t1 in range(t)) * etach) * timestep / 60 / evCapacity) - maximumSOCV2X) >= eps + (-maximumSOCV2X - eps) * (1 - model.zone4[t])
        model += model.zone4[t] - (1 - model.op_mode[t, 'discharge']) <= 0

        # Big-M constraint. limit rates for each rate, in each period.
    # this does 2 things:  it is equivalent to the upper bound parameter in the var declaration
    #                      It is a Big-M type of constraint which uses the binary var as a control <-- key point
    for t, m in TM:
        model += model.batt[t, m] <= upperLimit[m] * model.op_mode[t, m]
        model += model.batt[t, m] >= lowlimit[m] * model.op_mode[t, m]

    model += sum(model.batt[t,'charge'] for t in T) *etach - sum(model.batt[t,'discharge'] for t in T) * etadh == (departureSOC - initialSOC) * evCapacity * 60 / timestep
    model += sum(model.batt[t, 'discharge'] for t in T) * timestep / 60 <= 10
    # OBJ

    model += (sum(model.grid[t, 'buy'] * price[t] for t in T) + batteryDegradation * sum(model.batt[t, 'discharge'] for t in T) - sum(model.grid[t, 'sell'] * selling[t] for t in T)) * timestep / 60
    model.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=60 * 2))


    cPnew = []
    dPnew = []
    cSnew = []
    dSnew = []
    gPboughtnew = []
    gPsoldnew = []
    zone2State = []
    zone4State = []
    for m in M:
        for t in T:
            if m == 'charge':
                cPnew = np.append(cPnew, model.batt[t, m].varValue)
                cSnew = np.append(cSnew, model.op_mode[t, m].varValue)
            else:
                dPnew = np.append(dPnew, model.batt[t, m].varValue)
                dSnew = np.append(dSnew, model.op_mode[t, m].varValue)
    for t in T:
        zone2State = np.append(zone2State, model.zone2[t].varValue)
        zone4State = np.append(zone4State, model.zone4[t].varValue)

    for s in S:
        for t in T:
            if s == 'buy':
                gPboughtnew = np.append(gPboughtnew, model.grid[t, s].varValue)
            else:
                gPsoldnew = np.append(gPsoldnew, model.grid[t, s].varValue)

    return cPnew, dPnew,cSnew,dSnew,gPboughtnew,gPsoldnew,zone2State, zone4State

