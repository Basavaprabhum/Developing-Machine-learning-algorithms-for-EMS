# -------------------------------------------------------------------------------
# ----------------------Import section--------------------------------------------
# -------------------------------------------------------------------------------
from Loadnew import *
from batteryDegradationFunction import *
from renewable import *
from costElectricity import *
from optimizer import *
from addiction import *
from Loadnew import *
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.pyplot as plt
import tkinter as tk
from ele_cost_test import *

# -------------------------------------------------------------------------------
# ----------------------Input section--------------------------------------------
# -------------------------------------------------------------------------------
#This is the main. YOu will call the function main4 gui below form gui inteface and this will start

final= None
def main4gui(root):
    global final
    batteryCapacity = root.batterycap #XC40
    powerInjection = root.injectionFlag #Are you injecting or not? This is taken from the GUI


    batteryDegradation = degradationCoefficient(batteryCapacity, powerInjection) #€/kWh
    #batteryDegradation = root.deg


    socArrival = root.arrival_soc
    socDeparture = root.departure_soc


    hours_ = root.departure_time

    maxPowerEV = 10 # Power limitation imposed by vehicle
    maxPowerEVSE = 10 # Power limitation imposed by EVSE

    maxPowerGRID = root.peak
    minChargingPower = 4 # Minimum charging power

    discretizationStep = 5 # in minutes
    hours, minute_ = map(int, hours_.split(':')) # Needed to be able to import the time availability

    #now_rounded = dt.datetime.now() # Here you have the option to start simulation from the current time or from the a time you decide (next line)
    now_rounded = dt.datetime(2023,4,1,16,00,00)
    # Here you have two options. YOu can take the current time or you can fix a specific time of the day to start the simulations

    intervals = int((hours * 60 + minute_) / discretizationStep)
    final = now_rounded + dt.timedelta(minutes=(intervals * discretizationStep)) # Final time of the simulation

    days = mdates.drange(now_rounded, final, dt.timedelta(minutes=discretizationStep))
    days2plot = mdates.drange(now_rounded, final, dt.timedelta(minutes=1))
    if int(len(days)) > intervals: # This is a control for plotting results
        diff = int(len(days)) - intervals
        days = days[:-diff]
        days2plot = days2plot[:-1]

    minimumEnergyLevel = root.min_soc_level # Energy level taken form GUI
    EVMaximumEnergyLevel = root.max_soc_level
    EVMinimumV2XEnergyLevel = root.min_soc_v2x
    EVMaximumV2XEnergyLevel = root.max_soc_v2x

    marketPriceflag = root.priceflag
    optimizationFunction = root.optimizationFunction
    loadHouseForecast = loadDefinitionnew(days)
    renewableProduction = renewableArray(days, discretizationStep)
    netLoad = loadHouseForecast - renewableProduction  #if positive is to fulfill in negative in excess
    electricityCost = costElectricity1(now_rounded, intervals, discretizationStep)
    etaCharging = 0.97 # Comulative efficiency of the process
    etaDischarging = 0.97
    availablePower = maxPowerGRID - netLoad
    # -------------------------------------------------------------------------------
    # ----------------------Optimizer section----------------------------------------
    # -------------------------------------------------------------------------------
    time = 0
    socRegister = np.array([socArrival])
    cP = []
    cS = []
    dP = []
    dS = []
    gP = []
    zone2 = []
    zone4 = []
    zone4State = []
    wP = []

    # Variable descriptions:
    # socRegister is the tate of charge
    # cP charging power
    # cS charging state
    # dP discharging power
    # dS discharging state
    # zone2 quantify the presence (1) or not (0) in the zone 2 region imposed by the protocol
    # zone4 quantify the presence (1) or not (0) in the zone 4 region imposed by the protocol
    while time in range(intervals):
         if socRegister[-1] < minimumEnergyLevel: # if you are below the lowest threshold the charging is rule based
             if availablePower[time] >= minChargingPower:
                 if availablePower[time] <= min(maxPowerEV, maxPowerEVSE):
                     cP = np.append(cP, availablePower[time])
                     cS = np.append(cS, 1)
                     dS = np.append(dS, 0)
                     dP = np.append(dP, 0)
                     gP = np.append(gP, netLoad[time] + cP[-1])
                     zone2 = np.append(zone2, 0)
                     zone4 = np.append(zone4, 0)
                     wP = np.append(wP, 0)
                     socRegister = np.append(socRegister, socRegister[-1] + cP[
                         -1] * etaCharging * discretizationStep / 60 / batteryCapacity)
                     time = time + 1
                 else:
                     cP = np.append(cP, min(maxPowerEV, maxPowerEVSE))
                     cS = np.append(cS, 1)
                     dS = np.append(dS, 0)
                     dP = np.append(dP, 0)
                     gP = np.append(gP, netLoad[time] + cP[-1])
                     zone2 = np.append(zone2, 0)
                     zone4 = np.append(zone4, 0)
                     wP = np.append(wP, 0)
                     socRegister = np.append(socRegister, socRegister[-1] + cP[
                         -1] * etaCharging * discretizationStep / 60 / batteryCapacity)

                     time = time + 1
             else:
                     cP = np.append(cP, 0)
                     cS = np.append(cS, 0)
                     dS = np.append(dS, 0)
                     dP = np.append(dP, 0)
                     wP = np.append(wP, 0)
                     zone2 = np.append(zone2, 0)
                     zone4 = np.append(zone4, 0)
                     gP = np.append(gP, netLoad[time] + cP[-1])
                     socRegister = np.append(socRegister,socRegister[-1] + cP[-1] * etaCharging * discretizationStep / 60 / batteryCapacity)
                     time = time + 1
         else: # For a description of the optimization refer to the specific script
             if powerInjection == 2: # Here you check that you are not injecting power into the grid
                 if(socDeparture > EVMaximumV2XEnergyLevel):
                     [cPopt, dPopt, cSopt, dSopt, gPopt, zone2State, zone4State, wPopt] = optimizerHighSOC(electricityCost[time:], netLoad[time:],
                                                                                     (intervals - time), maxPowerEV, maxPowerEVSE,
                                                                                     minChargingPower, maxPowerGRID,
                                                                                     discretizationStep, batteryCapacity,
                                                                                     etaCharging, etaDischarging,
                                                                                     float(socRegister[-1]),
                                                                                     EVMinimumV2XEnergyLevel, EVMaximumEnergyLevel,
                                                                                     socDeparture,EVMaximumV2XEnergyLevel,optimizationFunction,batteryDegradation)
                     break
                 else:
                    [cPopt, dPopt, cSopt, dSopt, gPopt, zone2State,wPopt] = optimizerBase(electricityCost[time:], netLoad[time:], (intervals - time), maxPowerEV, maxPowerEVSE, minChargingPower, maxPowerGRID, discretizationStep, batteryCapacity, etaCharging, etaDischarging, float(socRegister[-1]), EVMinimumV2XEnergyLevel, EVMaximumV2XEnergyLevel, socDeparture, optimizationFunction,batteryDegradation)
                    break
             else: # here you inject power into the grid
                 gPs = np.zeros(time)
                 if marketPriceflag == 1:
                    sellingPrice = electricityCost
                 else:
                    sellingPrice = np.ones(electricityCost.shape) * root.fit

                 if(socDeparture > EVMaximumV2XEnergyLevel):
                     [cPopt, dPopt, cSopt, dSopt, gPopt, gPsopt, zone2State, zone4State] = optimizerHighSOCV2G(electricityCost[time:],sellingPrice, netLoad[time:],
                                                                                     (intervals - time), maxPowerEV, maxPowerEVSE,
                                                                                     minChargingPower, maxPowerGRID,
                                                                                     discretizationStep, batteryCapacity,
                                                                                     etaCharging, etaDischarging,
                                                                                     float(socRegister[-1]),
                                                                                     EVMinimumV2XEnergyLevel, EVMaximumEnergyLevel,
                                                                                     socDeparture,EVMaximumV2XEnergyLevel,optimizationFunction,batteryDegradation)

                     gPs = np.append(gPs, gPsopt)
                     break
                 else: # here you inject power into the grid
                    [cPopt, dPopt, cSopt, dSopt, gPopt, gPsopt, zone2State] = optimizerBaseV2G(electricityCost[time:], sellingPrice, netLoad[time:], (intervals - time), maxPowerEV, maxPowerEVSE, minChargingPower, maxPowerGRID, discretizationStep, batteryCapacity, etaCharging, etaDischarging, float(socRegister[-1]), EVMinimumV2XEnergyLevel, EVMaximumV2XEnergyLevel, socDeparture, optimizationFunction,batteryDegradation)
                    gPs = np.append(gPs, gPsopt)
                    break





    cP = np.append(cP, cPopt)
    cS = np.append(cS, cSopt)
    dS = np.append(dS, dSopt)
    dP = np.append(dP, dPopt)
    gP = np.append(gP, gPopt)
    if powerInjection == 2:
        wP = np.append(wP,wPopt)
    zone2 = np.append(zone2, zone2State)
    zone4 = np.append(zone4, zone4State)
    socRegister = fix_soc(cP, dP, socRegister[0], batteryCapacity, etaCharging, etaDischarging, discretizationStep,
                          intervals)



    if optimizationFunction == 1:
        objective = "Cost minimization"
    elif optimizationFunction == 2:
        objective = "Maximise self consumption"
    else:
        objective = "Peak shaving"


    ##########################################
    ######### PLOTTING SECTION ################
    ###########################################

    if powerInjection == 2:
        root.tree.insert("", "end",text = root.sim_count,  values=(socArrival, socDeparture, hours,maxPowerGRID, round(batteryDegradation,3) ,round(np.inner(gP,electricityCost)*discretizationStep/60,3),round(sum(loadHouseForecast)* discretizationStep / 60,3), round(sum(cP + dP)* discretizationStep / 60,3),round(sum(gP)*discretizationStep/60, 3)))
    else:
        root.tree.insert("", "end", text=root.sim_count, values=(
        socArrival, socDeparture, hours, maxPowerGRID, batteryDegradation,
        round(np.inner(gP, electricityCost) * discretizationStep / 60 - np.inner(gPs, sellingPrice) * discretizationStep / 60, 3) ,round(sum(loadHouseForecast)* discretizationStep / 60,3), round(sum(cP + dP)* discretizationStep / 60,3),round(sum(gP) * discretizationStep / 60, 3)))


    def graph5():
        fig, ax1 = plt.subplots()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d /%m - %H:%M'))
        plt.gcf().autofmt_xdate()
        ax1.plot(days, cS, 'o-')
        ax1.plot(days, dS,'o-')
        ax2 = ax1.twinx()
        ax2.bar(days, cP, alpha=0.5, width=0.002)
        ax2.bar(days, -dP, alpha=0.5, width=0.002)
        ax2.set(ylabel='Exchanged Power [kW]')
        ax1.set(ylabel='Charging State [-]')
        ax1.set_yticks([0,1])
        fig.legend(["x_ch ", "x_dh ","p_ch","p_dh"])
        plt.show()

    def graph4():
        fig, ax1 = plt.subplots()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d /%m - %H:%M'))
        plt.gcf().autofmt_xdate()
        ax1.plot(days, zone2)
        if (socDeparture > EVMaximumV2XEnergyLevel):
            ax1.plot(days, zone4)
            fig.legend(["Zone 2", "SOC"])
        ax2 = ax1.twinx()
        ax2.plot(days, socRegister, "k*-")
        ax1.set(ylabel='Zone State [-]')
        ax1.set_yticks([0, 1])
        ax2.set(ylabel= 'State of Charge [-]' )
        fig.legend(["Zone 2", "Zone 4","SOC"])
        plt.show()


    def graph3():
        fig, ax1 = plt.subplots()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d /%m - %H:%M'))
        plt.gcf().autofmt_xdate()
        ax1.plot(days,loadHouseForecast, 'o--')
        #ax1.plot(days, renewableProduction, 'o--', color= 'orange')
        #ax1.plot(days2plot, repeat_vector_elements(loadHouseReal, discretizationStep))
        ax1.fill_between(days, renewableProduction, facecolor='yellow', alpha=0.5,edgecolor = 'orange')
        ax2 = ax1.twinx()
        ax2.plot(days, electricityCost, 'r--', alpha=0.7)
        plt.gcf().autofmt_xdate()
        plt.gcf().autofmt_xdate()
        ax1.set(ylabel='Power [kW]')
        fig.legend(["Load Home  [kW]",  "Renewable production [kW]", "Electricity price  [€/kWh]"],loc='upper left', fontsize= 16,)
        ax1.set(ylabel='Electric load / generation  [kW]')
        ax1.set(xlabel='Time')
        ax2.set(ylabel='Electricity price  [€/kWh]')

        plt.show()

    def graph1():
        fig, ax1 = plt.subplots()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d /%m - %H:%M'))
        ax1.xaxis.set_major_locator(mdates.DayLocator())
        ax1.plot(days, socRegister,"k*-")
        ax2 = ax1.twinx()
        ax2.bar(days, cP, alpha=0.5,width = 0.002)
        ax2.bar(days, -dP ,alpha=0.5,width = 0.002)
        ax3 = ax1.twinx()
     #   ax3.plot(days, electricityCost,'r--', alpha=0.7)
        ax3.spines.right.set_position(("axes", 2))
        ax1.set(ylabel='SOC [-]')
        ax2.set(ylabel='Exchanged power with the battery  [kW]')
        fig.legend(["SoC","Pch","Pdh"])
        ax1.grid()
        ax1.margins(0)  # remove default margins (matplotlib verision 2+)
        ax1.axhspan(minimumEnergyLevel, EVMinimumV2XEnergyLevel, facecolor='mistyrose')
        ax1.axhspan(EVMinimumV2XEnergyLevel, EVMaximumV2XEnergyLevel, facecolor='lemonchiffon')
        plt.xticks(rotation=45)
        ax1.axhspan(EVMaximumV2XEnergyLevel, EVMaximumEnergyLevel, facecolor='lightcyan')
        plt.show()

    def graph2():
        fig, ax1 = plt.subplots()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d /%m - %H:%M'))
        plt.gcf().autofmt_xdate()
        ax1.bar(days, cP, alpha=0.5,width = 0.002)
        ax1.bar(days, -dP, alpha=0.5,width = 0.002)
        ax2 = ax1.twinx()
        ax2.plot(days, electricityCost, alpha=0.2)
        ax1.plot(days, netLoad, "ko--", alpha=0.5)
        ax1.plot(days, gP, "b*--")


        if powerInjection == 1:
            ax1.plot(days, -gPs,"r*--")
            fig.legend(["House Load", "Grid bought power", "Grid sold power", "Pch", "Pdh", "Elec cost"])
        else:
            ax1.plot(days, wP, "r--")
            fig.legend(["House Load", "Grid bought power", "Wasted Power", "Pch", "Pdh", "Elec cost"])

        ax1.set(ylabel='Power [-]')
        ax2.set(ylabel='Electricity cost [SEK/kWh]')

        plt.show()

    if powerInjection == 1:
        inj = "enabled"
    else:
        inj = "disabled"
    if marketPriceflag ==1:
        pp = "Market price"
    else:
        pp = "Feed in tariff of " + str(root.fit)

    if powerInjection == 1:
        label1 = tk.Label(root, text="Sim n: " + str(root.sim_count) + "  Obj: " + str(
            objective) + " with power injection " + str(inj) + " @ " + str(pp))
    else:
        label1 = tk.Label(root, text="Sim n: " + str(root.sim_count) + "  Obj: " + str(
            objective) + " with power injection " + str(inj))

    label1.grid(row=root.sim_count + 9, column=0)
    graph3_button = tk.Button(root, text="Input profiles", command=graph3)
    graph3_button.grid(row=root.sim_count+ 9, column=1)
    graph1_button = tk.Button(root, text="SOC and Electricty Price trends", command=graph1)
    graph1_button.grid(row=root.sim_count+ 9, column=2)
    graph2_button = tk.Button(root, text="Power profiles", command=graph2)
    graph2_button.grid(row=root.sim_count+ 9, column=3)




