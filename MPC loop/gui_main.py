# -------------------------------------------------------------------------------
# ----------------------Import section--------------------------------------------
# -------------------------------------------------------------------------------
from load import *
from renewable import *
from costElectricity import *
from optimizer import *
from addiction import *
import matplotlib.dates as mdates
import datetime as dt
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import tkinter as tk
from keras.models import load_model
from Ele_prediction_implementation_in_EMS import LoadPrediction, load_and_prepare_data_ele
from PV_prediction_implementation_in_EMS import PVPrediction, load_and_prepare_data
import pandas as pd

# -------------------------------------------------------------------------------
# ----------------------Input section--------------------------------------------
# -------------------------------------------------------------------------------
def main4gui(root):
    batteryCapacity = root.batterycap  #XC40
    batteryDegradation = root.deg  #€/kWh

    socArrival = root.arrival_soc
    socDeparture = root.departure_soc


    hours_ = root.departure_time

    maxPowerEV = 10
    maxPowerEVSE = 10

    maxPowerGRID = root.peak
    minChargingPower = 4

    samplingFrequency = 60 #in minutes
    discretizationStep = 30 # in minutes
    hours, minute_ = map(int, hours_.split(':'))


    #    now_rounded = dt.datetime.now() #considering the realtime datetime
    #now_rounded = dt.datetime(2023, 6, 22, 9, 00, 00) #for the particular date and time
    start_datetime = '2023-04-01 14:00:00'
    start_datetime_obj = datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')
    start_time_obj = datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')

    # Add 30 minutes to the start_datetime_obj
    start_time_obj += timedelta(minutes=30)


    intervals = int((hours * 60 + minute_) / discretizationStep)
    final_obj = start_datetime_obj + timedelta(minutes=(intervals * discretizationStep))
    final = final_obj.strftime('%Y-%m-%d %H:%M:%S')
    n_future= hours
    n_past=48
    file_path_PV = r"C:\Users\basav\PycharmProjects\pythonProject1\finalCode\MPC loop"
    preprocessed_data, scaler, pv_power_index = load_and_prepare_data(file_path_PV)
    model_folder_path_PV = r"C:\Users\basav\PycharmProjects\pythonProject1\finalCode\MPC loop\Models\PV_models"
    pv_predictor = PVPrediction(preprocessed_data, start_datetime, final, n_past, n_future, model_folder_path_PV)

    file_path_Ele = r"C:\Users\basav\PycharmProjects\pythonProject1\finalCode\MPC loop"
    preprocessed_data1, scaler, Ele_power_index = load_and_prepare_data_ele(file_path_Ele)
    model_folder_path_Ele = r"C:\Users\basav\PycharmProjects\pythonProject1\finalCode\MPC loop\Models\Ele_consumption"

    # Create an instance of PVPrediction class with the preprocessed data and other parameters
    Load_predictor = LoadPrediction(preprocessed_data1, start_datetime, final, n_past, n_future,model_folder_path_Ele)

#setting the samplingEvents to "hours" mainly because input predictions are done evey 1 hour
    samplingEvents = hours                    #int((hours * 60 + minute_) / samplingFrequency)
    datetime_values = pd.date_range(start=start_datetime, periods=intervals, freq='30min')


    days = mdates.drange(start_datetime_obj, final_obj, dt.timedelta(minutes=discretizationStep))
 #   days2plot = mdates.drange(start_datetime_obj, final_obj, dt.timedelta(minutes=1))
    if int(len(days)) > intervals:
        diff = int(len(days)) - intervals
        days = days[:-diff]
 #       days2plot = days2plot[:-1]

    minimumEnergyLevel = root.min_soc_level
    EVMaximumEnergyLevel = root.max_soc_level
    EVMinimumV2XEnergyLevel = root.min_soc_v2x
    EVMaximumV2XEnergyLevel = root.max_soc_v2x


    optimizationFunction = root.optimizationFunction
#    accuracy = 1
#    loadHouseForecast , loadHouseReal = loadArray(days, now_rounded, accuracy)
#    renewableProduction = main_function_for_PV_prediction(hours)
    #netLoad = loadHouseForecast - renewableProduction  #if positive is to fulfill in negative in excess
    electricityCost = costElectricity(start_datetime_obj, intervals, discretizationStep)
    etaCharging = 0.97
    etaDischarging = 0.97
    #availablePower = maxPowerGRID - netLoad
    # -------------------------------------------------------------------------------
    # ----------------------Optimizer section----------------------------------------
    # -------------------------------------------------------------------------------

    chargingPower = np.zeros(intervals)
    dischargingPower = np.zeros(intervals)
    gridPower = np.zeros(intervals)
    wastedPower = np.zeros(intervals)
    socRegisterdef = np.zeros((samplingEvents, intervals))

    for sampling in range(samplingEvents):
        time = 0
        if sampling == 0:
            socRegister = np.array([socArrival])
        else:
            socRegister = np.array([socRegisterdef[sampling-1][0 + int(sampling * samplingFrequency / discretizationStep) - 1 ]])

        cP = []
        cS = []
        dP = []
        dS = []
        gP = []
        zone2 = []
        zone4 = []
        zone4State = []
        wP = []
        predictionHorizon = int((intervals * discretizationStep - sampling * samplingFrequency) / discretizationStep)
        if sampling % 1==0:     #it is set to 1 because the sampling frequency is same as prediction time step
            Renewable_energy_prediction= pv_predictor.run_once(n_future,hours)
            Load_energy_prediction= Load_predictor.run_once(n_future,hours)
            n_future -= 1
            Load_energy_prediction_array = np.array(Load_energy_prediction)
            Load_energy_prediction_array_val= Load_energy_prediction_array[int(sampling * samplingFrequency/discretizationStep):]
            Renewable_energy_prediction_array = np.array(Renewable_energy_prediction)
            Renewable_energy_prediction_array_val= Renewable_energy_prediction_array [int(sampling * samplingFrequency/discretizationStep):]
            netLoad = Load_energy_prediction_array - Renewable_energy_prediction_array

        availablePower = maxPowerGRID - netLoad
        availablePowertemp = availablePower[int(sampling * samplingFrequency/discretizationStep):]
        netLoadtemp = netLoad[int(sampling * samplingFrequency/discretizationStep):]
        electricityCosttemp = electricityCost[int(sampling * samplingFrequency/discretizationStep):]

        while time in range(predictionHorizon):
             if socRegister[-1] < minimumEnergyLevel:
                 if availablePower[time] >= minChargingPower:
                    cP = np.append(cP, availablePowertemp[time])
                    cS = np.append(cS, 1)
                    dS = np.append(dS, 0)
                    dP = np.append(dP, 0)
                    gP = np.append(gP, netLoadtemp[time] + cP[-1])
                    zone2 = np.append(zone2,0)
                    zone4 = np.append(zone4, 0)
                    wP = np.append(wP, 0)
                    socRegister = np.append(socRegister, socRegister[-1] + cP[-1] * etaCharging * discretizationStep/60 /batteryCapacity)
                    time = time + 1
                 else:
                     cP = np.append(cP, 0)
                     cS = np.append(cS, 0)
                     dS = np.append(dS, 0)
                     dP = np.append(dP, 0)
                     wP = np.append(wP, 0)
                     zone2 = np.append(zone2, 0)
                     zone4 = np.append(zone4, 0)
                     gP = np.append(gP, netLoadtemp[time] + cP[-1])
                     socRegister = np.append(socRegister,socRegister[-1] + cP[-1] * etaCharging * discretizationStep / 60 / batteryCapacity)
                     time = time + 1
             elif(socDeparture > EVMaximumV2XEnergyLevel):
                [cPopt, dPopt, cSopt, dSopt, gPopt, zone2State, zone4State, wPopt] = optimizerHighSOC(electricityCosttemp[time:], netLoadtemp[time:],
                                                                                 (predictionHorizon - time), maxPowerEV, maxPowerEVSE,
                                                                                 minChargingPower, maxPowerGRID,
                                                                                 discretizationStep, batteryCapacity,
                                                                                 etaCharging, etaDischarging,
                                                                                 float(socRegister[-1]),
                                                                                 EVMinimumV2XEnergyLevel, EVMaximumEnergyLevel,
                                                                                 socDeparture,EVMaximumV2XEnergyLevel,optimizationFunction)
                break
             else:
                [cPopt, dPopt, cSopt, dSopt, gPopt, zone2State, wPopt] = optimizerBase(electricityCosttemp[time:],
                                                                                        netLoadtemp[time:],
                                                                                        (predictionHorizon - time),
                                                                                        maxPowerEV, maxPowerEVSE,
                                                                                        minChargingPower, maxPowerGRID,
                                                                                        discretizationStep,
                                                                                        batteryCapacity, etaCharging,
                                                                                        etaDischarging,
                                                                                        float(socRegister[-1]),
                                                                                        EVMinimumV2XEnergyLevel,
                                                                                        EVMaximumV2XEnergyLevel,
                                                                                        socDeparture,
                                                                                        optimizationFunction,
                                                                                        batteryDegradation)
                break
        cP = np.append(cP, cPopt)
        cS = np.append(cS, cSopt)
        dS = np.append(dS, dSopt)
        dP = np.append(dP, dPopt)
        gP = np.append(gP, gPopt)
        wP = np.append(wP,wPopt)
        zone2 = np.append(zone2, zone2State)
        zone4 = np.append(zone4, zone4State)
        socRegister = fix_soc(cP, dP, socRegister[0], batteryCapacity, etaCharging, etaDischarging, discretizationStep,predictionHorizon)

#        chargingPower = np.append(chargingPower,cP[:int(samplingFrequency/discretizationStep)])
#        dischargingPower = np.append(dischargingPower, dP[:int(samplingFrequency / discretizationStep)])
        socRegisterdef[sampling,int(sampling * samplingFrequency/ discretizationStep):] = socRegister
#        socRegister = np.append(socRegister, socRegistertemp[:int(samplingFrequency / discretizationStep)])
#        gridPower = np.append(gridPower, gP[:int(samplingFrequency / discretizationStep)])
#        wastedPower = np.append(wastedPower, wP[:int(samplingFrequency / discretizationStep)])
        current_cP_length = len(cP)
        current_dP_length = len(cP)
        current_gP_length = len(cP)
        current_wP_length = len(cP)
        chargingPower[-current_cP_length:] = cP
        dischargingPower[-current_dP_length:] = dP
        gridPower[-current_gP_length:] = gP
        wastedPower[-current_wP_length:] = wP

        #chargingPower = np.append(chargingPower, cP)
        #dischargingPower = np.append(dischargingPower, dP)
#        socRegister = np.append(socRegister, socRegister[-1] + cP[-1] * etaCharging * discretizationStep / 60 / batteryCapacity)
        #gridPower = np.append(gridPower, gP)
        #wastedPower = np.append(wastedPower, wP)

# plotting the Input energy values
        fig, ax1 = plt.subplots(figsize=(12, 8))
        Renewable_values = [None] * intervals
        load_values= [None] * intervals
        datetime_values = pd.date_range(start=start_time_obj, periods=intervals, freq='30min')

        Renewable_values[int(sampling * samplingFrequency/discretizationStep):] = Renewable_energy_prediction_array_val
        load_values[int(sampling * samplingFrequency / discretizationStep):] = Load_energy_prediction_array_val
        ax1.plot(datetime_values, Renewable_values, "ko--")
        ax1.plot(datetime_values, load_values, "r*--")
        fig.legend(['Renewable Energy', 'Load Energy'])
        ax1.set_ylabel('Energy(kWh)')
        # Explicitly set the x-axis limits
        plt.xlim([datetime_values[0], datetime_values[-1]])

        plt.xticks(rotation=45)
        plt.show()



        #plotting the values of graph1
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d /%m - %H:%M'))
        ax1.xaxis.set_major_locator(mdates.DayLocator())
#        ax1.plot(days[:int(len(days)-len(socRegister))], socRegister, "k*-")
        ax2 = ax1.twinx()
        ax2.bar(datetime_values, chargingPower, alpha=0.5, width=0.002)
        ax2.bar(datetime_values, -dischargingPower, alpha=0.5, width=0.002)
        ax3 = ax1.twinx()
        ax3.plot(days, electricityCost, 'r--', alpha=0.7)
        ax3.spines.right.set_position(("axes", 2))
        ax1.set(ylabel='SOC [-]')
        ax2.set(ylabel='Battery charging power [kW]')
        fig.legend([ "Pch", "Pdh", "Elec cost"])
        ax1.grid()
        ax1.margins(0)  # remove default margins (matplotlib verision 2+)
        ax1.axhspan(minimumEnergyLevel, EVMinimumV2XEnergyLevel, facecolor='purple', alpha=0.05)
        ax1.axhspan(EVMinimumV2XEnergyLevel, EVMaximumV2XEnergyLevel, facecolor='yellow', alpha=0.05)
        plt.xticks(rotation=45)
        ax1.axhspan(EVMaximumV2XEnergyLevel, EVMaximumEnergyLevel, facecolor='salmon', alpha=0.1)
        plt.show()

        #ploting the values of graph2
        fig, ax1 = plt.subplots(figsize=(12, 8))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d /%m - %H:%M'))
        plt.gcf().autofmt_xdate()
        ax1.bar(days, chargingPower, alpha=0.5, width=0.002)
        ax1.bar(days, -dischargingPower, alpha=0.5, width=0.002)
        ax2 = ax1.twinx()
        ax2.plot(days, electricityCost, alpha=0.2)
        ax1.plot(days, netLoad, "ko--", alpha=0.5)
        ax1.plot(days, gridPower, "k*--")
        ax1.plot(days, wastedPower, "r--")
        ax1.set(ylabel='Power [-]')
        ax2.set(ylabel='Electricity cost [SEK/kWh]')
        fig.legend(["House Load", "Total load", "Wasted Power", "Pch", "Pdh", "Elec cost"])
        plt.show()





    socRegister = fix_soc(chargingPower,dischargingPower,socArrival,batteryCapacity,etaCharging, etaDischarging, discretizationStep,intervals ) #qui sono fuori da tutti i for
    print(optimizationFunction)
    if optimizationFunction == 1:
        objective = "Cost minimization"
    elif optimizationFunction == 2:
        objective = "Maximise self consumption"
    else:
        objective = "Peak shaving"
    root.tree.insert("", "end",text = root.sim_count,  values=(socArrival, socDeparture, hours,maxPowerGRID, batteryDegradation ,round(np.inner(chargingPower,electricityCost)*discretizationStep/60,3), round(sum(chargingPower)*discretizationStep/60, 3)))

    def graph4():
        fig, ax1 = plt.subplots()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d /%m - %H:%M'))
        plt.gcf().autofmt_xdate()
        ax1.plot(days, zone2)
        if (socDeparture > EVMaximumV2XEnergyLevel):
            ax1.plot(days, zone4)
            fig.legend(["Zone 2", "SOC"])
        ax1.plot(days, socRegister, "k*-")
        ax1.set(ylabel='Zone State [-]')
        fig.legend(["Zone 2", "Zone 4","SOC"])
        plt.show()


    def graph3():
        fig, ax1 = plt.subplots(figsize=(12, 8))

        datetime_values = pd.date_range(start=start_time_obj, periods=intervals, freq='30min')

        ax1.plot(datetime_values, Renewable_energy_prediction_array, "ko--" )
        ax1.plot(datetime_values, Load_energy_prediction_array, "r*--")
        fig.legend(['Renewable Energy', 'Load Energy'])
        ax1.set_ylabel('Energy(kWh)')
        # Explicitly set the x-axis limits
#plt.xlim([datetime_values[0], datetime_values[-1]])

        plt.xticks(rotation=45)
        plt.show()


    def graph1():
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d /%m - %H:%M'))
        ax1.xaxis.set_major_locator(mdates.DayLocator())
        ax1.plot(datetime_values, socRegister,"k*-")
        ax2 = ax1.twinx()
        ax2.bar(datetime_values, chargingPower, alpha=0.5,width = 0.002)
        ax2.bar(datetime_values, -dischargingPower ,alpha=0.5,width = 0.002)
        ax3 = ax1.twinx()
    #    ax3.plot(days, electricityCost,'r--', alpha=0.7)
        ax3.spines.right.set_position(("axes", 2))
        ax1.set(ylabel='SOC [-]')
        ax2.set(ylabel='Battery charging power [kW]')
        fig.legend(["SoC","Pch","Pdh"])
        ax1.grid()
        ax1.margins(0)  # remove default margins (matplotlib verision 2+)
        ax1.axhspan(minimumEnergyLevel, EVMinimumV2XEnergyLevel, facecolor='purple', alpha=0.05)
        ax1.axhspan(EVMinimumV2XEnergyLevel, EVMaximumV2XEnergyLevel, facecolor='yellow', alpha=0.05)
        plt.xticks(rotation=45)
        ax1.axhspan(EVMaximumV2XEnergyLevel, EVMaximumEnergyLevel, facecolor='salmon', alpha=0.1)
        plt.show()

    def graph2():
        fig, ax1 = plt.subplots(figsize=(12, 12))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d /%m - %H:%M'))
        plt.gcf().autofmt_xdate()
        ax1.bar(days, chargingPower, alpha=0.5,width = 0.002)
        ax1.bar(days, -dischargingPower, alpha=0.5,width = 0.002)
        ax2 = ax1.twinx()
        ax2.plot(days, electricityCost, alpha=0.2)
        ax1.plot(days, netLoad, "ko--", alpha=0.5)
        ax1.plot(days, gridPower, "k*--")
        ax1.plot(days, wastedPower, "r--")
        ax1.set(ylabel='Power [-]')
        ax2.set(ylabel='Electricity cost [SEK/kWh]')
        fig.legend(["House Load","Total load","Wasted Power", "Pch","Pdh","Elec cost"])
        plt.show()


    label1 = tk.Label(root, text="Sim n: " + str(root.sim_count) + "  Obj: " + str(objective))
    label1.grid(row=root.sim_count + 9, column=0)
    graph3_button = tk.Button(root, text="Input profiles", command=graph3)
    graph3_button.grid(row=root.sim_count + 9, column=1)
    graph1_button = tk.Button(root, text="SOC and Electricty Price trends", command=graph1)
    graph1_button.grid(row=root.sim_count+ 9, column=2)
    graph2_button = tk.Button(root, text="Power profiles", command=graph2)
    graph2_button.grid(row=root.sim_count+ 9, column=3)


