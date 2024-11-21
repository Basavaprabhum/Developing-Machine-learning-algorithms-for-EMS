import tkinter as tk
import tkinter.font as TkFont
from tkinter import ttk
from gui_main import *
from costElectricity import *
root = tk.Tk()

root.sim_count = 0
[priceMax , priceMin] = guiPrice()
priceLimits = np.array(priceMax, priceMin)


def change_color(*args):
    try:
        value = float(FIT_entry.get())
        [priceMax, priceMin] = guiPrice()
        value = (value - priceMin)/(priceMax - priceMin)
        if value < 0.1:
            FIT_ref_entry.configure(bg="red3")
        elif value < 0.2:
            FIT_ref_entry.configure(bg="red2")
        elif value < 0.3:
            FIT_ref_entry.configure(bg="red1")
        elif value < 0.4:
            FIT_ref_entry.configure(bg="yellow4")
        elif value < 0.5:
            FIT_ref_entry.configure(bg="yellow3")
        elif value < 0.6:
            FIT_ref_entry.configure(bg="yellow2")
        elif value < 0.7:
            FIT_ref_entry.configure(bg="yellow1")
        elif value < 0.8:
            FIT_ref_entry.configure(bg="green4")
        elif value < 0.9:
            FIT_ref_entry.configure(bg="green2")
        elif value > 1:
            FIT_ref_entry.configure(bg="black")
        else:
            FIT_ref_entry.configure(bg="green1")
    except ValueError:
        pass



def delete_row(root, tree):

    selection = tree.selection()  # Get the selected item(s) from the treeview
    for item in selection:
        item_id = tree.item(selection)['text']
        #item_id = int(item_id[-1])# Get the index of the selected item
        tree.delete(item)  # Delete the selected item from the treeview
        widgets = [w for w in root.grid_slaves() if int(w.grid_info()['row']) == item_id + 9]

        # Remove each widget from the grid
        for widget in widgets:
            widget.grid_forget()

def simulate():

    root.sim_count = root.sim_count +1
    root.departure_time = (departure_time_entry.get())
    root.departure_soc = float(departure_soc_entry.get())
    root.arrival_soc = float(arrival_soc_entry.get())
    root.min_soc_level = float(min_soc_level_entry.get())
    root.min_soc_v2x = float(min_soc_v2x_entry.get())
    root.max_soc_v2x = float(max_soc_v2x_entry.get())
    root.max_soc_level = float(max_soc_level_entry.get())
    root.optimizationFunction = int(radio_var.get())
    root.injectionFlag = int(radio2_var.get())
    root.priceflag = int(radio3_var.get())
    root.fit = float(FIT_entry.get())
    root.batterycap = float(batterycap_entry.get())
    root.peak = float(peak_entry.get())
    root.deg = float(deg_entry.get())
    main4gui(root)


bold_font = TkFont.Font(family="Arial", size=9, weight="bold")
userinput_label = tk.Label(root, text="User Inputs", font=bold_font)
userinput_label.grid(row=0, column=0, padx=5, pady=5)
i = 1
# Create a label and entry for departure time
departure_time_label = tk.Label(root, text="Time for V2X (hh:mm) :")
departure_time_entry = tk.Entry(root)
departure_time_label.grid(row=0+i, column=0, padx=5, pady=5)
departure_time_entry.grid(row=0+i, column=1, padx=5, pady=5)
departure_time_entry.insert(0, "08:00") ##############################################################################
# Create a label and entry for departure SOC
departure_soc_label = tk.Label(root, text="Departure SOC (-) :")
departure_soc_entry = tk.Entry(root)
departure_soc_label.grid(row=1+i, column=0, padx=5, pady=5)
departure_soc_entry.grid(row=1+i, column=1, padx=5, pady=5)
departure_soc_entry.insert(0, "0.5") ##############################################################################

#OBJECTIVE FUNCTION
# Create a label for the radio button
radio_label = tk.Label(root, text="Select an objective function :", font=bold_font)
radio_label.grid(row=2+i, column=0, padx=5, pady=5)

# Create a radio button with two options
radio_var = tk.StringVar(value=1)
radio_button_1 = tk.Radiobutton(root, text="Cost minimization", variable=radio_var, value=1)
radio_button_2 = tk.Radiobutton(root, text="Maximize self consumption", variable=radio_var, value=2)

radio_button_1.grid(row=3+i, column=0, padx=5, pady=5)
radio_button_2.grid(row=3+i, column=1, padx=5, pady=5)

#POWER INJECTION
# Create a label for the radio button 2
radio2_label = tk.Label(root, text="Enable power injection :", font=bold_font)
radio2_label.grid(row=4+i, column=0, padx=5, pady=5)

# Create a radio button with two options
radio2_var = tk.StringVar(value=2)
radio2_button_1 = tk.Radiobutton(root, text="Yes", variable=radio2_var, value=1)
radio2_button_2 = tk.Radiobutton(root, text="No", variable=radio2_var, value=2)

radio2_button_1.grid(row=5+i, column=0, padx=5, pady=5)
radio2_button_2.grid(row=5+i, column=1, padx=5, pady=5)
#Market tariff
# Create a label for the radio button 2
radio3_label = tk.Label(root, text="Set tariff scheme:  :", font=bold_font)
radio3_label.grid(row=4+i, column=2, padx=5, pady=5)

# Create a radio button with two options
radio3_var = tk.StringVar(value=1)
radio3_button_1 = tk.Radiobutton(root, text="Retail Price", variable=radio3_var, value=1)
radio3_button_2 = tk.Radiobutton(root, text="FIT [€/kWh]", variable=radio3_var, value=2)
FIT_entry = tk.Entry(root)
FIT_ref_entry = tk.Entry(root,width=2)
radio3_button_1.grid(row=5+i, column=2, padx=5, pady=5)
radio3_button_2.grid(row=5+i, column=3, padx=5, pady=5)
FIT_entry.grid(row=5+i, column=4, padx=5, pady=5)
FIT_ref_entry.grid(row=5+i, column=5, padx=5, pady=5)
FIT_entry.insert(0, "0")
FIT_entry.bind('<KeyRelease>', change_color)


#######################################################################################
bold_font = TkFont.Font(family="Arial", size=9, weight="bold")
userinput_label = tk.Label(root, text="EV Inputs", font=bold_font)
userinput_label.grid(row=0, column=2, padx=5, pady=5)
# Create a label and entry for arrival SOC
arrival_soc_label = tk.Label(root, text="Arrival SOC:")
arrival_soc_entry = tk.Entry(root)
arrival_soc_label.grid(row=i, column=2, padx=5, pady=5)
arrival_soc_entry.grid(row=i, column=3, padx=5, pady=5)
arrival_soc_entry.insert(0, "0.15") ##############################################################################
# Create a label and entry for minimum SOC level
min_soc_level_label = tk.Label(root, text="Minimum SOC Level:")
min_soc_level_entry = tk.Entry(root)
min_soc_level_entry.insert(0, "0.2")
min_soc_level_label.grid(row=i+1, column=2, padx=5, pady=5)
min_soc_level_entry.grid(row=i+1, column=3, padx=5, pady=5)

# Create a label and entry for minimum SOC for V2X
min_soc_v2x_label = tk.Label(root, text="Minimum SOC for V2X:")
min_soc_v2x_entry = tk.Entry(root)
min_soc_v2x_entry.insert(0, "0.3")
min_soc_v2x_label.grid(row=i+2, column=2, padx=5, pady=5)
min_soc_v2x_entry.grid(row=i+2, column=3, padx=5, pady=5)

# Create a label and entry for maximum SOC for V2X
max_soc_v2x_label = tk.Label(root, text="Maximum SOC for V2X:")
max_soc_v2x_entry = tk.Entry(root)
max_soc_v2x_entry.insert(0, "0.8")
max_soc_v2x_label.grid(row=i+3, column=2, padx=5, pady=5)
max_soc_v2x_entry.grid(row=i+3, column=3, padx=5, pady=5)

# Create a label and entry for maximum SOC level
max_soc_level_label = tk.Label(root, text="Maximum SOC Level:")
max_soc_level_entry = tk.Entry(root)
max_soc_level_entry.insert(0, "0.97")
max_soc_level_label.grid(row=i, column=4, padx=5, pady=5)
max_soc_level_entry.grid(row=i, column=5, padx=5, pady=5)

# Create a label and entry for maximum SOC level
batterycap_label = tk.Label(root, text="Battery size [kWh]:")
batterycap_entry = tk.Entry(root)
batterycap_entry.insert(0, "69")
batterycap_label.grid(row=i+1, column=4, padx=5, pady=5)
batterycap_entry.grid(row=i+1, column=5, padx=5, pady=5)

# Create a label and entry for peak power
peak_label = tk.Label(root, text="Maximum Peak [kW]:")
peak_entry = tk.Entry(root)
peak_entry.insert(0, "13")
peak_label.grid(row=i+2, column=4, padx=5, pady=5)
peak_entry.grid(row=i+2, column=5, padx=5, pady=5)

# Create a label and entry for peak power
deg_label = tk.Label(root, text="Degradation [€/kWh tp]:")
deg_entry = tk.Entry(root)
deg_entry.insert(0, "0.01")
deg_label.grid(row=i+3, column=4, padx=5, pady=5)
deg_entry.grid(row=i+3, column=5, padx=5, pady=5)


# Create buttons

button2 = tk.Button(root, text="Simulate", command=simulate)
button2.grid(row=i+7, column=1, padx=5, pady=5)

delete_button = tk.Button(root, text="Delete", command=lambda: delete_row(root,root.tree))
delete_button.grid(row=i+7, column=2, padx=5, pady=5)




#  Treeview
root.result = tk.LabelFrame(root, text=" Input Summary", padx=10, pady=1,height=50)
root.result.grid(row=i+6,column = 0, columnspan= 7)
root.tree = ttk.Treeview(root.result, columns=("Result 1", "Result 2", "Result 3", "Result 4", "Result 5", "Result 6", "Result 7", "Result 8", "Result 9"))
root.tree.column("Result 1", width=85, anchor='c')
root.tree.column("Result 2", width=80, anchor='c')
root.tree.column("Result 3", width=100, anchor='c')
root.tree.column("Result 4", width=95, anchor='c')
root.tree.column("Result 5", width=100, anchor='c')
root.tree.column("Result 6", width=100, anchor='c')
root.tree.column("Result 7", width=130, anchor='c')
root.tree.column("Result 8", width=150, anchor='c')
# Add column headings
root.tree.heading("#0", text="Sim")
root.tree.heading("#1", text="Arrival SOC [-]")
root.tree.heading("#2", text="Dep SOC [-]")
root.tree.heading("#3", text="Hours for V2H [h]")
root.tree.heading("#4", text="Peak pow [kW]")
root.tree.heading("#5", text="Degr. Coeff. [€/kWh]")
root.tree.heading("#6", text="Total Cost [€]")
root.tree.heading("#7", text="Domestic Power [kWh]")
root.tree.heading("#8", text="Battery throughput [kWh]")
root.tree.heading("#9", text="Grid power [kWh]")
root.tree.grid(row=0,column= 0)

result_label = tk.Label(root, text="Simulations' results:", font=bold_font)
result_label.grid(row=7+i, column=0, padx=5, pady=5)
# button2 = tk.Button(root, text="Load & PV", command=load_and_pv)
# button2.pack(padx=10, pady=5)
#
# button3 = tk.Button(root, text="SoC and Charging Power", command=soc_and_charging_power)
# button3.pack(padx=10, pady=5)
root.mainloop()