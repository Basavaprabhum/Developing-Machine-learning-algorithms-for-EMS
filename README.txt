The folder is composed of multiple files, all necessary to run the model. 
Note that all these files must be stay in the same folder to be successufully run. 

In order to understand here is reported a list of the operations and characteristics of 
each script: 
1) gui interface 

To practically run the model use:  gui interface 
This will result in the opening of the GUI. There you can tune the parameter as you prefer and 
click on the button Simulate. Simulate is connected with the guimain script that contains the core of the optimization. 
However, to perform operations only gui interface must be triggered. 

2) gui main 
In gui main there is the core of the algorithm. However gui main use other function contained in the remaining 
scripts in the folder.   
Gui main is structered as follow: 
	a) Input parameter: some input parameter can be defined here such as the maximum power admitted for the battery or
	   EVSE or the efficiency. All the input that are not chosen by the user through the GUI are here defined. 
	b) Optimization section: in here you have an if that will understand according to user choices which optimization
	   function to use. For details on optimization function are reported below at point 4) 
	c) Plotting section

3) batteryDegradationModel contains the function to calculate the degradation coefficient. It is provided with 
the experimental coefficients coming from the literature. If interest refer to the thesis. 

4) optimizer 

This file is structed as 4 subfunction: 
a) V2H with target SOC below limit 
b) V2H with target SOC above limit
c) V2G with target SOC below limit 
d) V2G with target SOC above limit  

This choices has been done in order to be able to optimize the duration of the optimization. 
In other words, high soc target SOC will result in an addiction decision variable zone4 that monitor the presence in that region. 
However the same structure is present in all the objective functions: 
a) Bounds and input definition
b) Constrains definition 
c) Objective function 

5) load 
In this script yopu import the domestic load definition from a txt file. This is an input 

6)renewable 
In this script yopu import the renewable production from a txt file. This is an input 

7)pricelectricity 
In this script yopu import the price from Nordpool.
Data are inserted singularly and manually for four days in order to simulate multiple condition. Commenting lines it is possible to 
tune the desired price profile.  

8) addiction 
This is another script containing all the remaining function that are used in the model. These function are not very 
important and are just aimed at arranging, quantifying and managing data. 
The most important is the function fixsoc. This function simply takes in input the charging/ discharging power, a soc level 
and a interval in minutes and calculates the resulting soc of the vehicle. 




 