import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})


csv_datapath = os.path.abspath(os.getcwd()) + os.sep + "data"
first = 10081
last = first + 1440
testing_houses = 20
csv_datapath = csv_datapath + os.sep + f"decomposition_new{testing_houses}.csv"
with open(csv_datapath) as f:
    reader = csv.reader(f)
    for i in range(first-1):
        next(reader)
                
    data_read = 0 
    real_values20 = [[] for i in range(6)]
    for row in reader:
        for i in range(6):
            real_values20[i].append(float(row[i]))
        data_read += 1
        if data_read >= (last - first):
            break     




csv_datapath =  os.path.abspath(os.getcwd()) +os.sep+ "results_mse" +os.sep+  "results search for 20" +os.sep+ "place 1 in 2FNN_tr_20_te20_adam" +os.sep+ "tested on 20 houses" +os.sep+ "predictions.csv"
        
with open(csv_datapath) as f:
    reader = csv.reader(f)
    #next(reader)
    pred_single_category20 = [[] for i in range(6)]
    for row in reader:
        temp_data = [abs(float(cell)) for cell in row]          
        for i in range(6):
            pred_single_category20[i].append(temp_data[i])
            
            
            
            
csv_datapath = os.path.abspath(os.getcwd()) + os.sep + "data"
first = 10081
last = first + 1440
testing_houses = 20
csv_datapath = csv_datapath + os.sep + f"total_P_new{testing_houses}.csv"
with open(csv_datapath) as f:
    reader = csv.reader(f)
    for i in range(first-1):
        next(reader)
                
    data_read = 0 
    real_P20 = []
    for row in reader:
        real_P20.append(float(row[0]))
        data_read += 1
        if data_read >= (last - first):
            break     
        
for i in range(1440):
    for j in range(6):
        pred_single_category20[j][i] *= real_P20[i]
        real_values20[j][i] *= real_P20[i]
        
        
print(len(real_P20))

# Create a DataFrame instance
time_steps = range(1440)
x= time_steps

#Draw an area plot for the DataFrame data
#plt.fill_between(time_steps, real_P20, label = "total load")
#plt.fill_between(time_steps, real_values20[0], label = "category 1")
#plt.fill_between(time_steps, real_values20[1], label = "category 2")
#plt.fill_between(time_steps, real_values20[2], label = "category 3")
#plt.fill_between(time_steps, real_values20[3], label = "category 4")
#plt.fill_between(time_steps, real_values20[4], label = "category 5")
#plt.fill_between(time_steps, real_values20[5], label = "category 6")

y = [real_values20[0], real_values20[1], real_values20[2], real_values20[3], real_values20[4], real_values20[5]]
y = y[::-1]
labels = ['category 1','category 2', 'category 3', 'category 4', 'category 5', 'category 6']

labels = labels[::-1]
colors = ['#ce0ff0', '#489af4', '#008a00', '#fbe946', '#120df2', '#ff6f28']
plt.stackplot(x,y, colors= colors, labels=labels)
plt.legend(loc='upper left')

plt.grid()
plt.legend(loc="upper right")
plt.xlabel("time [time steps]")
plt.ylabel("real power [p.u.]")

#plt.show(block=True)
plt.tight_layout()
path_to_save = os.path.abspath(os.getcwd()) +os.sep+ "plots" +os.sep+ f"real_decomposition{testing_houses}.png"
plt.savefig(path_to_save)
#plt.show()
plt.close()


#Draw an area plot for the DataFrame data
#plt.fill_between(time_steps, real_P20, label = "total load")
#plt.fill_between(time_steps, pred_single_category20[0], label = "category 1")
#plt.fill_between(time_steps, pred_single_category20[1], label = "category 2")
#plt.fill_between(time_steps, pred_single_category20[2], label = "category 3")
#plt.fill_between(time_steps, pred_single_category20[3], label = "category 4")
#plt.fill_between(time_steps, pred_single_category20[4], label = "category 5")
#plt.fill_between(time_steps, pred_single_category20[5], label = "category 6")

#y = [real_P20, pred_single_category20[0], pred_single_category20[1], pred_single_category20[2], pred_single_category20[3], pred_single_category20[4], pred_single_category20[5]]
y = [pred_single_category20[0], pred_single_category20[1], pred_single_category20[2], pred_single_category20[3], pred_single_category20[4], pred_single_category20[5]]

y = y[::-1]
#labels = ['total load','category 1','category 2', 'category 3', 'category 4', 'category 5', 'category 6']
labels = ['category 1','category 2', 'category 3', 'category 4', 'category 5', 'category 6']

labels = labels[::-1]
#colors = ['#ce0ff0', '#489af4', '#008a00', '#60fbf3', '#fbe946', '#120df2', '#ff6f28']
colors = ['#ce0ff0', '#489af4', '#008a00', '#fbe946', '#120df2', '#ff6f28']
plt.stackplot(x,y, colors= colors, labels=labels)
plt.legend(loc='upper left')

plt.grid()
plt.legend(loc="upper right")
plt.xlabel("time [time steps]")
plt.ylabel("real power [p.u.]")

#plt.show(block=True)
plt.tight_layout()
path_to_save = os.path.abspath(os.getcwd()) +os.sep+ "plots" +os.sep+ f"predicted_decomposition{testing_houses}.png"
plt.savefig(path_to_save)
#plt.show()
plt.close()
