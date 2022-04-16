import os
import csv
import numpy as np
import matplotlib.pyplot as plt

csv_datapath = os.path.abspath(os.getcwd()) + os.sep + "data"
first = 9601
last = first + 2400
testing_houses = 20
csv_datapath = csv_datapath + os.sep + f"decomposition_new{testing_houses}.csv"
with open(csv_datapath) as f:
    reader = csv.reader(f)
    for i in range(first-1):
        next(reader)
                
    data_read = 0 
    real_values = [[] for i in range(6)]
    for row in reader:
        for i in range(6):
            real_values[i].append(float(row[i]))
        data_read += 1
        if data_read >= (last - first):
            break     


f1 = plt.figure()
for i in range(6):
    data = np.array(real_values[i])
    plt.plot(data)
#plt.title(f'real values')
plt.xlabel("time [time steps]")
plt.ylabel("share of a total demand [p.u.]")
plt.grid()
    #plt.xlim([0, 0.5])
plt.ylim([0, 1.1])
path_to_save = os.path.abspath(os.getcwd()) +os.sep+ f"real_dec_{testing_houses}.png"
plt.savefig(path_to_save, bbox_inches='tight')
plt.close()            