import csv
import os
import numpy as np
import matplotlib.pyplot as plt

PATH = os.path.abspath(os.getcwd()) 
testing_houses = 100
anntype = "FFN"

datapath = PATH + os.sep + f"tested on {testing_houses} houses"
csv_datapath =  datapath +os.sep+ "errors.csv"

with open(csv_datapath) as f:
    reader = csv.reader(f)
    #next(reader)
    data = []
    data_single_category = [[] for i in range(6)]
    for row in reader:
        temp_data = [abs(float(cell)) for cell in row]            
        data += temp_data
        for i in range(6):
            data_single_category[i].append(temp_data[i])
    
    
data = np.array(data)
f1 = plt.figure()
hx, hy, _ = plt.hist(data, bins= 100 + 1, density=True, color="lightblue")
plt.title(f'all categorioes\npdf for {testing_houses}_{anntype}')
plt.xlabel("absolute error value")
plt.ylabel("percentage of occurrence [%]")
plt.grid()
plt.xlim([0, 0.5])
plt.ylim([0, 35])
path_to_save = datapath +os.sep+ f"all_categorioes_pdf_on_{testing_houses}_{anntype}.png"
plt.savefig(path_to_save, bbox_inches='tight')
plt.close()
#f2 = plt.figure()
dx = hy[1] - hy[0]
F1 = np.cumsum(hx)*dx
print(len(F1))
plt.plot(hy[1:], F1)
plt.title(f'all categorioes\ncdf for {testing_houses}_{anntype}')
plt.xlabel("absolute error value")
plt.ylabel("fractions of errors smaller than corresponding value")
plt.grid()
plt.xlim([0, 0.5])
plt.ylim([0-0.1, 1+0.1])
#plt.show()
path_to_save = datapath +os.sep+f"all_categorioes_cdf_on_{testing_houses}_{anntype}.png"
plt.savefig(path_to_save, bbox_inches='tight')
plt.close()
for i in range(6):
    data = np.array(data_single_category[i])
    #f3 = plt.figure()
    hx, hy, _ = plt.hist(data, bins= 100 + 1, density=True, color="lightblue")
    plt.title(f'category {i}\npdf for {testing_houses}_{anntype}')
    plt.xlabel("absolute error value")
    plt.ylabel("percentage of occurrence [%]")
    plt.grid()
    plt.xlim([0, 0.5])
    plt.ylim([0, 35])
    
    path_to_save = datapath +os.sep+f"category {i}_pdf_on_{testing_houses}_{anntype}.png"
    plt.savefig(path_to_save, bbox_inches='tight')
    plt.close()
    #plt.close()
    #f4 = plt.figure()
    dx = hy[1] - hy[0]
    F1 = np.cumsum(hx)*dx
    print(len(F1))
    plt.plot(hy[1:], F1)
    plt.title(f'category {i}\ncdf for {testing_houses}_{anntype}')
    plt.xlabel("absolute error value")
    plt.ylabel("percentage of occurrence [%]")
    plt.grid()
    plt.xlim([0, 0.5])
    plt.ylim([0-0.1, 1+0.1])
    #plt.show()
    path_to_save = datapath +os.sep+f"category {i}_cdf_on_{testing_houses}_{anntype}.png"
    plt.xlabel("abolte error value")
    plt.ylabel("fractions of errors smaller than corresponding value")
    plt.savefig(path_to_save, bbox_inches='tight')
    plt.close()