import ann
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


#PATH = os.path.abspath(os.getcwd()) + os.sep + "search_results"
path = os.path.abspath(os.getcwd())
#FOLDERS = ["results_mse", "results_huber", "results_KLD", "results_mape", "results_msle"]

def main():
    
    testing_houses = 100
    newpath = path + os.sep + f"results_LSTMs" + os.sep + f"tested_on_{testing_houses}" +os.sep+ f"LSTM1"
    draw_graphs(datapath = newpath, testing_houses = testing_houses, anntype = "LSTM", loss_function = "mse")
    
    newpath = path + os.sep + f"results_LSTMs" + os.sep + f"tested_on_{testing_houses}" +os.sep+ f"LSTM2"
    draw_graphs(datapath = newpath, testing_houses = testing_houses, anntype = "LSTM", loss_function = "huber")
    
    newpath = path + os.sep + f"results_LSTMs" + os.sep + f"tested_on_{testing_houses}" +os.sep+ f"LSTM3"
    draw_graphs(datapath = newpath, testing_houses = testing_houses, anntype = "LSTM", loss_function = "huber")
    
    newpath = path + os.sep + f"results_LSTMs" + os.sep + f"tested_on_{testing_houses}" +os.sep+ f"LSTM4"
    draw_graphs(datapath = newpath, testing_houses = testing_houses, anntype = "LSTM", loss_function = "mape")    


def draw_graphs(datapath, testing_houses, anntype, loss_function):
    csv_datapath = os.path.abspath(os.getcwd()) + os.sep + "data"
    first = 10081
    last = first + 1440
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
            if data_read > (last - first):
                break     
            
    ###############################
    
    csv_datapath =  datapath +os.sep+ "predictions.csv"
            
    with open(csv_datapath) as f:
        reader = csv.reader(f)
        #next(reader)
        pred_single_category = [[] for i in range(6)]
        for row in reader:
            temp_data = [abs(float(cell)) for cell in row]          
            for i in range(6):
                pred_single_category[i].append(temp_data[i])
    ############################    
     
    data = []
    data_single_category = [[] for i in range(6)]
    real_errors = [[] for i in range(6)]
     
    for i in range(1440):
        for j in range(6):
            difference = real_values[j][i] - pred_single_category[j][i]
            real_errors[j].append(difference)
            data_single_category[j].append(abs(difference))
            data.append(abs(difference))
    
    ###################
    controllable_load_errors = []
    for i in range(1440):
        predicted = 0
        real = 0 
        for j in range(3):
            real += real_values[j][i]
            predicted += pred_single_category[j][i]
        difference = abs(real - predicted)
        controllable_load_errors.append(difference)
    ###################
    
    ############################################
    csv_datapath =  datapath +os.sep+ "abs_errors.csv"
    with open(csv_datapath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        for i in range(len(data_single_category[0])):
            data_to_write = [data_single_category[j][i] for j in range(6)]
            data_to_write.append(controllable_load_errors[i])
            writer.writerow(data_to_write) 
    ############################################
    
    
    #[adapted form https://moonbooks.org/Articles/How-to-calculate-and-plot-a-cumulative-distribution-function-with-matplotlib-in-python-/]
    data = np.array(controllable_load_errors)
    f1 = plt.figure()
    hx, hy, _ = plt.hist(data, bins= 100 + 1, density=True, color="lightblue")
    plt.title(f'controllable load\npdf for {testing_houses}_{anntype}')
    plt.xlabel("absolute error value")
    plt.ylabel("percentage of occurrence [%]")
    plt.grid()
    plt.xlim([0, 0.5])
    plt.ylim([0, 35])
    path_to_save = datapath +os.sep+ f"controllable_load_pdf_on_{testing_houses}_{anntype}.png"
    plt.savefig(path_to_save, bbox_inches='tight')
    plt.close()
    #f2 = plt.figure()
    dx = hy[1] - hy[0]
    F1 = np.cumsum(hx)*dx
    print(len(F1))
    plt.plot(hy[1:], F1)
    plt.title(f'controllable load\ncdf for {testing_houses}_{anntype}')
    plt.xlabel("absolute error value")
    plt.ylabel("fractions of errors smaller than corresponding value")
    plt.grid()
    plt.xlim([0, 0.5])
    plt.ylim([0-0.1, 1+0.1])
    #plt.show()
    path_to_save = datapath +os.sep+f"controllable_load_cdf_on_{testing_houses}_{anntype}.png"
    plt.savefig(path_to_save, bbox_inches='tight')
    plt.close()
    
    
##############################
    f1 = plt.figure()
    for i in range(6):
        data = np.array(real_errors[i])
        plt.plot(data)
    plt.title(f'error values values')
    plt.xlabel("time")
    plt.ylabel("error")
    plt.grid()
    #plt.xlim([0, 0.5])
    plt.ylim([-1.1, 1.1])
    path_to_save = datapath +os.sep+ f"errors.png"
    plt.savefig(path_to_save, bbox_inches='tight')
    plt.close()
#######################

    for i in range(6):
        #[adapted form https://moonbooks.org/Articles/How-to-calculate-and-plot-a-cumulative-distribution-function-with-matplotlib-in-python-/]
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
        plt.xlabel("absolute error value")
        plt.ylabel("fractions of errors smaller than corresponding value")
        plt.savefig(path_to_save, bbox_inches='tight')
        plt.close()
    
    
    csv_datapath =  datapath +os.sep+ "predictions.csv"
            
    with open(csv_datapath) as f:
        reader = csv.reader(f)
        #next(reader)
        pred_single_category = [[] for i in range(6)]
        for row in reader:
            temp_data = [abs(float(cell)) for cell in row]          
            for i in range(6):
                pred_single_category[i].append(temp_data[i])
    ############################
    f1 = plt.figure()
    for i in range(6):
        data = np.array(pred_single_category[i])
        plt.plot(data)
    plt.title(f'predictions values')
    plt.xlabel("time")
    plt.ylabel("percentage")
    plt.grid()
    #plt.xlim([0, 0.5])
    plt.ylim([0, 1.1])
    path_to_save = datapath +os.sep+ f"predictions.png"
    plt.savefig(path_to_save, bbox_inches='tight')
    plt.close()
    
    
    csv_datapath =  datapath +os.sep+ "90th percentiles.csv"
    percentiles = [[] for i in range(7)]

    with open(csv_datapath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        for i in range(6):
            data_single_category[i] = sorted(data_single_category[i])
            percentile_90 = int(0.9*len(data_single_category[i]))
            percentiles[i] = data_single_category[i][percentile_90]
            
        controllable_load_errors = sorted(controllable_load_errors)
        percentile_90 = int(0.9*len(controllable_load_errors))
        percentiles[6] = controllable_load_errors[percentile_90]
        
        writer.writerow(percentiles)
        
    csv_datapath = datapath.split(os.sep)
    
    csv_datapath = os.sep.join([i for i in csv_datapath])
    csv_datapath = csv_datapath +os.sep+ "hyperparameters.csv" 
    
    with open(csv_datapath) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            hyperparameters = row
        
    hyperparameters.append(loss_function)
    hyperparameters = "; ".join(hyperparameters)
    
    csv_datapath = csv_datapath.split(os.sep)
    for _ in range(2):
        csv_datapath.pop(-1)
    csv_datapath = os.sep.join([i for i in csv_datapath])
    
    csv_datapath = os.path.abspath(os.getcwd()) 
    csv_datapath = csv_datapath +os.sep+ f"percentiles_LSTM_{testing_houses}.csv"
    with open(csv_datapath, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        data_to_write = [hyperparameters]
        for p in percentiles:
            data_to_write.append(p)
        writer.writerow(data_to_write)
        
        
if __name__ == "__main__":
    main()