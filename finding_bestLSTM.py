#this program tests all possibilities of FNNs with 1 hidden layer
#networks are testded and results are saved in csv file

import ann
import itertools
import csv
import time

def write_ann_tested(ann_done, houses):
    
    with open(f'{houses}tested_LSTM_4.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)      
        writer.writerow(ann_done)


def main():
    
    
    number_steps_to_try = [i for i in range(1, 10)] + [10*j for j in range(1,19)]
    #number_steps_to_try = [80, 150, 180]
    #hyperparameters = [96, 61, 126, "tanh", "elu", "relu", "relu"]
    #hyperparameters = [11, 76, 126, "sigmoid", "relu", "tanh", "sigmoid"]
    #hyperparameters = [96, 126, "linear", "relu", "elu"]
    hyperparameters = [11, "relu", "sigmoid"]
    
    LSTMs_to_try = []
    LSTMs_to_try.append([hyperparameters])
    LSTMs_to_try.append(number_steps_to_try)
        
    LSTMs = list(itertools.product(*LSTMs_to_try))
    print(len(LSTMs))

    ################################
    ann_type = "LSTM"               #type of ANN
    train_method = "adam"
    layers = [8]                    #array of numbers of units in each layer 
    house_train = 20                
    house_test = 20
    house_test2 = 100
    files_to_train = [f"total_P_new{house_train}", f"total_Q_new{house_train}", f"decomposition_new{house_train}"]  #files that are used to train ANN
    files_to_test =  [f"total_P_new{house_test}", f"total_Q_new{house_test}", f"decomposition_new{house_test}"]     #files that are used to make predictions
    file_to_write = "predictions"           #file to save predictions
    files_to_test2 = [f"total_P_new{house_test2}", f"total_Q_new{house_test2}", f"decomposition_new{house_test2}"]
    model_to_load = f"test_model{ann_type}" #file to load a model (only relevant if you want to load), make sure you load the same type (e.g. FFN)
    steps = 25                              #number of steps (only relevant for LSTM)
    
    model = ann.ANN(ann_type = ann_type, layers = layers, files_to_train= files_to_train, files_to_test= files_to_test, file_to_write = file_to_write, steps = steps)
    model.add_testing_data(files_to_test2)
    print(f"number of ANNs to test: {len(LSTMs)}")
    all_tests = len(LSTMs)
    tests_done = 0
    
    start = time.time()
    for permutation in LSTMs:
        model.modify_data(permutation[1])
        mse, mse2, custom0, custom1, custom2, custom3, custom4, custom5 = model.create_test_ann(permutation[0], train_method)
        permutation = list(permutation[0]+ [permutation[1]])
        permutation2 = list(permutation)
        
        permutation.append(mse)
        permutation2.append(mse2)
        permutation.append(custom0)
        permutation2.append(custom1)
        
        write_ann_tested(permutation, 20)
        write_ann_tested(permutation2, 100)
        
        print(f"mse for 10 houses = {mse}")
        print(f"mse for 100 houses = {mse2}")
        
        tests_done += 1
        time_elapsed = time.time() - start
        fraction_done = tests_done/all_tests
        fraction_left = 1 - tests_done/all_tests 
        print(f"done {tests_done} out of {all_tests} tests")
        print(f"time elapsed: {int(time_elapsed/60)} minutes; time left: {int(time_elapsed*(fraction_left/fraction_done)/60)} minutes")
        
    end = time.time()

if __name__ == "__main__":
    main()
    